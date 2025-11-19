
import os, re, json
from functools import lru_cache
from typing import TypedDict, Optional, Dict, Any, Literal

import pandas as pd
from sqlalchemy import create_engine, text

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
import sqlglot

from utils.db_conn import exec_sql
from utils.helpers import df_to_split_payload

from langgraph.checkpoint.memory import MemorySaver

import plotly.express as px
import plotly.io as pio
import types
from utils.helpers import split_payload_to_df,make_column_inventory,make_join_hints,clean_sql,validate_sql

# ----------------------------
# Helpers & validation
# ----------------------------
# def make_column_inventory(catalog: Dict[str, Any]) -> str:
#     lines = []
#     for t in catalog.get("tables", []):
#         cols = ", ".join([c["name"] for c in t.get("columns", [])])
#         lines.append(f"- {t['name']}: {cols}")
#     return "\n".join(lines) if lines else "None"

# def make_join_hints(catalog: Dict[str, Any]) -> str:
#     rels = catalog.get("relationships", [])
#     if not rels:
#         return "None"
#     return "\n".join([
#         f"- {r['from_table']}.{','.join(r['from_columns'])} ↔ {r['to_table']}.{','.join(r['to_columns'])} [{r.get('type','')}]"
#         for r in rels
#     ])

# def clean_sql(sql: str) -> str:
#     """Strip ```sql fences and trim."""
#     if not sql:
#         return ""
#     s = sql.strip()
#     s = re.sub(r"^```[a-zA-Z]*\s*", "", s)  # remove opening ``` / ```sql
#     s = re.sub(r"\s*```$", "", s)           # remove trailing ```
#     return s.strip()

DISALLOWED = re.compile(r"\b(insert|update|delete|drop|alter|create|copy|grant|revoke|truncate|vacuum)\b", re.I)

# def validate_sql(sql: str) -> Optional[str]:
#     s = (sql or "").strip()
#     if not re.match(r"^\s*(with|select)\b", s, flags=re.I | re.S):
#         return "Only WITH/SELECT queries are allowed."
#     if DISALLOWED.search(s):
#         return "Disallowed SQL keyword detected."
#     # try:
#     #     parsed = sqlglot.parse_one(s, read="postgres")
#     # except Exception as e:
#     #     return f"SQL parse error: {e}"
#     # if parsed is None:
#     #     return "Empty or unparsable SQL."
#     return None


# ----------------------------

# Agent State
# ----------------------------
class AgentState(TypedDict):
    question: str
    sql: Optional[str]
    error: Optional[str]
    attempts: int
    summary: Optional[str]
    df:  Optional[Dict[str, Any]]
    sql_explain : Optional[str]
    want_chart : bool
    figure_json: Optional[str]
    call_source: Optional[str]


# ----------------------------
# DB helpers
# ----------------------------
# @lru_cache(maxsize=4)
# def get_engine_cached(fdaers_db_url: str):
#     return create_engine(fdaers_db_url, pool_pre_ping=True)

# def exec_sql(fdaers_db_url: str, sql: str) -> pd.DataFrame:
#     eng = get_engine_cached(fdaers_db_url)
#     with eng.connect() as conn:
#         return pd.read_sql(text(sql), conn)


# ----------------------------
# Agent builder
# ----------------------------
def build_pricing_agent(
    catalog: Dict[str, Any],
    *,
    safe_mode: bool = True,
    default_limit: int = 200,
):
    """
    Build an agent that:
      draft_sql -> validate_sql -> (revise_sql)* -> run_sql -> done
    Returns a LangGraph app; final state has keys: sql, df, error.
    """
    
    #column_inventory = make_column_inventory(catalog)
    #join_hints = make_join_hints(catalog)

    SYSTEM_SQL = f"""You are an expert PICING analyst who writes clean, safe PostgreSQL.

Rules:
- Output ONE SQL query only (no commentary, no code fences).
- Read-only: WITH/SELECT only; never DDL/DML or COPY.
- STRICTLY Use only tables/columns that appear in the SCHEMA CATALOG below.
- BE VERY CAREFUL AND CHECK WHETHER DATA IS ASKED FOR A BRAND NAME OR DRUG CLASS
- TO get drugs associated with a drug class, ALWAYS USE drug_classes TABLE and use ILIKE for filters.
- DO NOT USE atc_code as filter, ALWAYS USE atc_class_name
- Always use ilike on drug_classes.brand_name column to join drug_classes table
- ALWAYS use ilike with string matching like "%brand name%" to match brand names.
- DO NOT TAKE YEAR or MONTH level aggregates unless user specifically asks.
- Always use distinct keyword for the final subquery, we dont want duplicate rows
- Default LIMIT {default_limit} unless the user asks for more.
- Keep the query readable and minimal (CTEs encouraged).
- MAKE SURE THAT THE QUERY SYNTAX IS ACCURATE AND VALUES REFERENCED IN sub queries are used correctly

SCHEMA CATALOG:
{json.dumps(catalog)}

"""

    SYSTEM_REVISE = f"""You are repairing a PostgreSQL query to satisfy validator feedback.

- Keep WITH/SELECT only (no DDL/DML).

- Preserve user intent; keep :snake_case parameters; end with ONE SQL query only.

Validator feedback:
{{feedback}}
"""
    
    SYSTEM_EXPLAIN = f""" You are a SQL expert that is analysing a SQL query based on a question.
    Here is the question asked: {{question}}
    Here is the SQL query : {{sql}}
    Here is the database info: {{schema_catalog}}

    Provide a clear explanation of the query that is suitable for a semi-technical audience.
    Explain in detail which tables are used, how they are joined and how the counts are taken.
    Also explain what the tables mean and why these particular tables were used.
     
    STRICTLY limit your response to 150 words
    """

    llm = ChatOpenAI(model=os.getenv("PRICING_LLM_MODEL1", "gpt-4o"), temperature=0)
    llm_mini = ChatOpenAI(model=os.getenv("PRICING_LLM_MODEL2", "gpt-4o-mini"), temperature=0)


    # -------- Nodes --------
    def entry_node(state: AgentState) -> AgentState:
        return {}
    
    def decide_after_entry(state: AgentState) -> Literal["draft_sql", "plot"]:
        # Up to 2 repairs; then try to run (or fail in run_sql if still invalid)
        if state["call_source"] == "chart_toggle":
            return "plot"
        return "draft_sql"
    
    def draft_sql_node(state: AgentState) -> AgentState:
        msgs = [SystemMessage(SYSTEM_SQL), HumanMessage(state["question"])]
        raw_sql = llm.invoke(msgs).content.strip()
        state["sql"] = clean_sql(raw_sql)
        return state

    def validate_sql_node(state: AgentState) -> AgentState:
        state["sql"] = clean_sql(state["sql"])
        state["error"] = validate_sql(state["sql"] or "")
        return state

    def decide_next_after_validate(state: AgentState) -> Literal["revise_sql", "run_sql"]:
        return "revise_sql" if state["error"] else "run_sql"

    def revise_sql_node(state: AgentState) -> AgentState:
        state["attempts"] += 1
        msgs = [
            SystemMessage(SYSTEM_REVISE.format(feedback=state["error"])),
            HumanMessage(state["sql"] or "")
        ]
        raw_sql = llm.invoke(msgs).content.strip()
        state["sql"] = clean_sql(raw_sql)
        state["error"] = validate_sql(state["sql"] or "")
        return state

    def decide_next_after_revise(state: AgentState) -> Literal["revise_sql", "run_sql"]:
        # Up to 2 repairs; then try to run (or fail in run_sql if still invalid)
        if state["error"] and state["attempts"] < 2:
            return "revise_sql"
        return "run_sql"

    def run_sql_node(state: AgentState) -> AgentState:
        sql = (state["sql"] or "").strip()
        if not sql:
            state["error"] = "No SQL to execute."
            return state

        # Enforce validator in safe_mode; otherwise still block obvious DDL/DML
        if safe_mode:
            v = validate_sql(sql)
            if v:
                state["error"] = v
                return state
        else:
            if DISALLOWED.search(sql):
                state["error"] = "Refusing to run non-SELECT/DDL statements."
                return state

        try:
            state["df"] = df_to_split_payload(exec_sql(sql,db_key="pricing"))
        except Exception as e:
            state["error"] = str(e)
        return state

    def decide_next_after_run(state: AgentState) -> Literal["revise_sql", "done"]:
        # Up to 2 repairs; then try to run (or fail in run_sql if still invalid)
        if state["error"] and state["attempts"] < 2:
            return "revise_sql"
        return "explain"
    
    def explain_sql(state: AgentState)-> AgentState:
        msgs = [
            SystemMessage(SYSTEM_EXPLAIN.format(question=state["question"], sql = state["sql"], 
                          schema_catalog = json.dumps(catalog)))
        ]
        explain = llm_mini.invoke(msgs).content.strip()
        
        state["sql_explain"] = explain
        return state
    
    def plot_node(state: AgentState) -> AgentState:
        
        want_chart = state.get('want_chart')
        if not want_chart:
            return {}
   
        llm = ChatOpenAI(model="gpt-4o", temperature=0)

        def meta(df: Optional[pd.DataFrame]):
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                return {
                    "rows": len(df),
                    "cols": df.shape[1],
                    "columns": [str(c) for c in df.columns][:50],
                    "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
                }
            return {"rows": 0, "cols": 0, "columns": [], "dtypes": {}}

        
        def run_safely(df, code: str):
            # basic static hardening
            banned = [
                "__import__", "open(", "exec(", "eval(", "subprocess", "os.", "sys.",
                "requests", "pickle", "shutil", "pathlib", "socket", "import "
                "pandas.read_", "to_csv", "to_excel", "plotly.io.write", "fig.write"
            ]
            low = code.lower()
            if any(b in low for b in banned):
                #print(low)
                raise ValueError("Unsafe code detected in chart snippet.")
            import plotly.graph_objects as go  # allowed
            safe_globals = {
                "__builtins__": types.MappingProxyType({}),  # strip builtins
                "px": px, "go": go, "pd":pd
            }
            safe_locals = {"df": df}
            exec(code, safe_globals, safe_locals)  # code must set 'fig'
            fig = safe_locals.get("fig")
            if fig is None:
                raise ValueError("The snippet did not create a variable named 'fig'.")
            return fig

        question = state.get("question") or ""
        df_to_plot = split_payload_to_df(state.get("df"))
        if not isinstance(df_to_plot, pd.DataFrame) or len(df_to_plot) == 0:
            state['chart_error'] = "No data available for plotting from PRICING"
            return state
            
        pmeta = meta(df_to_plot)
        preview = df_to_plot.head(6).to_string(index=False)

        prompt = f"""
    You produce minimal Plotly Express code from a pandas DataFrame to visualize the user's question.

    Rules:
    - Use ONLY: df (pandas.DataFrame to plot), px (plotly.express), go (plotly.graph_objects optional).
    - ALL IMPORTS ARE ALREADY PRESENT, DO NOT ADD IMPORT STATEMENTS TO THE CODE.
    - DO NOT add any imports, files, network, or system access statements
    - End with a variable named: fig
    - Use clear defaults: informative title, axis labels, and template='plotly_white'.
    - Time trend -> line/area; categorical comparison -> bar; distribution -> histogram/box/violin; heatmap for matrix-like comparisons.
    - Use ONLY columns that exist in df.
    - For PRICING related questions, use a STEP line (Plotly line_shape='hv') instead of slope lines.

    User question:
    {question}

    DF preview (first 6 rows):
    {preview}

    PRICING columns: {pmeta['columns']}
    Return ONLY Python code that uses df and px (and optionally go) and ends with:
    fig = <plotly figure>
    """
        code = llm.invoke(prompt).content.strip()
        if code.startswith("```"):
            code = code.strip("` \n")
            if "\n" in code and code.split("\n", 1)[0].lower().startswith("python"):
                code = code.split("\n", 1)[1]

        try:
            fig = run_safely(df_to_plot, code)
            fig.update_layout(template="plotly_white")
            state["figure_json"] = fig.to_json()
            return state
        except Exception as e:
            state["chart_error"]= f"Chart generation failed: {e}"
            return state

    def done_node(state: AgentState) -> AgentState:
        # Nothing to do; terminal summarization could go here if desired.
        return state

    # -------- Graph wiring --------
    graph = StateGraph(AgentState)
    graph.add_node("entry",entry_node)
    graph.add_node("draft_sql", draft_sql_node)
    graph.add_node("validate_sql", validate_sql_node)
    graph.add_node("revise_sql", revise_sql_node)
    graph.add_node("run_sql", run_sql_node)
    graph.add_node("explain_sql",explain_sql)
    graph.add_node("plot",plot_node)
    graph.add_node("done", done_node)

    graph.set_entry_point("entry")
    graph.add_conditional_edges("entry", decide_after_entry, {
        "plot": "plot",
        "draft_sql": "draft_sql",
    })
    graph.add_edge("draft_sql", "validate_sql")
    graph.add_conditional_edges("validate_sql", decide_next_after_validate, {
        "revise_sql": "revise_sql",
        "run_sql": "run_sql",
    })
    graph.add_conditional_edges("revise_sql", decide_next_after_revise, {
        "revise_sql": "revise_sql",
        "run_sql": "run_sql",
    })
    graph.add_conditional_edges("run_sql", decide_next_after_run, {
        "revise_sql": "revise_sql",
        "explain": "explain_sql",
    })
    
    graph.add_edge("explain_sql", "plot")
    graph.add_edge("plot", "done")


    graph.add_edge("done", END)

    app = graph.compile(checkpointer=MemorySaver())
    return app
