# agent.py — LangGraph/validator wrapper WITH DB execution
# Adds an explicit `run_sql` node that executes the query, then goes to `done`.

import os, re, json
from functools import lru_cache
from typing import TypedDict, Optional, Dict, Any, Literal

import pandas as pd
from sqlalchemy import create_engine, text

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
import sqlglot

from langgraph.checkpoint.memory import MemorySaver
from utils.db_conn import exec_sql
from utils.helpers import df_to_split_payload


# ----------------------------
# Helpers & validation
# ----------------------------
def make_column_inventory(catalog: Dict[str, Any]) -> str:
    lines = []
    for t in catalog.get("tables", []):
        cols = ", ".join([c["name"] for c in t.get("columns", [])])
        lines.append(f"- {t['name']}: {cols}")
    return "\n".join(lines) if lines else "None"

def make_join_hints(catalog: Dict[str, Any]) -> str:
    rels = catalog.get("relationships", [])
    if not rels:
        return "None"
    return "\n".join([
        f"- {r['from_table']}.{','.join(r['from_columns'])} ↔ {r['to_table']}.{','.join(r['to_columns'])} [{r.get('type','')}]"
        for r in rels
    ])

def clean_sql(sql: str) -> str:
    """Strip ```sql fences and trim."""
    if not sql:
        return ""
    s = sql.strip()
    s = re.sub(r"^```[a-zA-Z]*\s*", "", s)  # remove opening ``` / ```sql
    s = re.sub(r"\s*```$", "", s)           # remove trailing ```
    return s.strip()

DISALLOWED = re.compile(r"\b(insert|update|delete|drop|alter|create|copy|grant|revoke|truncate|vacuum)\b", re.I)

def validate_sql(sql: str) -> Optional[str]:
    s = (sql or "").strip()
    if not re.match(r"^\s*(with|select)\b", s, flags=re.I | re.S):
        return "Only WITH/SELECT queries are allowed."
    if DISALLOWED.search(s):
        return "Disallowed SQL keyword detected."
    # try:
    #     parsed = sqlglot.parse_one(s, read="postgres")
    # except Exception as e:
    #     return f"SQL parse error: {e}"
    # if parsed is None:
    #     return "Empty or unparsable SQL."
    return None


# ----------------------------

# Agent State
# ----------------------------
class AgentState(TypedDict):
    question: str
    sql: Optional[str]
    error: Optional[str]
    attempts: int
    summary: Optional[str]
    df: Optional[Dict[str, Any]]
    sql_explain : Optional[str]



# ----------------------------
# Agent builder
# ----------------------------
def build_fdaers_agent(
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
    column_inventory = make_column_inventory(catalog)
    join_hints = make_join_hints(catalog)

    SYSTEM_SQL = f"""You are an expert FAERS analyst who writes clean, safe PostgreSQL.

Rules:
- Output ONE SQL query only (no commentary, no code fences).
- Read-only: WITH/SELECT only; never DDL/DML or COPY.
- STRICTLY Use only tables/columns that appear in the SCHEMA CATALOG below.
- Deduplicate at the report level with COUNT(DISTINCT demo.primaryid) or COUNT(DISTINCT drug_cases.primaryid).
- Prefer joins:
  - demo.primaryid ↔ drug_cases.primaryid
  - indi.(primaryid, indi_drug_seq) ↔ drug_cases.(primaryid, drug_seq)
  - reac.primaryid ↔ demo.primaryid
- Case-insensitive filters: use ILIKE for text.
- BE VERY CAREFUL AND CHECK WHETHER DATA IS ASKED FOR A BRAND NAME OR DRUG CLASS
- TO get drugs associated with a drug class, ALWAYS USE drug_classes TABLE and use ILIKE for filters.
- DO NOT USE atc_code as filter, ALWAYS USE atc_class_name
- For all drugs within a class, ALWAYS give data segregated by brand name.
- Guidance for comparative queries
    - IF comparsion among 2 drugs is asked, make sure to add a stratification by the brand_name too.
    - Always compute per-group aggregates.
    - Use a window function which partitions by whatever groups are required and sorts it by count.
    - Filter with WHERE rank_col <= N to return top-N per group.
    - Do not use a global LIMIT N, since that only applies across the whole result set.
    - Preserve deduplication (e.g., COUNT(DISTINCT demo.primaryid)) as usual.
- When SOC/ Organ system level counts are asked, ALWAYS GROUP BY PT and take sum for all PTs within every SOC.
- For SOC/ Organ System level counts, DO NOT GROUP BY SOC, always group by PT and take summation.
- DO NOT USE ROR tables UNLESS USER SPECIFICALLY ASKS. DEFAULT CHOICE SHOULD BE COUNT(DISTINCT demo.primaryid)).
- Guidance for ROR / Signal analysis queries
    - ALWAYS USE public.top_n_ror table for signal strength analysis.
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

    llm = ChatOpenAI(model=os.getenv("FAERS_LLM_MODEL1", "gpt-4o"), temperature=0)
    llm_mini = ChatOpenAI(model=os.getenv("FAERS_LLM_MODEL2", "gpt-4o-mini"), temperature=0)


    # -------- Nodes --------
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
            state["df"] = df_to_split_payload(exec_sql(sql,db_key="fdaers"))
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

    def done_node(state: AgentState) -> AgentState:
        # Nothing to do; terminal summarization could go here if desired.
        return state

    # -------- Graph wiring --------
    graph = StateGraph(AgentState)
    graph.add_node("draft_sql", draft_sql_node)
    graph.add_node("validate_sql", validate_sql_node)
    graph.add_node("revise_sql", revise_sql_node)
    graph.add_node("run_sql", run_sql_node)
    graph.add_node("explain_sql",explain_sql)
    graph.add_node("done", done_node)

    graph.set_entry_point("draft_sql")
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
    
    graph.add_edge("explain_sql", "done")

    graph.add_edge("done", END)
    app = graph.compile(checkpointer=MemorySaver())
    return app
