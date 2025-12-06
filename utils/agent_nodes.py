import os, re, json
from functools import lru_cache
from typing import TypedDict, Optional, Dict, Any, Literal,List

import pandas as pd

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
from utils.states import AgentState
from utils.prompts import SYSTEM_EXPLAIN, SYSTEM_REVISE
import streamlit as st

DISALLOWED = re.compile(r"\b(insert|update|delete|drop|alter|create|copy|grant|revoke|truncate|vacuum)\b", re.I)

llm = ChatOpenAI(model="gpt-4o", temperature=0,api_key=st.secrets.get("openai_api_key"))
llm_mini = ChatOpenAI(model = "gpt-4o-mini", temperature=0,api_key=st.secrets.get("openai_api_key"))

def entry_node(state: AgentState) -> AgentState:
    return {}
    
def decide_after_entry(state: AgentState) -> Literal["draft_sql", "plot"]:
    # Up to 2 repairs; then try to run (or fail in run_sql if still invalid)
    if state["call_source"] == "chart_toggle":
        return "plot"
    return "draft_sql"


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


def run_sql_node(state: AgentState,db_key:str,safe_mode=True) -> AgentState:
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
            state["df"] = df_to_split_payload(exec_sql(sql,db_key=db_key))
            state["error"] = None
        except Exception as e:
            state["error"] = str(e)
        return state
    
def decide_next_after_run(state: AgentState) -> Literal["revise_sql", "done"]:
    # Up to 2 repairs; then try to run (or fail in run_sql if still invalid)
    if state["error"] and state["attempts"] < 4:
        print("revised")
        return "revise_sql"
    return "explain"

def explain_sql(state: AgentState,catalog)-> AgentState:
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
        state['chart_error'] = "No data available"
        return state
        
    fmeta = meta(df_to_plot)
    preview = df_to_plot.head(6).to_string(index=False)

    prompt = f"""
    You produce minimal Plotly Express code from a pandas DataFrame to visualize the user's question.

    Rules:
    - Use ONLY: df (pandas.DataFrame to plot), px (plotly.express), go (plotly.graph_objects optional).
    - ALL IMPORTS ARE ALREADY PRESENT, DO NOT ADD IMPORT STATEMENTS TO THE CODE.
    - DO NOT add any imports, files, network, or system access statements
    - End with a variable named: fig
    - For any datetime values always convert the column using pd.to_datetime.
    - Use clear defaults: informative title, axis labels, and template='plotly_white'.
    - Time trend -> line/area; categorical comparison -> bar; distribution -> histogram/box/violin; heatmap for matrix-like comparisons.
    - Use ONLY columns that exist in df.
    - For PRICING related questions, use a STEP line (Plotly line_shape='hv') instead of slope lines.
    - Always show difference in drug names provided in the drugs list.
    
    User question:
    {question}

    DF preview (first 6 rows):
    {preview}

    FAERS columns: {fmeta['columns']}

    List of drugs in the data: {state["drugs"]}
    Return ONLY Python code that uses df and px (and optionally go) and ends with:
    fig = <plotly figure>
    """
    code = llm.invoke(prompt).content.strip()
    if code.startswith("```"):
        code = code.strip("` \n")
        if "\n" in code and code.split("\n", 1)[0].lower().startswith("python"):
            code = code.split("\n", 1)[1]

    try:
        print(code)
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

