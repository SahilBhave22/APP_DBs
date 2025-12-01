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


DISALLOWED = re.compile(r"\b(insert|update|delete|drop|alter|create|copy|grant|revoke|truncate|vacuum)\b", re.I)

llm = ChatOpenAI(model=os.getenv("AACT_LLM_MODEL1", "gpt-4o"), temperature=0)
llm_mini = ChatOpenAI(model=os.getenv("AACT_LLM_MODEL2", "gpt-4o-mini"), temperature=0)

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


def run_sql_node(state: AgentState,safe_mode=True) -> AgentState:
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
            state["df"] = df_to_split_payload(exec_sql(sql,db_key='aact'))
            state["error"] = None
        except Exception as e:
            state["error"] = str(e)
        return state
    
def decide_next_after_run(state: AgentState) -> Literal["revise_sql", "done"]:
    # Up to 2 repairs; then try to run (or fail in run_sql if still invalid)
    if state["error"] and state["attempts"] < 4:
        return "revise_sql"
    return "explain"

# def explain_sql(state: AgentState,catalog)-> AgentState:
#     msgs = [
#         SystemMessage(SYSTEM_EXPLAIN.format(question=state["question"], sql = state["sql"], 
#                         schema_catalog = json.dumps(catalog)))
#     ]
#     explain = llm_mini.invoke(msgs).content.strip()
    
#     state["sql_explain"] = explain
#     return state


def done_node(state: AgentState) -> AgentState:
        # Nothing to do; terminal summarization could go here if desired.
        return state

