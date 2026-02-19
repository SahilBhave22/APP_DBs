
import os, re, json
from functools import lru_cache
from typing import TypedDict, Optional, Dict, Any, Literal,List

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
from utils.agent_nodes import entry_node,decide_after_entry,validate_sql_node,decide_next_after_validate,revise_sql_node,decide_next_after_revise,run_sql_node,decide_next_after_run,explain_sql,plot_node,done_node
from utils.states import AgentState
from utils.prompts import SYSTEM_EXPLAIN,SYSTEM_REVISE
import streamlit as st
from functools import partial

DISALLOWED = re.compile(r"\b(insert|update|delete|drop|alter|create|copy|grant|revoke|truncate|vacuum)\b", re.I)


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

   
    llm = ChatOpenAI(model=os.getenv("PRICING_LLM_MODEL1", "gpt-4.1"), temperature=0,api_key=st.secrets.get("openai_api_key"))
    llm_mini = ChatOpenAI(model=os.getenv("PRICING_LLM_MODEL2", "gpt-4.1-mini"), temperature=0,api_key=st.secrets.get("openai_api_key"))


    # # -------- Nodes --------
    
    def draft_sql_node(state: AgentState) -> AgentState:
        SYSTEM_SQL = f"""You are an expert PICING analyst who writes clean, safe PostgreSQL.

Rules:
- Output ONE SQL query only (no commentary, no code fences).
- Read-only: WITH/SELECT only; never DDL/DML or COPY.
- STRICTLY Use only tables/columns that appear in the SCHEMA CATALOG below.
- ALWAYS use ilike with string matching like "%brand name%" to match brand names.
- DO NOT TAKE YEAR or MONTH level aggregates unless user specifically asks.
- Always use distinct keyword for the final subquery, we dont want duplicate rows
- Default LIMIT {default_limit} unless the user asks for more.
- Keep the query readable and minimal (CTEs encouraged).
- MAKE SURE THAT THE QUERY SYNTAX IS ACCURATE AND VALUES REFERENCED IN sub queries are used correctly

SCHEMA CATALOG:
{json.dumps(catalog)}

"""
        msgs = [SystemMessage(SYSTEM_SQL), HumanMessage(state["question"])]
        raw_sql = llm.invoke(msgs).content.strip()
        state["sql"] = clean_sql(raw_sql)
        return state

    
    # -------- Graph wiring --------
    graph = StateGraph(AgentState)
    graph.add_node("entry",entry_node)
    graph.add_node("draft_sql", draft_sql_node)
    graph.add_node("validate_sql", validate_sql_node)
    graph.add_node("revise_sql", revise_sql_node)
    graph.add_node("run_sql", partial(run_sql_node,db_key='pricing'))
    graph.add_node("explain_sql",partial(explain_sql,catalog=catalog))
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
