

import os, re, json
from functools import lru_cache
from typing import TypedDict, Optional, Dict, Any, Literal,List

import pandas as pd
from sqlalchemy import create_engine, text

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
import sqlglot
import streamlit as st
from langgraph.checkpoint.memory import MemorySaver
from utils.db_conn import exec_sql
from utils.helpers import df_to_split_payload

import plotly.express as px
import plotly.io as pio
import types
from utils.helpers import split_payload_to_df,make_column_inventory,make_join_hints,clean_sql,validate_sql

#DISALLOWED = re.compile(r"\b(insert|update|delete|drop|alter|create|copy|grant|revoke|truncate|vacuum)\b", re.I)

from utils.agent_nodes import entry_node,decide_after_entry,validate_sql_node,decide_next_after_validate,revise_sql_node,decide_next_after_revise,run_sql_node,decide_next_after_run,done_node,explain_sql,plot_node
from utils.states import AgentState

from functools import partial

# ----------------------------
# Agent builder
# ----------------------------
def build_marketaccess_agent(
    catalog: Dict[str, Any],
    *,
    safe_mode: bool = True,
    default_limit: int = 200,
):
   
    column_inventory = make_column_inventory(catalog)
    join_hints = make_join_hints(catalog)
     
    llm = ChatOpenAI(model=os.getenv("MA_LLM_MODEL1", "gpt-4o"), temperature=0)
    llm_mini = ChatOpenAI(model=os.getenv("MA_LLM_MODEL2", "gpt-4o-mini"), temperature=0,api_key=st.secrets.get("openai_api_key"))


    # -------- Nodes --------
        
    def draft_sql_node(state: AgentState) -> AgentState:
        SYSTEM_SQL = f"""
You are an expert Market Access analyst who writes clean, safe PostgreSQL.

Rules:
- Output ONE SQL query only (no commentary, no code fences).
- Read-only: WITH/SELECT only; never DDL/DML or COPY.
- STRICTLY use only tables and columns that appear in the SCHEMA CATALOG below.
- Case-insensitive drug matching: ALWAYS use ILIKE with % wildcards on brand_name.
  Example: brand_name ILIKE '%dupixent%'

Table selection rules:
- If the user asks for 2025 → use table mapped_access_2025
- If the user asks for 2026 → use table mapped_access_2026
- If the user asks for both or says “compare” → UNION ALL mapped_access_2025 and mapped_access_2026
  and include a literal column year (2025 / 2026) in the SELECT list.
- If no year is specified → default to mapped_access_2026.

Drug filtering rules:
- If multiple drugs are mentioned, use OR conditions on brand_name.
- NEVER assume an exact match; always use ILIKE with %.

Output rules:
- Return rows at the DRUG level (one row per brand_name).
- Include all payer tier and requirement columns present in the table.
- Do NOT aggregate unless the user explicitly asks for counts or summaries.
- Do NOT invent payer names or columns.
- Do NOT join to FAERS tables unless explicitly asked.
- Do NOT rename columns unless requested.

Comparative queries:
- If comparing across years, preserve both rows and distinguish them using a year column.
- Do NOT collapse payer columns into rows unless explicitly asked.

Limits:
- Default LIMIT {default_limit} unless the user asks for more.

Quality rules:
- Keep the query minimal and readable.
- Prefer a single SELECT when possible.
- Ensure SQL syntax is valid and executable.

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
    #graph.add_node("get_drugs",get_drugs)
    graph.add_node("validate_sql", validate_sql_node)
    graph.add_node("revise_sql", revise_sql_node)
    graph.add_node("run_sql", partial(run_sql_node,db_key='marketaccess'))
    graph.add_node("explain_sql",partial(explain_sql,catalog=catalog))
    graph.add_node("plot",plot_node)

    graph.add_node("done", done_node)

    graph.set_entry_point("entry")
    graph.add_conditional_edges("entry", decide_after_entry, {
        "plot": "plot",
        "draft_sql": "draft_sql"
    })
    #graph.add_edge("get_drugs","draft_sql")
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
