

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
def build_fdaers_agent(
    catalog: Dict[str, Any],
    *,
    safe_mode: bool = True,
    default_limit: int = 200,
):
   
    column_inventory = make_column_inventory(catalog)
    join_hints = make_join_hints(catalog)
     
    llm = ChatOpenAI(model=os.getenv("FAERS_LLM_MODEL1", "gpt-4o"), temperature=0)
    llm_mini = ChatOpenAI(model=os.getenv("FAERS_LLM_MODEL2", "gpt-4o-mini"), temperature=0,api_key=st.secrets.get("openai_api_key"))


    # -------- Nodes --------
        
    def draft_sql_node(state: AgentState) -> AgentState:
        SYSTEM_SQL = f"""You are an expert FAERS analyst who writes clean, safe PostgreSQL.

Rules:
- Output ONE SQL query only (no commentary, no code fences).
- Read-only: WITH/SELECT only; never DDL/DML or COPY.
- STRICTLY Use only tables/columns that appear in the SCHEMA CATALOG below.
- Deduplicate at the report level with COUNT(DISTINCT demo.primaryid) or COUNT(DISTINCT drug_cases.primaryid).
- ALWAYS use table `drug_cases` to get relevant primary ids for a specific brand name 
- Prefer joins:
  - demo.primaryid <-> drug_cases.primaryid
  - indi.(primaryid, indi_drug_seq) <-> drug_cases.(primaryid, drug_seq)
  - reac.primaryid <-> demo.primaryid
- Case-insensitive filters: use ILIKE for text.
- For a drugs list, ALWAYS give data segregated by brand name.
- DO NOT USE ROR tables UNLESS USER SPECIFICALLY ASKS. DEFAULT CHOICE SHOULD BE COUNT(DISTINCT demo.primaryid)).
- Guidance for comparative queries
    - IF comparsion among 2 drugs is asked, make sure to add a stratification by the brand_name too.
    - Always compute per-group aggregates.
    - Use a window function which partitions by whatever groups are required and sorts it by count.
    - Filter with WHERE rank_col <= N to return top-N per group.
    - Do not use a global LIMIT N, since that only applies across the whole result set.
    - Preserve deduplication (e.g., COUNT(DISTINCT demo.primaryid)) as usual.
- When SOC/ Organ system level counts are asked, ALWAYS GROUP BY PT and take sum for all PTs within every SOC.
- For SOC/ Organ System level counts, DO NOT GROUP BY SOC, always group by PT and take summation.
- Guidance for ROR / Signal analysis queries
    - ALWAYS USE public.top_n_ror table for signal strength analysis.
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
    #graph.add_node("get_drugs",get_drugs)
    graph.add_node("validate_sql", validate_sql_node)
    graph.add_node("revise_sql", revise_sql_node)
    graph.add_node("run_sql", partial(run_sql_node,db_key='fdaers'))
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
