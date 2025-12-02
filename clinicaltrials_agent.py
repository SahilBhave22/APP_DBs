
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

DISALLOWED = re.compile(r"\b(insert|update|delete|drop|alter|create|copy|grant|revoke|truncate|vacuum)\b", re.I)

from utils.agent_nodes import entry_node,decide_after_entry,validate_sql_node,decide_next_after_validate,revise_sql_node,decide_next_after_revise,run_sql_node,decide_next_after_run,explain_sql,plot_node,done_node

import streamlit as st
from functools import partial

# ----------------------------
# Agent builder
# ----------------------------
def build_clinicaltrials_agent(
    catalog: Dict[str, Any],
    sample_queries :Dict[str, Any],
    *,
    safe_mode: bool = True,
    default_limit: int = 200,
    
):
    """
    Build an agent that:
      draft_sql -> validate_sql -> (revise_sql)* -> run_sql -> done
    Returns a LangGraph app; final state has keys: sql, df, error.
    """
    print(default_limit)
    column_inventory = make_column_inventory(catalog)
    join_hints = make_join_hints(catalog)

    


    llm = ChatOpenAI(model=os.getenv("AACT_LLM_MODEL1", "gpt-4o"), temperature=0,api_key=st.secrets.get("openai_api_key"))
    llm_mini = ChatOpenAI(model=os.getenv("AACT_LLM_MODEL2", "gpt-4o-mini"), temperature=0,api_key=st.secrets.get("openai_api_key"))

    # -------- Nodes --------
    
    def draft_sql_node(state: AgentState) -> AgentState:
        print(state.get('drugs'))
        SYSTEM_SQL = f"""You are an expert ClinicalTrials.gov (AACT) analyst who writes clean, safe PostgreSQL.

Rules:
- Output ONE SQL query only (no commentary, no code fences).
- Read-only: WITH/SELECT only; never DDL/DML or COPY.
- ALWAYS CHECK data types of columns BEFORE joining tables.
- ALWAYS CHECK which columns are asked by the use and return all of them.
- ALWAYS CAST ALL types of id to varchar.
- Always inline all literal values directly in the SQL.
- Never use parameters or placeholders of any kind (no :param, @param, ?, $1, etc.).
- If the question lists multiple drug names, inline them in the SQL using an IN 
- ALWAYS USE public.drug_trials TABLE TO GET TRIAL IDS FOR A PARTICULAR DRUG. 
- DO NOT USE PRO related tables unless user explicitly mentions.
- Guidelines for endpoint/ outcome related queries
    - Endpoints mean outcomes.
    - BE very careful to check whether design outcomes or actual outcomes are asked.
    - DO NOT RETURN outcome titles, ALWAYS USE public.drug_trial_outcome_categories table.
    - To identify primary/secondary/other endpoints, ALWAYS USE either ctgov.outcomes.outcome_type or ctgov.design_outcomes.outcome_type.
    - IF outcome/ endpoint categories are asked, ALWAYS USE public.drug_trial_outcome_categories table.
    - USE PRO related tables only if the user asks for PRO or patient reported outcomes.
- Guidelines for PRO related queries:
    - ALWAYS USE public.drug_trial_outcomes_pro TABLE to get trial ids and outcome ids for patient reported outcomes (PRO) measures.
    - ALWAYS USE public.domain_score_match to get PRO domain or sub-scale information.
    - ALWAYS JOIN public.drug_trial_outcomes_pro using nct_id AND outcome_id BOTH
- ONLY USE FROM POSSIBLE VALUES GIVEN IN COLUMN DESCRIPTION
- MAKE SURE THE QUERY HAS CORRECT POSTGRESQL SYNTAX
- Use only tables/columns that appear in the SCHEMA CATALOG below.
- ALWAYS take counts of unique nct ids.
- Case-insensitive filters: use ILIKE for text.
Guidance for comparative queries
    - IF comparsion among 2 drugs is asked, make sure to add a stratification by the brand_name too.
    - Always compute per-group aggregates and counts should always be of nct_id.
    - Use a window function which partitions by whatever groups are required and sorts it by count.
    - ALWAYS Filter with WHERE rank_col <= N to return top-N per group.
    - Do not use a global LIMIT N, since that only applies across the whole result set.
    - Preserve deduplication (e.g., COUNT(DISTINCT nct_id)) as usual.
- Default LIMIT {default_limit} unless the user asks for more.
- Keep the query readable and minimal (CTEs encouraged).
- DO NOT AGGREGATE or AVERAGE any values unless specifically asked.

SCHEMA CATALOG:
{json.dumps(catalog)}


SAMPLE QUERIES for guidance: 
{json.dumps(sample_queries)}

- DO NOT CHANGE SAMPLE QUERIES, USE IT AS IT IS.

Validator feedback:
{{feedback}}

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
    graph.add_node("run_sql", partial(run_sql_node,db_key='aact'))
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
