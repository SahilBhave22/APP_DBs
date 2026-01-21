
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
    #print(default_limit)
    column_inventory = make_column_inventory(catalog)
    join_hints = make_join_hints(catalog)

    


    llm = ChatOpenAI(model=os.getenv("AACT_LLM_MODEL1", "gpt-4.1"), temperature=0,api_key=st.secrets.get("openai_api_key"))
    llm_mini = ChatOpenAI(model=os.getenv("AACT_LLM_MODEL2", "gpt-4.1-mini"), temperature=0,api_key=st.secrets.get("openai_api_key"))
    llm_max = ChatOpenAI(model=os.getenv("AACT_LLM_MODEL4", "gpt-4.1-mini"), temperature=0,api_key=st.secrets.get("openai_api_key"))
    # -------- Nodes --------
    def extract_trial_scope_node(state: AgentState,catalog) -> AgentState:
        """
        Dynamically extract explicit trial-level scope constraints
        from the user question and merge them into active_trial_scope.

        active_trial_scope is a LIST of constraint dicts:
        [
        { "field": str, "operator": str, "value": Any }
        ]
        """

        llm = llm_mini 
        
        prev_scope = state.get("active_trial_scope", [])
        
        if not isinstance(prev_scope, list):
            prev_scope = []

        prompt = f"""
    You are Clinical Trial expert extracting EXPLICIT clinical trial scope constraints
    from a user question.

    Refer to the Schema Catalog for information on the database.
    - Use only tables/columns that appear in the SCHEMA CATALOG below.

    ONLY extract constraints that are CLEARLY and EXPLICITLY stated.

    Carefully check which column from the database is being referenced by the user question.

    Each constraint MUST be returned as:
    - field: a canonical database-level concept 
    - operator: one of (=, !=, >, >=, <, <=, IN)
    - value: a string, number, or list

    DO NOT INCLUDE ANY scope constraints on 'brand_name'
    If NO explicit scope constraints are present, return an empty list [].

    User question:
    {state["question"]}

    Database Schema Catalog: {json.dumps(catalog)}

    Return STRICT JSON ONLY as a LIST:
    [
    {{
        "field": "<string>",
        "operator": "<string>",
        "value": <any>
    }}
    ]
    """

        resp = llm.invoke(prompt).content
        resp = re.sub(r"^```(?:json)?|```$", "", resp.strip())

        try:
            extracted = json.loads(resp)
            if not isinstance(extracted, list):
                extracted = []
        except Exception:
            extracted = []

       
        # --- Merge dynamically (monotonic, field-based overwrite) ---
        merged = { (c["field"]): c for c in prev_scope if "field" in c }

        for c in extracted:
            if not all(k in c for k in ("field", "operator", "value")):
                continue
            merged[c["field"]] = c  # overwrite only same-field constraint

        state["active_trial_scope"] = list(merged.values())
        
        return state


    def classify_trial_stage_node(state: AgentState) -> AgentState:
        """
        Classify the clinical trials query into:
        - overview
        - trial_details
        - results
        """

          # cheap + deterministic

        classify_prompt = f"""
You are a ClinicalTrials.gov (AACT) expert whose task is to classify
the INTENT of a user question about clinical trials.

You MUST determine the DATA GRANULARITY being requested.

Return ONLY ONE of the following labels (lowercase, exact match):
- overview
- trial_details
- results

--------------------
IMPORTANT EXAMPLES
--------------------

"Give me clinical trial data for <drug>" → overview
"Clinical trials for <drug>" → overview
"How many trials exist for <drug>?" → overview

"Show trial arms for <drug>" → trial_details
"Eligibility criteria for <drug> trials" → trial_details

"Primary outcomes for <drug> trials" → results
"What did the <drug> show?" → results
"Safety results in <drug> trials" → results

--------------------
INTENT DEFINITIONS (FOR REFERENCE)
--------------------

overview:
High-level, aggregated, or unspecified trial information.
This includes:
- "clinical trial data"
- "trial data"
- "clinical trials for <drug>"
- counts, summaries, distributions

trial_details:
Structural trial-level information (no outcome values).

results:
Outcome-level or value-level trial results.

--------------------
HARD PRECEDENCE RULES (MUST FOLLOW)
--------------------

RULE 1 — RESULTS ARE EXPLICIT ONLY:
You may return "results" ONLY IF the user EXPLICITLY asks for:
- results
- outcomes
- endpoints
- efficacy
- safety
- adverse events
- comparisons between arms
- outcome values

If NONE of the above are explicitly mentioned,
you MUST NOT return "results".

--------------------

RULE 2 — TRIAL DETAILS ARE STRUCTURAL:
Return "trial_details" ONLY IF the user EXPLICITLY asks for:
- arms
- interventions
- design groups
- eligibility
- inclusion/exclusion
- dosing arms
- randomization
- trial design

If NONE of the above are explicitly mentioned,
you MUST NOT return "trial_details".

DO NOT interpret the phrases "clinical trial data for <drug>" as a trial details requirement.

--------------------

RULE 3 — DEFAULT TO OVERVIEW:
If the question does NOT meet the explicit criteria
for "results" or "trial_details",
you MUST return "overview".



--------------------
USER QUESTION
--------------------
{state["question"]}

Return ONLY the label.
"""



        resp = llm_max.invoke(classify_prompt).content.strip().lower()
        print(resp)
        if resp not in {"overview", "trial_details", "results"}:
            resp = "overview"

        state["trial_stage"] = resp
        
        return state

    def stage_prefix_prompt(stage: str) -> str:
        if stage == "overview":
            return """
    You are generating an OVERVIEW-LEVEL clinical trials SQL query.

    STRICT RULES:
    - ALWAYS include the following columns: brand_name, nct_id, phase, overall_status, primary_completion_date
    - DO NOT include:
        - trial arms
        - interventions
        - outcomes
        - outcome_measurements
    """

        if stage == "trial_details":
            return """
    You are generating a TRIAL-DETAILS clinical trials SQL query.

    STRICT RULES:
    - Return trial arms, interventions, design groups, eligibility
    - DO NOT return outcome values or measurements
    - If a list of trials is already available, FILTER to those nct_id values
    """

        if stage == "results":
            return """
    You are generating a RESULTS-LEVEL clinical trials SQL query.

    STRICT RULES:
    - ALWAYS use outcomes AND outcome_measurements tables
    - NEVER scan all trials unless explicitly requested
    - DO NOT return trial design information
    - If available, FILTER using existing nct_id values
    """

        return ""

    
    def draft_sql_node(state: AgentState,catalog) -> AgentState:
        #print(state.get('drugs'))

        stage = state.get("trial_stage", "overview")

        STAGE_PREFIX = stage_prefix_prompt(stage)
        drugs_df = ""
        if state.get("drugs") is not None:
            drugs_df = split_payload_to_df(state.get("drugs")).to_records("records")

        SYSTEM_SQL = f"""{STAGE_PREFIX}

        You are an expert ClinicalTrials.gov (AACT) analyst who writes clean, safe PostgreSQL.

        IMPORTANT CONTEXT & SCOPE RULES (MUST FOLLOW):

- The list of drugs provided to you represents the ACTIVE and AUTHORITATIVE trial scope.
- You MUST restrict all clinical trial queries to these drugs unless the user EXPLICITLY asks to change or expand the drug scope.
- An ACTIVE TRIAL SCOPE with non-drug constraints is also provided.
- You MUST apply ALL scope constraints to every query.
- If the user asks a follow-up or drill-down question (e.g., outcomes, arms, phases) WITHOUT mentioning drugs, you MUST reuse the provided drug list.
- You are NOT allowed to infer, assume, or expand to additional drugs or trial scope beyond the provided list.

active drugs: {drugs_df}
ACTIVE_SCOPE = {json.dumps(state.get("active_trial_scope", []), indent=2)}

SQL query generation Rules:
- Output ONE SQL query only (no commentary, no code fences).
- Read-only: WITH/SELECT only; never DDL/DML or COPY.
- ALWAYS CHECK data types of columns BEFORE joining tables.
- ALWAYS CHECK which columns are asked by the use and return all of them.
- ALWAYS CAST ALL types of id to varchar.
- Always inline all literal values directly in the SQL.
- Never use parameters or placeholders of any kind (no :param, @param, ?, $1, etc.).
- If the question lists multiple drug names, inline them in the SQL using an IN 
- ALWAYS USE public.drug_trials TABLE TO GET TRIAL IDS FOR A PARTICULAR DRUG. 
- Guidelines for pipeline drug related queries:
    - If the question mentions pipeline/upcoming drugs -> extract data from public.onco_pipeline_trials
    - Ignore the `drugs` list provided.
    - For PRO queries related to pipeline drugs use public.design_outcomes_pro table.
- DO NOT USE PRO related tables unless user explicitly mentions.
- Guidelines for endpoint/ outcome related queries
    - Endpoints mean outcomes.
    - BE very careful to check whether design outcomes or actual outcomes are asked.
    - DO NOT USE design outcome tables unless user SPECIFICALLY asks for it.
    - DO NOT RETURN outcome titles, ALWAYS USE public.drug_trial_outcome_categories table.
    - To identify primary/secondary/other endpoints, ALWAYS USE either ctgov.outcomes.outcome_type or ctgov.design_outcomes.outcome_type.
    - IF outcome/ endpoint categories are asked, ALWAYS USE public.drug_trial_outcome_categories table.
    - USE PRO related tables only if the user asks for PRO or patient reported outcomes.
    - WHen results are asked: ALWAYS USE `outcomes` and `outcome_measurements` tables.
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
For adverse events relates queries:
    - Always return only the top 5 adverse event terms per trial by number of subjects affected.
    - Always get data by brand name.
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
    graph.add_node("extract_trial_scope",partial(extract_trial_scope_node,catalog=catalog))
    graph.add_node("classify_trial_stage", classify_trial_stage_node)
    graph.add_node("draft_sql", partial(draft_sql_node,catalog=catalog))
    graph.add_node("validate_sql", validate_sql_node)
    graph.add_node("revise_sql", revise_sql_node)
    graph.add_node("run_sql", partial(run_sql_node,db_key='aact'))
    graph.add_node("explain_sql",partial(explain_sql,catalog=catalog))
    graph.add_node("plot",plot_node)
    graph.add_node("done", done_node)

    graph.set_entry_point("entry")
    graph.add_conditional_edges("entry", decide_after_entry, {
        "plot": "plot",
        "draft_sql": "extract_trial_scope",
    })
    graph.add_edge("extract_trial_scope","classify_trial_stage")
    graph.add_edge("classify_trial_stage", "draft_sql")
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
