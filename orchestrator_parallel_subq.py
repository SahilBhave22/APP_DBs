# orchestrator_parallel_subq.py
from __future__ import annotations
from typing import TypedDict, Optional, Dict, Any,List,Literal
from dataclasses import dataclass
import json
import pandas as pd

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from langsmith.run_helpers import traceable
#from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

#from IPython.display import Image
import re
from utils.helpers import split_payload_to_df, df_to_split_payload
from utils.db_conn import exec_sql

from utils.prompts import DRUG_DETECTOR_SYSTEM,FETCH_RELEVANT_DRUGS_SYSTEM, ROUTER_SYSTEM
from utils.states import OrchestratorState

# ---------- Router with per-DB sub-questions ----------
@dataclass
class Signals:
    faers_terms: list[str]
    aact_terms: list[str]
    pricing_terms: list[str]

def default_signals() -> Signals:
    return Signals(
        faers_terms=[
            "fdaers","aers","adverse event","ae","meddra","pt","soc",
            "ror","reported odds ratio","prr","ic","primaryid","caseid",
            "dechallenge","rechallenge","serious","death","hospitalization",
            "suspect","concomitant","reac","demo","indi","drug_cases","outcomes"
        ],
        aact_terms=[
            "trials","aact","nct","nct id","nctid",
            "phase","phase 2","phase 3","pivotal","randomized",
            "primary outcome","secondary outcome","endpoint","intervention",
            "sponsor","condition","eligibility","inclusion","exclusion",
            "recruiting","completion date","results","arms","outcomes","patient reported","PRO"
        ],
        pricing_terms=[
            "pricing","annual price", "cycle","price change"
        ],
    )

def drug_detector(state: OrchestratorState) -> OrchestratorState:
    if(state.get('call_source')!= 'database'):
        return state
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prev_drugs = state.get("drugs")

    DRUG_DETECTOR_USER = f"""
    previously_active_drugs:
    {prev_drugs}

    Current user question:
    {state.get("question")}

    Follow the system instructions and return STRICT JSON only.
"""
    
    resp = llm.invoke([
        {"role": "system", "content": DRUG_DETECTOR_SYSTEM},
        {"role": "user", "content": DRUG_DETECTOR_USER},
    ])

    out_raw = re.sub(r"^```[a-zA-Z]*\n?|```$", "", resp.content).strip()
    out = json.loads(out_raw)
    #out = json.loads(resp.content)
    # print("enter ehllo")
    # print(type(out.get('drugs')))
    # print(out.get('criteria_phrases'))
    # print(out.get('rationale'))
    # print("enter exit")
    
    state['drugs'] = df_to_split_payload(pd.DataFrame({'Brand Name': out.get('drugs')}))
    state['criteria'] = out.get('criteria_phrases')
    return state

def decide_next_after_entry(state: OrchestratorState) -> Literal["router", "get_relevant_drugs"]:
        # print(len(split_payload_to_df(state['drugs'])))
        return "router" if len(split_payload_to_df(state['drugs']))>0 else "get_relevant_drugs"

def get_relevant_drugs(state: OrchestratorState) -> OrchestratorState:
    
    with open("catalogs/drugs_schema_catalog.json", "r", encoding="utf-8") as f:
        drugs_json =  json.load(f)
    
    #print(drugs_json)
    FETCH_RELEVANT_DRUGS_USER = f"""
drugs_json: {drugs_json}

criteria_phrases:
{state.get('criteria')}

Return STRICT JSON only.
"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    resp = llm.invoke([
            {"role": "system", "content": FETCH_RELEVANT_DRUGS_SYSTEM},
            {"role": "user", "content": FETCH_RELEVANT_DRUGS_USER},
        ])

    #print(resp.content)
    out_raw = re.sub(r"^```[a-zA-Z]*\n?|```$", "", resp.content).strip()
    out = json.loads(out_raw)
    
    # print("Enter")
    # print(out.get('selected_drug_classes'))
    # print(out.get('selected_drug_indications'))
    # print(out.get('rationale'))
    # print("Exit")

    drugs_query = """
    select distinct dr.brand_name 
    from drugs dr 
    left join drug_classes dc
    on dr.brand_name = dc.brand_name
    left join drug_indications di
    on dr.brand_name = di.brand_name
    where 1=1
    """

    where_clauses = []
    params = {}

    
    # normalize to lowercase to match LOWER() in SQL
    selected_classes = [f"%{c.lower()}%" for c in out.get("selected_drug_classes")]
    selected_indications = [f"%{i.lower()}%" for i in out.get("selected_drug_indications")]

    # print(selected_classes)
    # print(selected_indications)

    if len(selected_classes)>0:
        where_clauses.append(
            "AND LOWER(dc.atc_class_name) ILIKE ANY(:selected_classes)"
        )
        params["selected_classes"] = selected_classes

    if len(selected_indications)>0:
        where_clauses.append(
            "AND LOWER(di.indication_name) ILIKE ANY(:selected_indications)"
        )
        params["selected_indications"] = selected_indications

    
    drugs_query += "\n" + "\n".join(where_clauses)

    # print(drugs_query)
    drugs_list = exec_sql(drugs_query,db_key="drugs",params = params)
    #print(drugs_list)
    state["drugs"] = df_to_split_payload(drugs_list)

    return state

def router_node(state: OrchestratorState) -> OrchestratorState:

    if(state.get('call_source')!= 'database'):
        return {
            "need_faers": state.get('faers_df') is not None,
            "need_aact": state.get('aact_df') is not None,
            "need_pricing": state.get('pricing_df') is not None
        }
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    sig = default_signals()

    
    user = f"""
Resolved drugs for this turn (FINAL, DO NOT OVERRIDE):
{state.get('drugs')}

Current user question:
{state['question']}

FAERS hints: {", ".join(sig.faers_terms)}
AACT hints: {", ".join(sig.aact_terms)}
PRICING hints: {", ".join(sig.pricing_terms)}

Return JSON ONLY NO TRAILING CHARACTERS:
{{
  "need_faers": <bool>,
  "need_aact": <bool>,
  "need_pricing": <bool>,
  "router_rationale": "<string>",
  "faers_subq": "<string or null>",
  "aact_subq": "<string or null>",
  "pricing_subq": "<string or null>"
}}
"""
    try:
        resp = llm.invoke([{"role":"system","content":ROUTER_SYSTEM},{"role":"user","content":user}]).content
        resp = re.sub(r"^```(?:json)?|```$", "", resp.strip(), flags=re.MULTILINE).strip()
        data = json.loads(resp)
    except Exception as e:
        print(e)
        data = {
            "need_faers": False,
            "need_aact": False,
            "need_pricing": False,
            "router_rationale": "Fallback: parse error; defaulting to None",
            "faers_subq": None,
            "aact_subq": None,
            "pricing_subq": None
        }

    #print(data.get("drugs"))
    #print(data.get("criteria"))
    return {
        "need_faers": bool(data.get("need_faers", False)),
        "need_aact": bool(data.get("need_aact", False)),
        "need_pricing": bool(data.get("need_pricing", False)),
        "router_rationale": data.get("router_rationale", ""),
        "faers_subq": data.get("faers_subq") if data.get("faers_subq") is not None else state.get('faers_subq'),
        "aact_subq": data.get("aact_subq") if data.get("aact_subq") is not None else state.get('aact_subq'),
        "pricing_subq": data.get("pricing_subq") if data.get("pricing_subq") is not None else state.get('pricing_subq'),
    }




# ---------- Orchestrator builder (wire in your compiled agent apps) ----------
def build_orchestrator_parallel_subq(faers_app, aact_app,pricing_app):
    """
    faers_app / aact_app are your compiled LangGraph agent apps from build_agent(...),
    which take {"question", "sql", "error", "attempts", "summary", "df"} and return with df/sql/error filled.
    """
    
    @traceable(name="FAERS Agent")
    def call_faers(state: OrchestratorState,config) -> OrchestratorState:
        if not state.get("need_faers") or state.get("call_source")=="summary_toggle":
            return {}
        
        subq = state.get("faers_subq") or state["question"]
        want_chart = state.get("want_chart")

        #print(state.get('drugs'))
        #print(state.get('criteria'))
        if state.get("call_source") == "chart_toggle":
            s_in = {"question": subq, "df":state.get("faers_df"), "drugs":state.get("drugs"),
                    "want_chart":want_chart,"call_source":state.get("call_source")}
            
            out = faers_app.invoke(s_in,config = config)
            return {"faers_figure_json":out.get("figure_json")}
    
        s_in = {"question": subq, "sql": None, "error": None, "want_chart" : want_chart,
                "attempts": 0, "summary": None, "df": None,"sql_explain":None,
                "call_source":state.get("call_source"), "drugs":state.get('drugs'),
                "criteria":state.get('criteria')}
        out = faers_app.invoke(s_in,config = config)
            
        return {"faers_sql": out.get("sql"), "faers_df": out.get("df"), "faers_figure_json":out.get("figure_json"),
                "faers_error": out.get("error"), "faers_sql_explain":out.get("sql_explain")}

    @traceable(name="AACT Agent")
    def call_aact(state: OrchestratorState,config) -> OrchestratorState:
        if not state.get("need_aact") or state.get("call_source")=="summary_toggle":
            return {}
        
        subq = state.get("aact_subq") or state["question"]
        want_chart = state.get("want_chart")
        if state.get("call_source") == "chart_toggle":
            s_in = {"question": subq, "df":state.get("aact_df"),"drugs":state.get("drugs"),
                    "want_chart":want_chart,"call_source":state.get("call_source")}
            
            out = aact_app.invoke(s_in,config = config)
            return {"aact_figure_json":out.get("figure_json")}

        s_in = {"question": subq, "sql": None, "error": None, "want_chart":want_chart,
                "attempts": 0, "summary": None, "df": None,"call_source":state.get("call_source"),
                "drugs":state.get('drugs'),
                "criteria":state.get('criteria')}
        out = aact_app.invoke(s_in,config = config)
        return {"aact_sql": out.get("sql"), "aact_df": out.get("df"), "aact_figure_json":out.get("figure_json"),
                "aact_error": out.get("error"), "aact_sql_explain":out.get("sql_explain")}
    
    @traceable(name="PRICING Agent")
    def call_pricing(state: OrchestratorState,config) -> OrchestratorState:
        if not state.get("need_pricing") or state.get("call_source")=="summary_toggle":
            return {}
        
        subq = state.get("pricing_subq") or state["question"]
        want_chart = state.get("want_chart")

        if state.get("call_source") == "chart_toggle":
            s_in = {"question": subq, "df":state.get("pricing_df"),"drugs":state.get("drugs"),
                    "want_chart":want_chart,"call_source":state.get("call_source")}
            
            out = pricing_app.invoke(s_in,config = config)
            return {"pricing_figure_json":out.get("figure_json")}
        
        s_in = {"question": subq, "sql": None, "error": None, "want_chart":want_chart,
                "attempts": 0, "summary": None, "df": None,"sql_explain":None,"call_source":state.get("call_source"),
                "drugs":state.get('drugs'),
                "criteria":state.get('criteria')}
        out = pricing_app.invoke(s_in,config = config)
        
        return {"pricing_sql": out.get("sql"), "pricing_df": out.get("df"), "pricing_figure_json":out.get("figure_json"),
                "pricing_error": out.get("error"), "pricing_sql_explain":out.get("sql_explain")}


    def gather_node(state: OrchestratorState) -> OrchestratorState:
        # Barrier; nothing to compute—just ensures both agent branches completed.
        return {}

    def summarize_node(state: OrchestratorState) -> OrchestratorState:
        want_summary = state.get('want_summary')
        if not want_summary:
            return {}
        llm = ChatOpenAI(model="gpt-4o", temperature=0)

        def meta(df: Optional[pd.DataFrame]):
            if isinstance(df, pd.DataFrame):
                return {
                    "rows": len(df),
                    "cols": df.shape[1],
                    "columns": [str(c) for c in df.columns][:40],
                    "sample": df.head(100).to_dict("records")
                }
            return {"rows": 0, "cols": 0, "columns": [], "sample": []}

        fmeta = meta(split_payload_to_df(state.get("faers_df")))
        ameta = meta(split_payload_to_df(state.get("aact_df")))
        pmeta = meta(split_payload_to_df(state.get("pricing_df")))

        drugs_df = split_payload_to_df(state.get("drugs")).to_records("records")
        hist = state.get("chat_history", [])[-8:]  # last few turns
        hist_str = "\n".join(f"{r.upper()}: {c}" for r, c in hist) if hist else "None"
        criteria_str = "\n".join(f"{c}" for c in state.get("criteria")) if state.get("criteria") else "None"
        #drugs_str = "\n".join(f"{d}" for d in state.get()) if hist else "None"
        prompt = f"""
Write a concise, decision-ready report using the available sources.

Recent questions:
{hist_str}

User Search Criteria:
{criteria_str}

Drugs of interest:
{drugs_df}


Router:
- FAERS sub-question: {state.get('faers_subq') or '—'}
- AACT  sub-question: {state.get('aact_subq') or '—'}
- PRICING  sub-question: {state.get('pricing_subq') or '—'}

FAERS:
- rows: {fmeta['rows']}, cols: {fmeta['cols']}
- columns (first 40): {fmeta['columns']}
- sample rows (≤100): {fmeta['sample']}
- SQL: {(state.get('faers_sql') or '')[:1200]}
- error: {state.get('faers_error') or 'None'}

AACT:
- rows: {ameta['rows']}, cols: {ameta['cols']}
- columns (first 40): {ameta['columns']}
- sample rows (≤100): {ameta['sample']}
- SQL: {(state.get('aact_sql') or '')[:1200]}
- error: {state.get('aact_error') or 'None'}

PRICING:
- rows: {pmeta['rows']}, cols: {pmeta['cols']}
- columns (first 40): {pmeta['columns']}
- sample rows (≤100): {pmeta['sample']}
- SQL: {(state.get('pricing_sql') or '')[:1200]}
- error: {state.get('pricing_error') or 'None'}

Instructions:
- DO NOT REPEAT THE RESULTS OF SQL QUERIES.
- Use the per-DB sub-questions as the lens for interpreting each table.
- Do NOT dump tables; surface patterns, outliers, caveats.
- Suggest 1–3 concrete next steps.
- DO NOT MAKE UP summary, please derive only from results of sql data.
- Preserve all numeric fields exactly as shown.
- Confine your summary to 2000 words.
"""
        ans = llm.invoke(prompt).content
        return {"final_answer": ans}

    
    # ---------- Graph (parallel fan-out) ----------
    graph = StateGraph(OrchestratorState)
    graph.add_node("drug_detector", drug_detector)
    graph.add_node("get_relevant_drugs",get_relevant_drugs)
    graph.add_node("router", router_node)
    graph.add_node("faers", call_faers)
    graph.add_node("aact", call_aact)
    graph.add_node("pricing", call_pricing)
    graph.add_node("gather", gather_node)
    graph.add_node("summarize", summarize_node)
    #graph.add_node("plot",plot_node)
    graph.set_entry_point("drug_detector")
    #graph.set_entry_point("router")

    # Fan-out: run both agents concurrently; each checks its own flag
    #graph.add_edge("drug_detector","get_relevant_drugs")
    graph.add_conditional_edges("drug_detector", decide_next_after_entry, {
        "router": "router",
        "get_relevant_drugs": "get_relevant_drugs",
    })
    graph.add_edge("get_relevant_drugs","router")
    graph.add_edge("router", "faers")
    graph.add_edge("router", "aact")
    graph.add_edge("router", "pricing")

    # Join
    graph.add_edge("faers", "gather")
    graph.add_edge("aact", "gather")
    graph.add_edge("pricing", "gather")

    # Summarize
    graph.add_edge("gather", "summarize")
    graph.add_edge("summarize",END)
    #graph.add_edge("plot", END)
    graph_final = graph.compile(checkpointer=MemorySaver())
    #Image(graph_final.get_graph().draw_mermaid_png())


    return graph_final

