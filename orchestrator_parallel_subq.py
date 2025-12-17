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
    if('pipeline' in state.get("question")):
        state["drugs"] = None
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
    
    new_drugs_df = pd.DataFrame({'Brand Name': out.get('drugs')})
    new_drugs = df_to_split_payload(new_drugs_df)
  
    state['drugs'] = new_drugs if len(new_drugs_df)>0 else prev_drugs
    
    state['criteria'] = out.get('criteria_phrases')
    state['companies'] = out.get('companies')
    state['user_criteria_changed'] = out.get('user_criteria_changed')
    #print(out.get('user_criteria_changed'))
    #print(out.get('rationale'))
    return state

def decide_next_after_entry(state: OrchestratorState) -> Literal["router", "get_relevant_drugs"]:
        # print(len(split_payload_to_df(state['drugs'])))
        # if "faers_df" in state or "aact_df" in state or "pricing_df" in state or (state['drugs'] is not None and len(state['drugs'])>0):
        #     return "router"
        # else:
        #     return "get_relevant_drugs"
        
        if state.get('user_criteria_changed'):
            return "get_relevant_drugs"
        else:
            return "router"
        
def get_relevant_drugs(state: OrchestratorState) -> OrchestratorState:
    
    with open("catalogs/drugs_schema_catalog.json", "r", encoding="utf-8") as f:
        drugs_json =  json.load(f)
    
    
    FETCH_RELEVANT_DRUGS_USER = f"""
drugs_json: {drugs_json}

criteria_phrases:
{state.get('criteria')}

drug manufacturer companies:
{state['companies']}

Return STRICT JSON only.
"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    resp = llm.invoke([
            {"role": "system", "content": FETCH_RELEVANT_DRUGS_SYSTEM},
            {"role": "user", "content": FETCH_RELEVANT_DRUGS_USER},
        ])

    #print(resp.content)
    out_raw = re.sub(r"^```[a-zA-Z]*\n?|```$", "", resp.content).strip()
    out = json.loads(out_raw)
    
    
    drugs_query = """
    select distinct dr.brand_name 
    from drugs dr 
    left join drug_classes dc
    on dr.brand_name = dc.brand_name
    left join drug_indications di
    on dr.brand_name = di.brand_name
    left join drug_manufacturers dm
    on dr.brand_name = dm.brand_name
    where 1=1
    """

    where_clauses = []
    params = {}

    
    # normalize to lowercase to match LOWER() in SQL
    selected_classes = [f"%{c.lower().strip()}%" for c in out.get("selected_drug_classes")]
    selected_indications = [f"%{i.lower().strip()}%" for i in out.get("selected_drug_indications")]
    selected_companies = [f"%{m.lower().strip()}%" for m in out.get("selected_drug_companies")]

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

    if len(selected_companies)>0:
        where_clauses.append(
            "AND LOWER(dm.manufacturer) ILIKE ANY(:selected_companies)"
        )
        params["selected_companies"] = selected_companies

    
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
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
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
                "drugs":state.get('drugs'), "active_trial_scope":state.get('active_trial_scope'),
                "criteria":state.get('criteria')}
        out = aact_app.invoke(s_in,config = config)
        return {"aact_sql": out.get("sql"), "aact_df": out.get("df"), "aact_figure_json":out.get("figure_json"),
                "aact_error": out.get("error"), "aact_sql_explain":out.get("sql_explain"),"active_trial_scope":out.get("active_trial_scope")}
    
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
        llm_mini = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        def chunk_df(df, size=500):
            return [df[i:i+size] for i in range(0, len(df), size)]

        def meta(df: Optional[pd.DataFrame]):
            if isinstance(df, pd.DataFrame):
                return {
                    "rows": len(df),
                    "cols": df.shape[1],
                    "columns": list(df.columns)
                }
            return {"rows": 0, "cols": 0, "columns": []}
        
        f_df = split_payload_to_df(state.get("faers_df"))
        a_df = split_payload_to_df(state.get("aact_df"))
        p_df = split_payload_to_df(state.get("pricing_df"))

        fmeta, ameta, pmeta = meta(f_df), meta(a_df), meta(p_df)

        drugs_df = ""
        if state.get("drugs") is not None:
            drugs_df = split_payload_to_df(state.get("drugs")).to_records("records")
        hist = state.get("chat_history", [])[-8:]  # last few turns
        hist_str = "\n".join(f"{r.upper()}: {c}" for r, c in hist) if hist else "None"
        criteria_str = "\n".join(f"{c}" for c in state.get("criteria")) if state.get("criteria") else "None"
        #drugs_str = "\n".join(f"{d}" for d in state.get()) if hist else "None"
        def summarize_chunk(df, source_name):
            """Summarize a DF chunk with contextual understanding."""
            text = df.to_json(orient="records")
            prompt = f"""
You are a domain-aware pharma/clinical data analyst.

You are analyzing a CHUNK of a dataset from: **{source_name}**

User Search Criteria:
{criteria_str}

Drugs of Interest:
{drugs_df}

Chat history context:
{hist_str}

Column definitions in this chunk:
{list(df.columns)}

Your goals:
1. Identify meaningful patterns, trends, safety signals, clinical themes, or pricing behavior.
2. Interpret both numerical and text fields (e.g., reaction names, intervention types, costs).
3. Flag anomalies, outliers, or surprising co-occurrences.
4. DO NOT hallucinate missing columns or values.
5. DO NOT repeat row-level data.
6. Think as if this chunk is *part of a larger dataset*; highlight patterns that might scale.

Chunk Data (JSON, full fidelity):
{text[:30000]}

Produce a coherent summary (~200–300 words) for this chunk only.
"""
            resp = llm_mini.invoke(prompt)
            return resp.content.strip()
        
        def process_source(df, name):
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                chunks = chunk_df(df)
                out = []
                for i, c in enumerate(chunks):
                    out.append(summarize_chunk(c, f"{name} (chunk {i+1})"))
                return out
            return []
        
        faers_summaries = process_source(f_df, "FAERS")
        aact_summaries = process_source(a_df, "AACT")
        pricing_summaries = process_source(p_df, "PRICING")

        all_summaries = faers_summaries + aact_summaries + pricing_summaries

        prompt_final = f"""
You are a senior pharma / clinical trial / safety data analyst.

Below are summaries derived from chunk-level analysis of FAERS, AACT, and PRICING datasets.
Your task is to combine them into ONE unified, decision-ready report.
If multiple chunks mention the same pattern, consolidate them into one single insight in the final summary. 
Do not repeat similar points.

Context:
User Search Criteria:
{criteria_str}

Drugs:
{drugs_df}

FAERS Meta: {fmeta}
AACT Meta: {ameta}
PRICING Meta: {pmeta}

Chat History:
{hist_str}

Your goals:
- Merge all chunk insights into a coherent end-to-end narrative.
- Identify cross-database relationships (e.g., trial signals vs. AE patterns vs. pricing).
- Capture real patterns only from summaries (no hallucinations).
- Highlight safety issues, clinical insights, pricing implications.
- Add 1–3 recommended next steps.
- Keep ≤2000 words.

Chunk Summaries:
{"\n\n-----\n\n".join(all_summaries)}
"""

        final_answer = llm.invoke(prompt_final).content
        return {"final_answer": final_answer}
    


    
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

