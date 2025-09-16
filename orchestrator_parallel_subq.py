# orchestrator_parallel_subq.py
from __future__ import annotations
from typing import TypedDict, Optional, Dict, Any
from dataclasses import dataclass
import json
import pandas as pd

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from langsmith.run_helpers import traceable
#from langchain_core.runnables import RunnableConfig
import plotly.express as px
import plotly.io as pio
import types


from IPython.display import Image
import re

# ---------- Orchestrator state ----------
class OrchestratorState(TypedDict, total=False):
    question: str

    # Router outputs (fan-out flags + sub-questions)
    need_faers: bool
    need_aact: bool
    router_rationale: str
    faers_subq: Optional[str]
    aact_subq: Optional[str]

    # FAERS artifacts
    faers_sql: Optional[str]
    faers_df: Optional[pd.DataFrame]
    faers_error: Optional[str]

    # AACT artifacts
    aact_sql: Optional[str]
    aact_df: Optional[pd.DataFrame]
    aact_error: Optional[str]

    figure_json: str
    chart_source: str
    # Final
    final_answer: Optional[str]

# ---------- Router with per-DB sub-questions ----------
@dataclass
class Signals:
    faers_terms: list[str]
    aact_terms: list[str]

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
    )

def router_node(state: OrchestratorState) -> OrchestratorState:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    sig = default_signals()

    system = (
        "You are a routing planner for a multi-DB QA system.\n"
        "Decide which databases are needed and craft concise, natural language sub-questions for each.\n"
        "Output STRICT JSON with keys: need_faers (bool), need_aact (bool), "
        "router_rationale (str), faers_subq (str or null), aact_subq (str or null).\n"
        "DO NOT GIVE ANY TRAILING OR LEADING CHARACTERS AROUND THE JSON"
        "Guidance:\n"
        "- FAERS for spontaneous safety. Take aggregates across primary ids. \n"
        "- AACT for trials .Take aggregates across nct ids.\n"
        "- If the user intent implies both, set both booleans true.\n"
        "- If the user criteria applies to both databases, frame sub-questions accordingly.\n"
        "- Sub-questions should be in natural language and should only be extracted from the original question\n"
        "- DO NOT invent or rephrase extra conditions for sub-questions.\n"
    )
    user = f"""
User question:
{state['question']}

FAERS hints: {", ".join(sig.faers_terms)}
AACT hints: {", ".join(sig.aact_terms)}

Return JSON ONLY NO TRAILING CHARACTERS:
{{
  "need_faers": <bool>,
  "need_aact": <bool>,
  "router_rationale": "<string>",
  "faers_subq": "<string or null>",
  "aact_subq": "<string or null>"
}}
"""
    try:
        resp = llm.invoke([{"role":"system","content":system},{"role":"user","content":user}]).content
        resp = re.sub(r"^```(?:json)?|```$", "", resp.strip(), flags=re.MULTILINE).strip()
        data = json.loads(resp)
    except Exception as e:
        print(e)
        data = {
            "need_faers": True,
            "need_aact": False,
            "router_rationale": "Fallback: parse error; defaulting to FAERS only.",
            "faers_subq": state["question"],
            "aact_subq": None
        }

    return {
        "need_faers": bool(data.get("need_faers", False)),
        "need_aact": bool(data.get("need_aact", False)),
        "router_rationale": data.get("router_rationale", ""),
        "faers_subq": data.get("faers_subq"),
        "aact_subq": data.get("aact_subq"),
    }

# ---------- Orchestrator builder (wire in your compiled agent apps) ----------
def build_orchestrator_parallel_subq(faers_app, aact_app,want_chart,want_summary):
    """
    faers_app / aact_app are your compiled LangGraph agent apps from build_agent(...),
    which take {"question", "sql", "error", "attempts", "summary", "df"} and return with df/sql/error filled.
    """
    @traceable(name="FAERS Agent")
    def call_faers(state: OrchestratorState) -> OrchestratorState:
        if not state.get("need_faers"):
            return {}
        subq = state.get("faers_subq") or state["question"]
        s_in = {"question": subq, "sql": None, "error": None, "attempts": 0, "summary": None, "df": None}
        out = faers_app.invoke(s_in)
        return {"faers_sql": out.get("sql"), "faers_df": out.get("df"), "faers_error": out.get("error")}

    @traceable(name="AACT Agent")
    def call_aact(state: OrchestratorState) -> OrchestratorState:
        if not state.get("need_aact"):
            return {}
        subq = state.get("aact_subq") or state["question"]
        s_in = {"question": subq, "sql": None, "error": None, "attempts": 0, "summary": None, "df": None}
        out = aact_app.invoke(s_in)
        return {"aact_sql": out.get("sql"), "aact_df": out.get("df"), "aact_error": out.get("error")}

    def gather_node(state: OrchestratorState) -> OrchestratorState:
        # Barrier; nothing to compute—just ensures both agent branches completed.
        return {}

    def summarize_node(state: OrchestratorState) -> OrchestratorState:
        if not want_summary:
            return {}
        llm = ChatOpenAI(model="gpt-4o", temperature=0)

        def meta(df: Optional[pd.DataFrame]):
            if isinstance(df, pd.DataFrame):
                return {
                    "rows": len(df),
                    "cols": df.shape[1],
                    "columns": [str(c) for c in df.columns][:40],
                    "sample": df.head(100).to_dict("records"),
                }
            return {"rows": 0, "cols": 0, "columns": [], "sample": []}

        fmeta = meta(state.get("faers_df"))
        ameta = meta(state.get("aact_df"))

        prompt = f"""
Write a concise, decision-ready report using the available sources.

Main question:
{state['question']}

Router:
- Rationale: {state.get('router_rationale')}
- FAERS sub-question: {state.get('faers_subq') or '—'}
- AACT  sub-question: {state.get('aact_subq') or '—'}

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

Instructions:
- DO NOT REPEAT THE RESULTS OF SQL QUERIES.
- Use the per-DB sub-questions as the lens for interpreting each table.
- Do NOT dump tables; surface patterns, outliers, caveats.
- Suggest 1–3 concrete next steps.
- DO NOT MAKE UP summary, please derive only from results of sql data.
- Confine your summary to 2000 words.
"""
        ans = llm.invoke(prompt).content
        return {"final_answer": ans}

    def plot_node(state: OrchestratorState) -> OrchestratorState:

        if not want_chart:
            return {}
   
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        def meta(df: Optional[pd.DataFrame]):
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                return {
                    "rows": len(df),
                    "cols": df.shape[1],
                    "columns": [str(c) for c in df.columns][:50],
                    "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
                }
            return {"rows": 0, "cols": 0, "columns": [], "dtypes": {}}

        def choose_df(question: str, f_df: Optional[pd.DataFrame], a_df: Optional[pd.DataFrame]):
            override = (state.get("plot_source") or "").lower()
            if override in {"faers", "aact"}:
                return (f_df, "faers") if override == "faers" else (a_df, "aact")
            q = (question or "").lower()
            f_len = len(f_df) if isinstance(f_df, pd.DataFrame) else 0
            a_len = len(a_df) if isinstance(a_df, pd.DataFrame) else 0
            if any(k in q for k in ["faers", "adverse", "ae", "meddra", "ror", "prr"]):
                return (f_df, "faers")
            if any(k in q for k in ["trial", "aact", "nct", "phase", "enrollment"]):
                return (a_df, "aact")
            if f_len >= a_len and f_len > 0:
                return (f_df, "faers")
            if a_len > 0:
                return (a_df, "aact")
            return (None, "none")

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
                "px": px, "go": go
            }
            safe_locals = {"df": df}
            exec(code, safe_globals, safe_locals)  # code must set 'fig'
            fig = safe_locals.get("fig")
            if fig is None:
                raise ValueError("The snippet did not create a variable named 'fig'.")
            return fig

        question = state.get("question") or ""
        f_df = state.get("faers_df")
        a_df = state.get("aact_df")
        df_to_plot, source = choose_df(question, f_df, a_df)

        if not isinstance(df_to_plot, pd.DataFrame) or len(df_to_plot) == 0:
            return {"chart_error": "No data available for plotting from FAERS or AACT.", "chart_source": source}

        fmeta = meta(f_df)
        ameta = meta(a_df)
        preview = df_to_plot.head(6).to_string(index=False)

        prompt = f"""
    You produce minimal Plotly Express code from a pandas DataFrame to visualize the user's question.

    Rules:
    - Use ONLY: df (pandas.DataFrame to plot), px (plotly.express), go (plotly.graph_objects optional).
    - ALL IMPORTS ARE ALREADY PRESENT, DO NOT ADD IMPORT STATEMENTS TO THE CODE.
    - DO NOT add any imports, files, network, or system access statements
    - End with a variable named: fig
    - Use clear defaults: informative title, axis labels, and template='plotly_white'.
    - Time trend -> line/area; categorical comparison -> bar; distribution -> histogram/box/violin; heatmap for matrix-like comparisons.
    - Use a stacked bar chart wherever possible
    - Use ONLY columns that exist in df.

    User question:
    {question}

    Chosen dataset to plot: {source.upper()}

    DF preview (first 6 rows):
    {preview}

    FAERS columns: {fmeta['columns']}
    AACT columns : {ameta['columns']}

    Return ONLY Python code that uses df and px (and optionally go) and ends with:
    fig = <plotly figure>
    """
        code = llm.invoke(prompt).content.strip()
        if code.startswith("```"):
            code = code.strip("` \n")
            if "\n" in code and code.split("\n", 1)[0].lower().startswith("python"):
                code = code.split("\n", 1)[1]

        try:
            fig = run_safely(df_to_plot, code)
            fig.update_layout(template="plotly_white")
            return {
                "figure_json": fig.to_json(),
                "chart_source": source,
                # "chart_code": code,  # uncomment if you want to inspect snippets
            }
        except Exception as e:
            print(e)
            return {"chart_error": f"Chart generation failed: {e}", "chart_source": source}
    # ---------- Graph (parallel fan-out) ----------
    graph = StateGraph(OrchestratorState)
    graph.add_node("router", router_node)
    graph.add_node("faers", call_faers)
    graph.add_node("aact", call_aact)
    graph.add_node("gather", gather_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("plot",plot_node)
    graph.set_entry_point("router")

    # Fan-out: run both agents concurrently; each checks its own flag
    graph.add_edge("router", "faers")
    graph.add_edge("router", "aact")

    # Join
    graph.add_edge("faers", "gather")
    graph.add_edge("aact", "gather")

    # Summarize
    graph.add_edge("gather", "summarize")
    graph.add_edge("summarize","plot")
    graph.add_edge("plot", END)
    
    graph_final = graph.compile()
    #Image(graph_final().get_graph().draw_mermaid_png())


    return graph_final

