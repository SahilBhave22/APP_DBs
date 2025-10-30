# app_orchestrator_parallel.py
import os, json
import streamlit as st
import pandas as pd
from urllib.parse import quote_plus

from fdaers_agent import build_fdaers_agent
from clinicaltrials_agent import build_clinicaltrials_agent
from pricing_agent import build_pricing_agent
from orchestrator_parallel_subq import build_orchestrator_parallel_subq
from IPython.display import Image
import plotly.io as pio

from functools import lru_cache

from google.oauth2 import service_account
from google.cloud.sql.connector import Connector, IPTypes
import sqlalchemy
import uuid
from utils.helpers import split_payload_to_df

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [] 

st.set_page_config(page_title="FDAERS + Clinical Trials ", layout="wide")

OPENAI_KEY     = st.secrets.get("openai_api_key")

for k in ("LANGCHAIN_TRACING","LANGCHAIN_ENDPOINT","LANGCHAIN_API_KEY","LANGCHAIN_PROJECT"):
    if k in st.secrets['langchain_creds']:
        os.environ[k] = st.secrets['langchain_creds'][k]

if OPENAI_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY

@st.cache_resource
def load_json(key: str, fallback: str):
    if key in st.secrets:
        return json.loads(st.secrets[key])
    path = st.secrets.get(f"{key}_path", fallback)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def get_apps():

    faers_catalog = load_json("faers_schema_catalog", "faers_schema_catalog.json")
    aact_catalog  = load_json("clinicaltrials_schema_catalog",  "aact_schema_catalog.json")
    pricing_catalog  = load_json("pricing_schema_catalog",  "pricing_schema_catalog.json")
    aact_sample_queries = load_json("clinicaltrials_sample_queries",  "clinicaltrials_sample_queries.json")

    faers_app = build_fdaers_agent(faers_catalog)
    aact_app  = build_clinicaltrials_agent(aact_catalog,  aact_sample_queries)
    pricing_app  = build_pricing_agent(pricing_catalog,  safe_mode=safe_mode)
    orch_app  = build_orchestrator_parallel_subq(faers_app, aact_app,pricing_app)
    
    return orch_app

st.title("Apperture Dashboard")


with st.sidebar:
    st.header("Settings")
    safe_mode = True #st.toggle("Safe mode (SELECT/WITH only)", value=True)
    want_summary = st.toggle("Want a summary?", value = True)
    want_chart = st.toggle("Want a chart?", value=True)
    safe_mode = True #st.toggle("Safe mode (SELECT/WITH only)", value=True)
    default_limit = st.number_input("Default LIMIT", 10, 10000, 200, step=10)

q = st.text_area(
    "Ask a question",
    placeholder="Example: Compare top 10 conditions across fdaers and clinical trials for Bavencio",
    height=140,
)

if st.button("Run", type="primary"):
    if not q.strip():
        st.error("Please enter a question.")
        st.stop()

    st.session_state.messages.append(("user", q))

    orch = get_apps()
    cfg = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.spinner("Thinking..."):
        out = orch.invoke({"question": q,
                            "want_chart": want_chart,
                            "want_summary": want_summary,
                            "default_limit": int(default_limit),
                            "chat_history": st.session_state.messages[-5:]
                        }, config=cfg)
        st.session_state.current_result = out    

out = st.session_state.get("current_result")
if out:
    # Sidebar results per source
    with st.sidebar:
        if st.button("Reset chat"):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.messages = []
            if "current_result" in st.session_state:
                del st.session_state.current_result
            st.rerun()
        
        st.subheader("FDAERS")
        if out.get("faers_df"):
            fdf = split_payload_to_df(out["faers_df"])
            with st.expander("FDAERS data"):
                st.caption(f"{len(fdf):,} rows · {fdf.shape[1]} cols")
                st.dataframe(fdf.head(default_limit), use_container_width=True, hide_index=True)
                st.download_button("Download FDAERS CSV", fdf.to_csv(index=False).encode("utf-8"), "faers_results.csv", use_container_width=True)
        if out.get("faers_sql"):
            with st.expander("FDAERS SQL query"):
                st.code(out["faers_sql"], language="sql")
        if out.get("faers_sql_explain"):
            with st.expander("FAERS SQL Explained"):
                st.text(out["faers_sql_explain"])
        if out.get("faers_error"):
            st.error(f"FAERS error: {out['faers_error']}")

        st.divider()

        st.subheader("Clinical Trials")
        if out.get("aact_df"):
            adf = split_payload_to_df(out["aact_df"])
            with st.expander("Clinical Trials data"):
                st.caption(f"{len(adf):,} rows · {adf.shape[1]} cols")
                st.dataframe(adf.head(default_limit), use_container_width=True, hide_index=True)
                st.download_button("Download Clinical Trials CSV", adf.to_csv(index=False).encode("utf-8"), "aact_results.csv", use_container_width=True)
        if out.get("aact_sql"):
            with st.expander("Clinical Trials SQL query"):
                st.code(out["aact_sql"], language="sql")
        if out.get("aact_sql_explain"):
            with st.expander("Clinical Trials SQL Explained"):
                st.text(out["aact_sql_explain"])
        if out.get("aact_error"):
            st.error(f"AACT error: {out['aact_error']}")

        st.divider()

        st.subheader("Pricing")
        if out.get("pricing_df"):
            pdf = split_payload_to_df(out["pricing_df"])
            with st.expander("Pricing data"):
                st.caption(f"{len(pdf):,} rows · {pdf.shape[1]} cols")
                st.dataframe(pdf.head(default_limit), use_container_width=True, hide_index=True)
                st.download_button("Download Pricing Data CSV", pdf.to_csv(index=False).encode("utf-8"), "pricing_results.csv", use_container_width=True)
        if out.get("pricing_sql"):
            with st.expander("Pricing SQL query"):
                st.code(out["pricing_sql"], language="sql")
        if out.get("pricing_sql_explain"):
            with st.expander("Pricing SQL Explained"):
                st.text(out["pricing_sql_explain"])
        if out.get("pricing_error"):
            st.error(f"PRICING error: {out['pricing_error']}")

    # Main pane: router rationale + summary
    st.caption(out.get("router_rationale") or "No router rationale")
    st.subheader("Summary report")
    # with st.expander("Show orchestrator diagram", expanded=False):
    #     st.image(orch.get_graph().draw_mermaid_png()) 
    st.markdown(out.get("final_answer") or "_No summary produced_")

    st.subheader("Data Chart")
    if out.get("figure_json"):
        fig = pio.from_json(out["figure_json"])
        st.plotly_chart(fig, use_container_width=True)
    elif out.get("chart_error"):
        st.info(out["chart_error"])
    else:
        st.markdown("_No Chart Produced_")
