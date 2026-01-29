# app_orchestrator_parallel.py
import os, json
import streamlit as st
import pandas as pd
from urllib.parse import quote_plus

from agents.fdaers_agent import build_fdaers_agent
from agents.clinicaltrials_agent import build_clinicaltrials_agent
from agents.pricing_agent import build_pricing_agent
from agents.marketaccess_agent import build_marketaccess_agent
from orchestrator_parallel_subq import build_orchestrator_parallel_subq
from IPython.display import Image
import plotly.io as pio
import langgraph
from functools import lru_cache

from google.oauth2 import service_account
from google.cloud.sql.connector import Connector, IPTypes
#import sqlalchemy
import uuid
from utils.helpers import split_payload_to_df,make_market_access_multiindex
from PIL import Image
from io import BytesIO
import base64
import hashlib
from functools import partial

#from langgraph.config import mermaid_config_context

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [] 

if "last_want_summary" not in st.session_state:
    st.session_state.last_want_summary = True
if "summary_refresh" not in st.session_state:
    st.session_state.summary_refresh = False

if "last_want_chart" not in st.session_state:
    st.session_state.last_want_chart = True
if "chart_refresh" not in st.session_state:
    st.session_state.chart_refresh = False

if "drugs" not in st.session_state:
    st.session_state.drugs = None

if "active_trial_scope" not in st.session_state:
    st.session_state.active_trial_scope = []

if "criteria" not in st.session_state:
    st.session_state.criteria = None

if "current_result" not in st.session_state:
    st.session_state.current_result = None


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- Your user credentials (you can load from env vars instead) ---
USERS = {
    st.secrets['login_creds']['username1']: hash_password(st.secrets['login_creds']['password1'])
}

def login():
    st.title("Apperture Dashboard Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username] == hash_password(password):
            st.session_state["logged_in"] = True
            st.session_state["user"] = username
            st.rerun()
        else:
            st.error("❌ Invalid username or password.")


if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    login()
    st.stop()  # Prevent the rest of the app from showing


def encode_image(path):
    img = Image.open(path)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

logo_base64 = encode_image("assets/logos/APP_logo1.png") 

st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 10px;">
        <img src="data:image/png;base64,{logo_base64}" width="70">
        <h1 style="margin: 0;">Apperture Dashboard</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.set_page_config(page_title="FDAERS + Clinical Trials", layout="wide", page_icon='assets/logos/APP_logo1.png')


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
def get_apps(default_limit:int):

    faers_catalog = load_json("faers_schema_catalog", "faers_schema_catalog.json")
    aact_catalog  = load_json("clinicaltrials_schema_catalog",  "aact_schema_catalog.json")
    pricing_catalog  = load_json("pricing_schema_catalog",  "pricing_schema_catalog.json")
    marketaccess_catalog  = load_json("marketaccess_schema_catalog",  "marketaccess_schema_catalog.json")

    aact_sample_queries = load_json("clinicaltrials_sample_queries",  "clinicaltrials_sample_queries.json")

    faers_app = build_fdaers_agent(catalog = faers_catalog,default_limit=default_limit)
    aact_app  = build_clinicaltrials_agent(aact_catalog,  aact_sample_queries,default_limit=default_limit)
    pricing_app  = build_pricing_agent(pricing_catalog,  default_limit=default_limit,safe_mode=safe_mode)
    marketaccess_app  = build_marketaccess_agent(marketaccess_catalog,  default_limit=default_limit,safe_mode=safe_mode)

    orch_app  = build_orchestrator_parallel_subq(faers_app, aact_app,pricing_app, marketaccess_app)
    
    return orch_app

#st.title("Apperture Dashboard")


with st.sidebar:
    st.header("Settings")
    safe_mode = True #st.toggle("Safe mode (SELECT/WITH only)", value=True)
    want_summary = st.toggle("Want a summary?", value = True)
    want_chart = st.toggle("Want a chart?", value=False)
    safe_mode = True #st.toggle("Safe mode (SELECT/WITH only)", value=True)
    default_limit = st.number_input("Default LIMIT", 10, 20000, 2000, step=10)

q = st.text_area(
    "Ask a question",
    placeholder="Example: Compare top 10 conditions across fdaers and clinical trials for Bavencio",
    height=140,
)
orch = get_apps(default_limit=default_limit)


#img_bytes = orch.get_graph(xray=1).draw_mermaid_png()

cfg = {"configurable": {"thread_id": st.session_state.thread_id}}

# with st.sidebar:
#     with st.expander("Workflow graph"):
#         st.image(img_bytes, caption="Workflow graph", use_container_width=True)
#     st.divider()


if want_summary != st.session_state.last_want_summary:
    st.session_state.last_want_summary = want_summary
    if want_summary and q.strip() and st.session_state.summary_refresh:
        with st.spinner("Generating summary...."):
            out = orch.invoke({"question": q,
                                "want_chart": want_chart,
                                "want_summary": want_summary,
                                "default_limit": int(default_limit),
                                "chat_history": st.session_state.messages[-5:],
                                "call_source":"summary_toggle",
                                "drugs": st.session_state.drugs,
                                "criteria": st.session_state.criteria
                            }, config=cfg)
            st.session_state.current_result = out
        
if want_chart != st.session_state.last_want_chart:
    st.session_state.last_want_chart = want_chart
    if want_chart and q.strip() and st.session_state.chart_refresh:
        with st.spinner("Generating chart...."):
            out = orch.invoke({"question": q,
                                "want_chart": want_chart,
                                "want_summary": want_summary,
                                "default_limit": int(default_limit),
                                "chat_history": st.session_state.messages[-5:],
                                "call_source":"chart_toggle",
                                "drugs": st.session_state.drugs,
                                "criteria": st.session_state.criteria
                            }, config=cfg)
            st.session_state.current_result = out

if st.button("Run", type="primary"):
    if not q.strip():
        st.error("Please enter a question.")
        st.stop()

    st.session_state.messages.append(("user", q))

    

    with st.spinner("Getting data...."):
        out = orch.invoke({"question": q,
                            "want_chart": want_chart,
                            "want_summary": want_summary,
                            "default_limit": int(default_limit),
                            "chat_history": st.session_state.messages[-5:],
                            "call_source":"database",
                            "drugs":st.session_state.drugs,
                            "criteria": st.session_state.criteria,
                            "active_trial_scope": st.session_state.active_trial_scope

                        }, config=cfg)
        st.session_state.current_result = out
        st.session_state.summary_refresh = True    
        st.session_state.chart_refresh = True
        st.session_state.drugs = out.get('drugs')
        st.session_state.criteria = out.get('criteria')
        st.session_state.active_trial_scope = out.get('active_trial_scope')
        #st.session_state.aact_df

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
        
        st.divider()
        if out.get("drugs"):
            df = split_payload_to_df(out["drugs"])
            with st.expander("Available Drugs"):
                st.caption(f"{len(df):,} rows · {df.shape[1]} cols")
                st.dataframe(df.head(default_limit), use_container_width=True, hide_index=True)
                st.download_button("Download Drugs CSV", df.to_csv(index=False).encode("utf-8"), "drugs_results.csv", use_container_width=True)

        st.divider()

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

        st.divider()

        st.subheader("Market Access")
        if out.get("ma_df"):
            mdf = split_payload_to_df(out["ma_df"])
            mdf = make_market_access_multiindex(mdf)
            with st.expander("Market Access data"):
                st.caption(f"{len(mdf):,} rows · {mdf.shape[1]} cols")
                mdf = mdf.reindex(level=0)
                st.dataframe(
                    mdf.head(default_limit),
                    use_container_width=True,
                    hide_index=True
                )
                st.download_button("Download Market Access CSV", mdf.to_csv(index=False).encode("utf-8"), "marketaccess_results.csv", use_container_width=True)
        if out.get("ma_sql"):
            with st.expander("Market Access SQL query"):
                st.code(out["ma_sql"], language="sql")
        if out.get("ma_sql_explain"):
            with st.expander("Market Access SQL Explained"):
                st.text(out["ma_sql_explain"])
        if out.get("ma_error"):
            st.error(f"Market Access error: {out['ma_error']}")

    # Main pane: router rationale + summary
    st.caption(out.get("router_rationale") or "No router rationale")

    if want_summary:
        st.subheader("Summary report") 
        st.markdown(out.get("final_answer") or "_No summary produced_")
        st.session_state.summary_refresh = False

    if want_chart:
        st.session_state.chart_refresh = False
        st.subheader("Data Chart")
        ftab, atab,ptab,mtab = st.tabs(["FAERS", "Clinical Trials","Pricing","Market Access"])
        with ftab:
            if out.get("faers_figure_json"):
                fig = pio.from_json(out["faers_figure_json"])
                st.plotly_chart(fig, use_container_width=True)
            elif out.get("faers_chart_error"):
                st.info(out["chart_error"])
            else:
                st.markdown("_No Chart Produced_")
        with atab:
            if out.get("aact_figure_json"):
                fig = pio.from_json(out["aact_figure_json"])
                st.plotly_chart(fig, use_container_width=True)
            elif out.get("aact_chart_error"):
                st.info(out["aact_chart_error"])
            else:
                st.markdown("_No Chart Produced_")
        with ptab:
            if out.get("pricing_figure_json"):
                fig = pio.from_json(out["pricing_figure_json"])
                st.plotly_chart(fig, use_container_width=True)
            elif out.get("pricing_chart_error"):
                st.info(out["chart_error"])
            else:
                st.markdown("_No Chart Produced_")
        with mtab:
            if out.get("ma_figure_json"):
                fig = pio.from_json(out["ma_figure_json"])
                st.plotly_chart(fig, use_container_width=True)
            elif out.get("ma_chart_error"):
                st.info(out["chart_error"])
            else:
                st.markdown("_No Chart Produced_")


