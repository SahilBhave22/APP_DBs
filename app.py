# app_orchestrator_parallel.py
import os, json
import streamlit as st
import pandas as pd
from urllib.parse import quote_plus

from fdaers_agent import build_fdaers_agent
from clinicaltrials_agent import build_clinicaltrials_agent
from orchestrator_parallel_subq import build_orchestrator_parallel_subq
from IPython.display import Image

st.set_page_config(page_title="FDAERS + Clinical Trials ", layout="wide")

OPENAI_KEY     = st.secrets.get("openai_api_key")

#DB_PASSWORD = st.secrets.get("db_password") 
#password = quote_plus(DB_PASSWORD)


PUBLIC_IP = st.secrets.get("public_ip")  # Cloud SQL public IP
DB_USER   = st.secrets.get("db_user")
DB_PASS   = st.secrets.get("db_pass")
DB_NAME_AACT   = st.secrets.get("db_name_aact")
DB_NAME_FDAERS = st.secrets.get("db_name_fdaers")

FAERS_DB_URL =  f"postgresql://{DB_USER}:{DB_PASS}@{PUBLIC_IP}:5432/{DB_NAME_FDAERS}?sslmode=require&connect_timeout=70"
AACT_DB_URL = f"postgresql://{DB_USER}:{DB_PASS}@{PUBLIC_IP}:5432/{DB_NAME_AACT}?sslmode=require&connect_timeout=70"


#FAERS_DB_URL   = st.secrets.get("faers_db_url")
#AACT_DB_URL    = st.secrets.get("aact_db_url")

for k in ("LANGCHAIN_TRACING","LANGCHAIN_ENDPOINT","LANGCHAIN_API_KEY","LANGCHAIN_PROJECT"):
    if k in st.secrets:
        os.environ[k] = st.secrets[k]

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
def get_apps(safe_mode=True, default_limit=200):
    if not FAERS_DB_URL or not AACT_DB_URL:
        st.error("Missing FAERS_DB_URL or AACT_DB_URL in secrets.")
        st.stop()

    faers_catalog = load_json("faers_schema_catalog", "faers_schema_catalog.json")
    aact_catalog  = load_json("clinicaltrials_schema_catalog",  "aact_schema_catalog.json")

    faers_app = build_fdaers_agent(faers_catalog, fdaers_db_url=FAERS_DB_URL, safe_mode=safe_mode, default_limit=default_limit)
    aact_app  = build_clinicaltrials_agent(aact_catalog,  clinicaltrials_db_url=AACT_DB_URL,  safe_mode=safe_mode, default_limit=default_limit)
    orch_app  = build_orchestrator_parallel_subq(faers_app, aact_app)
    #Image(orch_app().get_graph().draw_mermaid_png())
    return orch_app

st.title("FDAERS + Clinical Trials")

with st.sidebar:
    st.header("Settings")
    safe_mode = st.toggle("Safe mode (SELECT/WITH only)", value=True)
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

    orch = get_apps(safe_mode=safe_mode, default_limit=int(default_limit))
    out = orch.invoke({"question": q})

    # Sidebar results per source
    with st.sidebar:
        st.subheader("FDAERS")
        if isinstance(out.get("faers_df"), pd.DataFrame):
            fdf = out["faers_df"]
            with st.expander("FDAERS data"):
                st.caption(f"{len(fdf):,} rows · {fdf.shape[1]} cols")
                st.dataframe(fdf.head(500), use_container_width=True, hide_index=True)
                st.download_button("Download FDAERS CSV", fdf.to_csv(index=False).encode("utf-8"), "faers_results.csv", use_container_width=True)
        if out.get("faers_sql"):
            with st.expander("FDAERS SQL query"):
                st.code(out["faers_sql"], language="sql")
        if out.get("faers_subq"):
            with st.expander("FAERS Sub-question"):
                st.text(out["faers_subq"])
        if out.get("faers_error"):
            st.error(f"FAERS error: {out['faers_error']}")

        st.divider()

        st.subheader("Clinical Trials")
        if isinstance(out.get("aact_df"), pd.DataFrame):
            adf = out["aact_df"]
            with st.expander("Clinical Trials data"):
                st.caption(f"{len(adf):,} rows · {adf.shape[1]} cols")
                st.dataframe(adf.head(500), use_container_width=True, hide_index=True)
                st.download_button("Download Clinical Trials CSV", adf.to_csv(index=False).encode("utf-8"), "aact_results.csv", use_container_width=True)
        if out.get("aact_sql"):
            with st.expander("Clinical Trials SQL query"):
                st.code(out["aact_sql"], language="sql")
        if out.get("aact_subq"):
            with st.expander("Clinical Trials Sub-question"):
                st.text(out["aact_subq"])
        if out.get("aact_error"):
            st.error(f"AACT error: {out['aact_error']}")

    # Main pane: router rationale + summary
    st.caption(out.get("router_rationale") or "No router rationale")
    st.subheader("Summary report")
    # with st.expander("Show orchestrator diagram", expanded=False):
    #     st.image(orch.get_graph().draw_mermaid_png()) 
    st.markdown(out.get("final_answer") or "_No summary produced_")
