# db.py
from functools import lru_cache
import streamlit as st
import pandas as pd
from sqlalchemy import text
import sqlalchemy

from google.oauth2 import service_account
from google.cloud.sql.connector import Connector, IPTypes

# ---- Load base config ----
INSTANCE = st.secrets["gcp"]["instance_connection_name"]

# Service account as a TOML dict (no json.loads needed)
credentials = service_account.Credentials.from_service_account_info(
    dict(st.secrets["gcp"]["service_account"])
)

APP_USER = st.secrets["db_creds"]["db_user"]
APP_PASS = st.secrets["db_creds"]["db_pass"]

DBS = {
    "aact":    st.secrets["dbs"]["db_name_aact"],
    "fdaers":  st.secrets["dbs"]["db_name_fdaers"],
    "pricing": st.secrets["dbs"]["db_name_pricing"],
}

# One shared connector
_connector = Connector(credentials=credentials)

def _creator(db_name: str):
    def create_conn():
        return _connector.connect(
            INSTANCE,
            driver="pg8000",
            user=APP_USER,
            password=APP_PASS,
            db=db_name,
            ip_type=IPTypes.PUBLIC,  # secure tunnel over public IP; no allowlists needed
        )
    return create_conn

@lru_cache(maxsize=8)
def get_engine(db_key: str) -> sqlalchemy.Engine:
    if db_key not in DBS:
        raise ValueError(f"Unknown db_key: {db_key}. Use one of {list(DBS)}")
    db_name = DBS[db_key]
    return sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=_creator(db_name),
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=2,
        pool_timeout=30,
    )

def exec_sql(sql: str, db_key: str, params: dict | None = None) -> pd.DataFrame:
    eng = get_engine(db_key)
    with eng.connect() as conn:
        # Optional: set schema if needed
        # conn.execute(text("SET search_path TO public"))
        return pd.read_sql(text(sql), conn, params=params)
