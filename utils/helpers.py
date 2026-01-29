# helpers.py
import json
import pandas as pd
import re
from typing import TypedDict, Optional, Dict, Any, Literal

def df_to_split_payload(df: pd.DataFrame) -> Dict[str, Any]:
    
    return json.loads(df.to_json(orient="split", date_format="iso"))

def split_payload_to_df(payload: Dict[str, Any]) -> pd.DataFrame:
    
    if(payload is None):
        return None
    df = pd.DataFrame(payload["data"], columns=payload["columns"])
    idx = payload.get("index")
    if idx is not None:
        try:
            df.index = pd.Index(idx)
        except Exception:
            # If index is malformed, just keep the default RangeIndex
            pass
    return df

def make_column_inventory(catalog: Dict[str, Any]) -> str:
    lines = []
    for t in catalog.get("tables", []):
        cols = ", ".join([c["name"] for c in t.get("columns", [])])
        lines.append(f"- {t['name']}: {cols}")
    return "\n".join(lines) if lines else "None"

def make_join_hints(catalog: Dict[str, Any]) -> str:
    rels = catalog.get("relationships", [])
    if not rels:
        return "None"
    return "\n".join([
        f"- {r['from_table']}.{','.join(r['from_columns'])} ↔ {r['to_table']}.{','.join(r['to_columns'])} [{r.get('type','')}]"
        for r in rels
    ])

def clean_sql(sql: str) -> str:
    """Strip ```sql fences and trim."""
    if not sql:
        return ""
    s = sql.strip()
    s = re.sub(r"^```[a-zA-Z]*\s*", "", s)  # remove opening ``` / ```sql
    s = re.sub(r"\s*```$", "", s)           # remove trailing ```
    s = re.sub(r"\bdo\b", "doc", s)
    return s.strip()


def validate_sql(sql: str) -> Optional[str]:
    DISALLOWED = re.compile(r"\b(insert|update|delete|drop|alter|create|copy|grant|revoke|truncate|vacuum)\b", re.I)
    
    PARAM_TOKENS = re.compile(
    r"(?<!:):[a-zA-Z_][a-zA-Z0-9_]*"  # :param but NOT ::type
    r"|@[a-zA-Z_][a-zA-Z0-9_]*"       # @param
    r"|\$\d+"                         # $1, $2...
    r"|\?",                           # ? param (if you want to forbid it)
    re.MULTILINE,
    )

    
    s = (sql or "").strip()
    if not re.match(r"^\s*(with|select)\b", s, flags=re.I | re.S):
        return "Only WITH/SELECT queries are allowed."
    
    if DISALLOWED.search(s):
        return "Disallowed SQL keyword detected."
    
    if PARAM_TOKENS.search(s):
        return "Parameters are not allowed. Inline all literal values (no :param, @param, $1, ?)."

    return None


import pandas as pd

def make_market_access_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts flat payer columns into a 2-level header:
    Level 0: payer name
    Level 1: tier / req
    """
    new_cols = []

    for col in df.columns:
        if col in ["brand_name", "generic_name"]:
            new_cols.append((col, ""))  # keep metadata columns flat
        else:
            payer, metric = col.split("_", 1)
            new_cols.append((payer.capitalize(), metric))

    df.columns = pd.MultiIndex.from_tuples(
        new_cols,
        names=["payer", "field"]
    )

    return df
