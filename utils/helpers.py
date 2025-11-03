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
    return s.strip()


def validate_sql(sql: str) -> Optional[str]:
    DISALLOWED = re.compile(r"\b(insert|update|delete|drop|alter|create|copy|grant|revoke|truncate|vacuum)\b", re.I)
    s = (sql or "").strip()
    if not re.match(r"^\s*(with|select)\b", s, flags=re.I | re.S):
        return "Only WITH/SELECT queries are allowed."
    if DISALLOWED.search(s):
        return "Disallowed SQL keyword detected."

    return None
