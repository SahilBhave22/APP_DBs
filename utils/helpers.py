# helpers.py
import json
import pandas as pd
from typing import Dict, Any

def df_to_split_payload(df: pd.DataFrame) -> Dict[str, Any]:
    # Pure-Python dict: {"columns": [...], "index": [...], "data": [[...], ...]}
    return json.loads(df.to_json(orient="split", date_format="iso"))

def split_payload_to_df(payload: Dict[str, Any]) -> pd.DataFrame:
    # Build the frame from data/columns, then (optionally) set the row index vector
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
