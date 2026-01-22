

from typing import TypedDict, Optional, Dict, Any,List,Literal

class OrchestratorState(TypedDict, total=False):
    question: str

    drugs : Optional[List[str]]
    #classes : Optional[List[str]]
    criteria : Optional[List[str]]
    companies : Optional[List[str]]
    user_criteria_changed: Optional[bool]
    
    want_summary:bool
    want_chart : bool
    call_source: Optional[str]
    # Router outputs (fan-out flags + sub-questions)
    need_faers: bool
    need_aact: bool
    need_pricing : bool
    need_ma : bool
    router_rationale: str
    faers_subq: Optional[str]
    aact_subq: Optional[str]
    pricing_subq: Optional[str]
    ma_subq: Optional[str]


    # FAERS artifacts
    faers_sql: Optional[str]
    faers_df: Optional[Dict[str, Any]] 
    faers_error: Optional[str]

    # AACT artifacts
    aact_sql: Optional[str]
    aact_df: Optional[Dict[str, Any]] 
    aact_error: Optional[str]
    active_trial_scope: Optional[Any]

    # PRICING artifacts
    pricing_sql: Optional[str]
    pricing_df: Optional[Dict[str, Any]] 
    pricing_error: Optional[str]

    # Market Access artifacts
    ma_sql: Optional[str]
    ma_df: Optional[Dict[str, Any]] 
    ma_error: Optional[str]


    faers_figure_json: Optional[str]
    aact_figure_json: Optional[str]
    pricing_figure_json: Optional[str]
    ma_figure_json: Optional[str]

    faers_chart_error : Optional[str]
    aact_chart_error : Optional[str]
    pricing_chart_error : Optional[str]
    ma_chart_error : Optional[str]     

    chart_source: str

    faers_sql_explain : Optional[str]
    aact_sql_explain : Optional[str]
    pricing_sql_explain : Optional[str]
    ma_sql_explain : Optional[str]

    # Final
    final_answer: Optional[str]

class AgentState(TypedDict):
    question: str
    sql: Optional[str]
    error: Optional[str]
    attempts: int
    summary: Optional[str]
    df:  Optional[Dict[str, Any]]
    sql_explain : Optional[str]
    want_chart : bool
    figure_json: Optional[str]
    call_source: Optional[str]
    drugs: Optional[List[str]]
    criteria : Optional[List[str]]
    trial_stage : Optional[str] 
    active_trial_scope: Optional[Any]
