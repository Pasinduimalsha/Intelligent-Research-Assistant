from typing import TypedDict, List, Annotated
import operator

class ResearchState(TypedDict):
    query: str
    original_query: str
    research_plan: str
    context_documents: list[dict]
    sources: Annotated[List[str], operator.add]
    notes: Annotated[List[str], operator.add]
    draft: str
    revision_count: int
    is_complete: bool
    is_safe_input: bool
    is_safe_output: bool
    safety_violation_reason: str
    needed_sources: List[str]
    followup_query: str
    needs_reroute: bool
    reroute_count: int
