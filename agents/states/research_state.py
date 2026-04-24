from typing import TypedDict, List, Annotated
import operator

class ResearchState(TypedDict):
    query: str
    research_plan: str
    sources: Annotated[List[str], operator.add]
    notes: Annotated[List[str], operator.add]
    draft: str
    revision_count: int
    is_complete: bool
