from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from agents.states.research_state import ResearchState
from agents.prompts import REVIEWER_SYSTEM

class ReviewerAgent:
    """Agent responsible for evaluating the draft against the query."""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    async def __call__(self, state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
        print("\n--- REVIEWER AGENT ---")
        prompt = ChatPromptTemplate.from_messages([
            ("system", REVIEWER_SYSTEM),
            ("user", "QUERY: {query}\n\nDRAFT: {draft}")
        ])
        chain = prompt | self.llm
        result = await chain.ainvoke({"query": state["query"], "draft": state.get("draft", "")})
        
        feedback = result.content
        print(f"Reviewer Feedback: {feedback[:100]}...")
        
        if "YES" in feedback.upper()[:10]:
            return {"is_complete": True}
        else:
            return {
                "is_complete": False, 
                "revision_count": state.get("revision_count", 0) + 1,
                "research_plan": f"Feedback to address: {feedback}"
            }
