from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from agents.states.research_state import ResearchState

class PlannerAgent:
    """Agent responsible for breaking down the query into a plan."""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    async def __call__(self, state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
        print("\n--- PLANNER AGENT ---")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert research planner. Briefly break down the user's query into a strategy to find the best information."),
            ("user", "{query}")
        ])
        chain = prompt | self.llm
        # Using ainvoke since we are in an async __call__
        result = await chain.ainvoke({"query": state["query"]})
        return {"research_plan": result.content}
