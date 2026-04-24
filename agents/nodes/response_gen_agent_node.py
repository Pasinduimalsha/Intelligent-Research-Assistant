from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from agents.states.research_state import ResearchState

class ResponseGenAgent:
    """Agent responsible for writing the research draft."""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    async def __call__(self, state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
        print("\n--- RESPONSE GEN AGENT ---")
        all_notes = "\n\n".join(state.get("notes", []))
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert technical writer. Write a concise, comprehensive draft answering the query based ONLY on the provided notes. Cite your sources.\n\nNOTES:\n{notes}"),
            ("user", "{query}")
        ])
        chain = prompt | self.llm
        result = await chain.ainvoke({"query": state["query"], "notes": all_notes})
        return {"draft": result.content}
