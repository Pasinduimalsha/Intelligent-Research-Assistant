from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agents.states.research_state import ResearchState
from agents.prompts import INPUT_GUARDRAIL_SYSTEM

class InputGuardrailAgent:
    """Agent responsible for checking for prompt injection and malicious queries."""
    
    def __init__(self, llm: ChatOpenAI):
        # Use structured output to get a clean JSON response
        self.llm = llm.with_structured_output(
            schema={
                "title": "InputGuardrailResponse",
                "type": "object",
                "properties": {
                    "is_safe": {"type": "boolean", "description": "True if the query is safe, False if it is malicious, a prompt injection, or asks for inappropriate content."},
                    "reason": {"type": "string", "description": "The reason why the query was flagged as unsafe. Empty if safe."}
                },
                "required": ["is_safe", "reason"]
            }
        )

    async def __call__(self, state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
        print("\n" + "="*50)
        print("--- INPUT GUARDRAIL AGENT ---")
        print("="*50)
        query = state["query"]
        
        response = await self.llm.ainvoke(
            [SystemMessage(content=INPUT_GUARDRAIL_SYSTEM), HumanMessage(content=state["query"])],
            config=config
        )
        
        is_safe = response.get("is_safe", True)
        reason = response.get("reason", "")
        
        if not is_safe:
            print(f"🚨 INPUT BLOCKED: {reason}")
            
        return {
            "is_safe_input": is_safe,
            "safety_violation_reason": reason
        }
