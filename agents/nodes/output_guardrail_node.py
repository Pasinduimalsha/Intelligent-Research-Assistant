from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agents.states.research_state import ResearchState
from agents.prompts import OUTPUT_GUARDRAIL_SYSTEM

class OutputGuardrailAgent:
    """Agent responsible for scanning the final draft for PII and toxicity."""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm.with_structured_output(
            schema={
                "title": "OutputGuardrailResponse",
                "type": "object",
                "properties": {
                    "is_safe": {"type": "boolean", "description": "True if the draft is safe, False if it contains PII, toxicity, or extreme hallucinations."},
                    "reason": {"type": "string", "description": "The reason why it was flagged. Empty if safe."}
                },
                "required": ["is_safe", "reason"]
            }
        )

    async def __call__(self, state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
        print("\n" + "="*50)
        print("--- OUTPUT GUARDRAIL AGENT ---")
        print("="*50)
        
        response = await self.llm.ainvoke(
            [SystemMessage(content=OUTPUT_GUARDRAIL_SYSTEM), HumanMessage(content=state.get("draft", ""))],
            config=config
        )
        
        is_safe = response.get("is_safe", True)
        reason = response.get("reason", "")
        
        if not is_safe:
            print(f"🚨 OUTPUT BLOCKED: {reason}")
            # We override the draft so the user doesn't see the bad content
            return {
                "is_safe_output": is_safe,
                "safety_violation_reason": reason,
                "draft": f"[REDACTED] The generated response was blocked by the safety guardrail. Reason: {reason}"
            }
            
        return {
            "is_safe_output": is_safe,
            "safety_violation_reason": reason
        }
