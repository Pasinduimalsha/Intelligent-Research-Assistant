from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agents.states.research_state import ResearchState

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
        print("\n--- OUTPUT GUARDRAIL AGENT ---")
        
        system_prompt = """You are a strict security guardrail for an AI research assistant.
Your job is to analyze the final drafted response and detect any of the following:
1. Personally Identifiable Information (PII) like real phone numbers, SSNs, or private emails.
2. Toxic, offensive, or harmful language.

If the draft is safe and professional, return is_safe=True.
If the draft violates policy, return is_safe=False and provide a brief reason."""

        response = await self.llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=state.get("draft", ""))],
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
