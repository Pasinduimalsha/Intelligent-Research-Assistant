from typing import Dict, Any, List
import json
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from agents.states.research_state import ResearchState

class ToolExecutionAgent:
    """Unified Agent responsible for executing any tool dynamically provided by the MCPClient."""
    
    def __init__(self, llm: BaseChatModel, tools: List[BaseTool]):
        self.llm = llm
        self.tools = tools
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    async def __call__(self, state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
        print("\n" + "="*50)
        print("--- UNIFIED TOOL EXECUTION AGENT ---")
        print("="*50)
        
        search_query = state["query"]
        
        if state.get("revision_count", 0) > 0:
            search_query += " latest details and context"
            
        print(f"Asking LLM to use available tools to answer: '{search_query}'")
        
        prompt = f"Please use any appropriate tools to find information regarding: {search_query}"
        ai_message = await self.llm_with_tools.ainvoke([HumanMessage(content=prompt)])
        
        new_sources = []
        new_notes = []
        
        for tool_call in ai_message.tool_calls:
            selected_tool = next((t for t in self.tools if t.name == tool_call["name"]), None)
            if selected_tool:
                try:
                    print(f"Executing tool: {tool_call['name']} with args: {tool_call['args']}")
                    tool_result = await selected_tool.ainvoke(tool_call)
                    
                    # Parse results and format them
                    parsed_result = None
                    if isinstance(tool_result, str):
                        try:
                            parsed_result = json.loads(tool_result)
                        except json.JSONDecodeError:
                            parsed_result = tool_result
                    else:
                        parsed_result = tool_result
                        
                    # Basic extraction for standard web/file results
                    if isinstance(parsed_result, list):
                        for item in parsed_result:
                            if isinstance(item, dict):
                                # Web search heuristic
                                if 'href' in item:
                                    new_sources.append(item['href'])
                                    title = item.get('title', 'Unknown')
                                    snippet = item.get('body', str(item))
                                    new_notes.append(f"Source: {item['href']}\nTitle: {title}\nContent: {snippet}")
                                # Generic
                                else:
                                    new_notes.append(f"Tool {tool_call['name']} Result: {json.dumps(item)}")
                            else:
                                new_notes.append(f"Tool {tool_call['name']} Result: {str(item)}")
                    else:
                        new_notes.append(f"Tool {tool_call['name']} Result: {str(parsed_result)}")

                except Exception as e:
                    print(f"Error executing tool {tool_call['name']}: {e}")
                    
        return {"sources": new_sources, "notes": new_notes}
