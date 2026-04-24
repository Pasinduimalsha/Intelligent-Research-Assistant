import json
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from duckduckgo_search import DDGS

app = Server('web-mcp')

@app.list_tools()
async def list_tools():
    return [
        Tool(name='web_search',
             description='Search the web for current information using DuckDuckGo',
             inputSchema={'type':'object',
                          'properties':{'query':{'type':'string'},
                                        'max_results':{'type':'integer','default':5}},
                          'required':['query']})
    ]

@app.call_tool()
async def call_tool(name, arguments):
    if name == 'web_search':
        try:
            query = arguments['query']
            max_results = arguments.get('max_results', 5)
            results = list(DDGS().text(query, max_results=max_results))
            return [TextContent(type='text', text=json.dumps(results))]
        except Exception as e:
            return [TextContent(type='text', text=f"Error searching web: {str(e)}")]

async def main():
    async with stdio_server() as (r, w):
        await app.run(r, w, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
