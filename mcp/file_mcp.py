import os
import json
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

app = Server('file-mcp')

@app.list_tools()
async def list_tools():
    return [
        Tool(name='list_directory',
             description='List contents of a directory',
             inputSchema={'type':'object','properties':{'path':{'type':'string'}},'required':['path']}),
        Tool(name='read_file',
             description='Read file content',
             inputSchema={'type':'object','properties':{'path':{'type':'string'}},'required':['path']})
    ]

@app.call_tool()
async def call_tool(name, arguments):
    if name == 'list_directory':
        try:
            contents = os.listdir(arguments['path'])
            return [TextContent(type='text', text=json.dumps(contents))]
        except Exception as e:
            return [TextContent(type='text', text=f"Error: {str(e)}")]
            
    if name == 'read_file':
        try:
            with open(arguments['path'], 'r') as f:
                content = f.read()
            return [TextContent(type='text', text=content)]
        except Exception as e:
            return [TextContent(type='text', text=f"Error: {str(e)}")]

async def main():
    async with stdio_server() as (r, w):
        await app.run(r, w, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
