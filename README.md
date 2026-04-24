# Intelligent-Research-Assistant

A FastAPI-based backend application powered by LangGraph and the Model Context Protocol (MCP) for autonomous research.

## Architecture Diagram



## Setup Instructions

1. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment**:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Create a `.env` file in the root directory based on `.env.example` (if present) and ensure `OPENAI_API_KEY` is set.

## Running the Application

Start the unified FastAPI server:
```bash
python main.py
```

The REST API and internal FastMCP host will be accessible at `http://0.0.0.0:8000/`.

## Testing the Application
Use the provided `client.py` to test the LangGraph pipeline via Server-Sent Events (SSE):
```bash
python client.py
```

## Project Structure
- `main.py`: Main FastAPI application entry point, mounting LangGraph routers and FastMCP.
- `client.py`: Frontend test script to consume SSE endpoints.
- `mcp/`: Contains all Model Context Protocol Subprocess Servers (`web_mcp.py`, `file_mcp.py`).
- `agents/`: Contains all LangGraph Nodes, States, and Graph builders.
- `controllers/`: API endpoints connecting FastAPI to LangGraph.
- `config/`: Pydantic settings loading from `.env`.