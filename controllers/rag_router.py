import uuid
import io
import pandas as pd
from fastapi import APIRouter, Request, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from qdrant_client.http import models
from util.logger import Logger

logger = Logger().get_logger("controllers.rag_router")

router = APIRouter(prefix="/rag", tags=["RAG"])

from typing import Union, List

async def extract_text_from_file(file: UploadFile) -> Union[str, List[str]]:
    """Extract text content from various file types. Returns list for structured data."""
    content = await file.read()
    filename = file.filename.lower()
    
    try:
        if filename.endswith(".txt"):
            return content.decode("utf-8")
        elif filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
            return [
                ", ".join([f"{col}: {val}" for col, val in row.items()])
                for _, row in df.iterrows()
            ]
        elif filename.endswith((".xlsx", ".xls")):
            # Read all sheets
            excel_data = pd.read_excel(io.BytesIO(content), sheet_name=None)
            all_rows = []
            for sheet_name, df in excel_data.items():
                for _, row in df.iterrows():
                    row_str = ", ".join([f"{col}: {val}" for col, val in row.items()])
                    all_rows.append(f"Sheet: {sheet_name} | {row_str}")
            return all_rows
        else:
            return content.decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {str(e)}")

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Simple character-based chunking with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

@router.post("/ingest")
async def ingest_document(
    request: Request, 
    file: UploadFile = File(None),
    content: str = Form(None),
    metadata_str: str = Form("{}"),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200)
):
    """
    Endpoint to ingest text or files into Qdrant with chunking.
    """
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")

    config = orchestrator.config
    
    # Determine content
    final_content = None
    source_name = "manual_input"
    if file:
        final_content = await extract_text_from_file(file)
        source_name = file.filename
    elif content:
        final_content = content
    else:
        try:
            body = await request.json()
            final_content = body.get("content", "")
        except:
            raise HTTPException(status_code=400, detail="No content or file provided")

    if final_content is None or (isinstance(final_content, str) and not final_content):
        raise HTTPException(status_code=400, detail="Content is empty")

    # 1. Chunk/Flatten the content
    if isinstance(final_content, list):
        # For structured data, each row is already a "chunk"
        chunks = final_content
    else:
        chunks = chunk_text(final_content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
    logger.info(f"Processing {len(chunks)} segments from {source_name}.")

    from services.rag.embedding_service import EmbeddingService
    from services.rag.qdrant_service import QdrantService
    
    embedding_service = EmbeddingService(config)
    qdrant_service = QdrantService(config)
    
    if not qdrant_service.client:
        raise HTTPException(status_code=500, detail="Qdrant not configured")

    # 2. Ensure collection exists
    qdrant_service.ensure_collection(
        config.qdrant_collection_name, 
        config.embedding_dimension
    )
    
    # 3. Generate embeddings for all chunks
    # We use generate_embeddings_batch for efficiency
    vectors = await embedding_service.generate_embeddings_batch(chunks)
    
    # 4. Prepare points for Qdrant
    import json
    try:
        payload_metadata = json.loads(metadata_str)
    except:
        payload_metadata = {}

    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        point_id = str(uuid.uuid4())
        payload = {
            "content": chunk,
            "source": source_name,
            "chunk_index": i,
            **payload_metadata
        }
        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
        )
    
    # 5. Bulk Upsert
    await qdrant_service.upsert(
        config.qdrant_collection_name,
        points=points
    )
    
    return {
        "status": "success", 
        "chunks_processed": len(chunks),
        "source": source_name
    }
