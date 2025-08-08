# main.py

import time
import logging
import asyncio
from typing import List
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from models import HackRxRequest, HackRxResponse
from pdf_handler import PDFHandler
from vector_store import VectorStore
from llm_service import LLMService
from config import settings

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency function to verify the bearer token."""
    HACKATHON_API_KEY = "8842022c10b3749bd60d03f2acb5fa2006892edf7bebf19063e17eae760839ca"
    if credentials.scheme != "Bearer" or credentials.credentials != HACKATHON_API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing Bearer token"
        )
    return credentials

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HackRx LLM Query-Retrieval System",
    description="API to answer questions about PDF documents for the HackRx challenge.",
    version="1.1.0"
)

pdf_handler = PDFHandler()
vector_store = VectorStore()
llm_service = LLMService()


@app.post("/hackrx/run", response_model=HackRxResponse)
async def process_hackrx_query(
    request: HackRxRequest,
    token: dict = Depends(verify_token)
) -> HackRxResponse:
    """
    Optimized: process document once, answer multiple questions in parallel.
    Should stay under 15 seconds for Railway's timeout.
    """
    start_time = time.time()
    try:
        logger.info("Step 1: Downloading PDF...")
        pdf_content = await pdf_handler.download_pdf(request.documents)

        logger.info("Step 2: Extracting text...")
        document_text = pdf_handler.extract_text(pdf_content)

        logger.info("Step 3: Chunking text...")
        text_chunks = pdf_handler.chunk_text(
            document_text,
            settings.CHUNK_SIZE,
            settings.CHUNK_OVERLAP
        )

        logger.info(f"Step 4: Generating embeddings for {len(text_chunks)} chunks...")
        chunk_embeddings = await llm_service.generate_embeddings(text_chunks)

        logger.info("Step 5: Upserting vectors to Pinecone...")
        await vector_store.upsert_vectors(
            vectors=chunk_embeddings,
            texts=text_chunks,
            metadata={'document_url': request.documents}
        )

        logger.info("Step 6: Handling questions in parallel...")

        async def handle_question(question_text: str):
            q_embed = await llm_service.generate_embeddings([question_text])
            relevant_chunks = await vector_store.query_vectors(
                query_vector=q_embed[0],
                top_k=settings.TOP_K_RESULTS
            )
            ans = await llm_service.generate_answer(question_text, relevant_chunks)
            return ans['answer']

        final_answers = await asyncio.gather(
            *[handle_question(q) for q in request.questions]
        )

        logger.info(f"✅ Completed in {time.time() - start_time:.2f} seconds")
        return HackRxResponse(answers=final_answers)

    except Exception as e:
        logger.error(f"Critical error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """A simple endpoint to check if the API is running."""
    return {"status": "healthy"}
