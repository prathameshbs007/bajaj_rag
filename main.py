# # """
# # Main FastAPI application
# # """

# # import time
# # import logging
# # from typing import List
# # from fastapi import FastAPI, HTTPException
# # from .models import QueryRequest, QueryResponse, Answer
# # from .pdf_handler import PDFHandler
# # from .vector_store import VectorStore
# # from .llm_service import LLMService
# # from .config import settings

# # # Configure logging
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)

# # # Initialize FastAPI app
# # app = FastAPI(
# #     title="LLM-Powered Query-Retrieval System",
# #     description="A system for answering questions about PDF documents using LLM and vector search",
# #     version="1.0.0"
# # )

# # # Initialize services
# # pdf_handler = PDFHandler()
# # vector_store = VectorStore()
# # llm_service = LLMService()

# # @app.post("/query", response_model=QueryResponse)
# # async def process_query(request: QueryRequest) -> QueryResponse:
# #     """
# #     Process a query request with document URL and questions
# #     """
# #     start_time = time.time()
    
# #     try:
# #         # Download and process PDF
# #         pdf_content = await pdf_handler.download_pdf(str(request.document_url))
# #         document_text = pdf_handler.extract_text(pdf_content)
        
# #         # Chunk the document
# #         text_chunks = pdf_handler.chunk_text(
# #             document_text,
# #             settings.CHUNK_SIZE,
# #             settings.CHUNK_OVERLAP
# #         )
        
# #         # Generate embeddings for chunks
# #         chunk_embeddings = await llm_service.generate_embeddings(text_chunks)
        
# #         # Store chunks in vector database
# #         await vector_store.upsert_vectors(
# #             vectors=chunk_embeddings,
# #             texts=text_chunks,
# #             metadata={'document_url': str(request.document_url)}
# #         )
        
# #         # Process each question
# #         answers: List[Answer] = []
# #         for question in request.questions:
# #             # Generate embedding for question
# #             question_embedding = await llm_service.generate_embeddings([question.text])
            
# #             # Retrieve relevant chunks
# #             relevant_chunks = await vector_store.query_vectors(
# #                 query_vector=question_embedding[0],
# #                 top_k=settings.TOP_K_RESULTS
# #             )
            
# #             # Generate answer
# #             answer_data = await llm_service.generate_answer(
# #                 question.text,
# #                 relevant_chunks
# #             )
            
# #             answers.append(Answer(
# #                 question=question.text,
# #                 answer=answer_data['answer'],
# #                 context_chunks=answer_data['context_chunks'],
# #                 confidence_score=answer_data['confidence_score']
# #             ))
        
# #         # Calculate processing time
# #         processing_time = time.time() - start_time
        
# #         return QueryResponse(
# #             document_url=request.document_url,
# #             answers=answers,
# #             processing_time=processing_time
# #         )
        
# #     except Exception as e:
# #         logger.error(f"Error processing query: {str(e)}")
# #         raise HTTPException(status_code=500, detail=str(e))

# # @app.get("/health")
# # async def health_check():
# #     """
# #     Health check endpoint
# #     """
# #     return {"status": "healthy"}

# # @app.get("/test-openai")
# # async def test_openai():
# #     """
# #     Test OpenAI connectivity
# #     """
# #     try:
# #         llm_service = LLMService()
# #         response = await llm_service.generate_embeddings(["test"])
# #         return {"status": "success", "message": "OpenAI API is working"}
# #     except Exception as e:
# #         logger.error(f"OpenAI test failed: {str(e)}")
# #         raise HTTPException(status_code=500, detail=str(e))





# """
# Main FastAPI application for the HackRx Challenge
# """

# import time
# import logging
# from typing import List
# from fastapi import FastAPI, HTTPException, Depends
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# from .models import HackRxRequest, HackRxResponse
# from .pdf_handler import PDFHandler
# from .vector_store import VectorStore
# from .llm_service import LLMService
# from .config import settings

# security = HTTPBearer()

# def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
#     """
#     Dependency function to verify the bearer token.
#     For this hackathon, we can use a simple static token.
#     """
#     HACKATHON_API_KEY = "your-secret-bearer-token-12345"
#     if credentials.scheme != "Bearer" or credentials.credentials != HACKATHON_API_KEY:
#         raise HTTPException(
#             status_code=403,
#             detail="Invalid or missing Bearer token"
#         )
#     return credentials

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI(
#     title="HackRx LLM Query-Retrieval System",
#     description="API to answer questions about PDF documents for the HackRx challenge.",
#     version="1.0.1"
# )

# pdf_handler = PDFHandler()
# vector_store = VectorStore()
# llm_service = LLMService()

# @app.post("/hackrx/run", response_model=HackRxResponse)
# async def process_hackrx_query(
#     request: HackRxRequest,
#     token: dict = Depends(verify_token)
# ) -> HackRxResponse:
#     """
#     Processes a document and a list of questions to return answers.
#     """
#     try:
#         pdf_content = await pdf_handler.download_pdf(request.documents)
#         document_text = pdf_handler.extract_text(pdf_content)

#         text_chunks = pdf_handler.chunk_text(
#             document_text,
#             settings.CHUNK_SIZE,
#             settings.CHUNK_OVERLAP
#         )

#         chunk_embeddings = await llm_service.generate_embeddings(text_chunks)

#         await vector_store.upsert_vectors(
#             vectors=chunk_embeddings,
#             texts=text_chunks,
#             metadata={'document_url': request.documents}
#         )

#         final_answers: List[str] = []
#         for question_text in request.questions:
#             question_embedding = await llm_service.generate_embeddings([question_text])

#             relevant_chunks = await vector_store.query_vectors(
#                 query_vector=question_embedding[0],
#                 top_k=settings.TOP_K_RESULTS
#             )

#             answer_data = await llm_service.generate_answer(
#                 question_text,
#                 relevant_chunks
#             )

#             final_answers.append(answer_data['answer'])

#         return HackRxResponse(answers=final_answers)

#     except Exception as e:
#         logger.error(f"Critical error processing request: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/health")
# async def health_check():
#     """A simple endpoint to check if the API is running."""
#     return {"status": "healthy"}



"""
Main FastAPI application for the HackRx Challenge
"""

import time
import logging
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
    """
    Dependency function to verify the bearer token.
    """
    HACKATHON_API_KEY = "your-secret-bearer-token-12345"
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
    version="1.0.1"
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
    Processes a document and a list of questions to return answers.
    """
    try:
        pdf_content = await pdf_handler.download_pdf(request.documents)
        document_text = pdf_handler.extract_text(pdf_content)

        text_chunks = pdf_handler.chunk_text(
            document_text,
            settings.CHUNK_SIZE,
            settings.CHUNK_OVERLAP
        )

        chunk_embeddings = await llm_service.generate_embeddings(text_chunks)

        await vector_store.upsert_vectors(
            vectors=chunk_embeddings,
            texts=text_chunks,
            metadata={'document_url': request.documents}
        )

        final_answers: List[str] = []
        for question_text in request.questions:
            question_embedding = await llm_service.generate_embeddings([question_text])
            relevant_chunks = await vector_store.query_vectors(
                query_vector=question_embedding[0],
                top_k=settings.TOP_K_RESULTS
            )
            answer_data = await llm_service.generate_answer(
                question_text,
                relevant_chunks
            )
            final_answers.append(answer_data['answer'])

        return HackRxResponse(answers=final_answers)

    except Exception as e:
        logger.error(f"Critical error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """A simple endpoint to check if the API is running."""
    return {"status": "healthy"}
