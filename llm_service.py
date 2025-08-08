"""
LLM service for embeddings and answer generation using Google Gemini
and local Sentence Transformers.
"""

import logging
from typing import List, Dict, Any

import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        """Initialize Google Gemini client and load local embedding model."""
        try:
            logger.info("Initializing Google Gemini client...")
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.genai_model = genai.GenerativeModel('gemini-1.5-flash-latest')
            logger.info("Google Gemini client initialized successfully.")

            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}...")
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize LLMService: {str(e)}")
            raise

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using the local model.
        """
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts locally.")
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            return [embedding.tolist() for embedding in embeddings]
        except Exception as e:
            logger.error(f"Unexpected error generating embeddings: {str(e)}")
            raise

    def _prepare_context(self, chunks: List[Dict[str, Any]], max_tokens: int) -> str:
        """
        Prepare context for the LLM while respecting token limits.
        (This function remains largely the same)
        """
        context = ""
        max_chars = max_tokens * 4
        current_chars = 0

        sorted_chunks = sorted(chunks, key=lambda x: x['score'], reverse=True)

        for chunk in sorted_chunks:
            chunk_text = chunk['text']
            if current_chars + len(chunk_text) > max_chars:
                break
            context += f"\n\n---\n\n{chunk_text}"
            current_chars += len(chunk_text)
        return context

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_answer(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate answer using Google Gemini Pro based on context chunks.
        """
        try:
            context = self._prepare_context(
                context_chunks,
                settings.MAX_CONTEXT_TOKENS
            )

            prompt = f"""
            Based ONLY on the context provided below, answer the following question.
            Do not use any external knowledge. If the answer is not found in the context,
            state that clearly.

            CONTEXT:
            {context}

            QUESTION:
            {question}

            ANSWER:
            """

            logger.info("Generating answer with Google Gemini Pro...")
            response = self.genai_model.generate_content(prompt)

            avg_confidence = sum(chunk['score'] for chunk in context_chunks) / len(context_chunks) if context_chunks else 0

            return {
                'answer': response.text.strip(),
                'context_chunks': [chunk['text'] for chunk in context_chunks],
                'confidence_score': avg_confidence
            }

        except Exception as e:
            logger.error(f"Error generating answer with Gemini: {str(e)}")
            raise
