"""
LLM service for embeddings and answer generation
"""

import logging
from typing import List, Dict, Any
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config import settings

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        """Initialize OpenAI client"""
        try:
            logger.info(f"Initializing OpenAI client with API key: {settings.OPENAI_API_KEY[:8]}...")
            import httpx
            # Create a custom httpx client with longer timeouts and keep-alive
            timeout = httpx.Timeout(30.0, connect=10.0)
            http_client = httpx.Client(
                timeout=timeout,
                transport=httpx.HTTPTransport(retries=3)
            )
            
            # Initialize OpenAI client - let OpenAI handle authentication
            self.client = openai.OpenAI(
                api_key=settings.OPENAI_API_KEY.strip(),  # Ensure no whitespace
                max_retries=3,  # Add retry configuration
                timeout=30.0  # Set timeout
            )
            
            # Test the connection
            logger.info("Testing OpenAI connection...")
            models = self.client.models.list()
            logger.info("Successfully connected to OpenAI API")
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIError))
    )
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        """
        try:
            logger.info(f"Generating embeddings using model: {settings.OPENAI_EMBEDDING_MODEL}")
            logger.info(f"Number of texts to process: {len(texts)}")
            
            # Process texts in batches to handle rate limits
            batch_size = 2  # Reduced batch size to avoid rate limits
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1} of {(len(texts) + batch_size - 1)//batch_size}")
                
                try:
                    response = self.client.embeddings.create(
                        model=settings.OPENAI_EMBEDDING_MODEL,
                        input=batch
                    )
                    batch_embeddings = [embedding.embedding for embedding in response.data]
                    all_embeddings.extend(batch_embeddings)
                    logger.info(f"Successfully generated embeddings for batch {i//batch_size + 1}")
                except Exception as e:
                    logger.error(f"Failed to generate embeddings for batch: {str(e)}")
                    raise
            
            logger.info("Successfully generated all embeddings")
            return all_embeddings
            
        except openai.RateLimitError as e:
            logger.error(f"OpenAI Rate limit exceeded: {str(e)}")
            raise
        except openai.AuthenticationError as e:
            logger.error(f"OpenAI Authentication error: {str(e)}")
            raise
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating embeddings: {str(e)}")
            raise

    def _prepare_context(self, chunks: List[Dict[str, Any]], max_tokens: int) -> str:
        """
        Prepare context for the LLM while respecting token limits
        Using a simple character-based approximation instead of tiktoken
        Assuming average of 4 characters per token as a conservative estimate
        """
        context = ""
        max_chars = max_tokens * 4  # Approximate chars per token
        current_chars = 0

        # Sort chunks by score in descending order
        sorted_chunks = sorted(chunks, key=lambda x: x['score'], reverse=True)

        for chunk in sorted_chunks:
            chunk_text = chunk['text']
            chunk_chars = len(chunk_text)

            if current_chars + chunk_chars > max_chars:
                break

            context += f"\n\nContext chunk:\n{chunk_text}"
            current_chars += chunk_chars

        return context

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIError))
    )
    async def generate_answer(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate answer using GPT-4 based on context chunks
        """
        try:
            # Prepare context within token limits
            context = self._prepare_context(
                context_chunks,
                settings.MAX_CONTEXT_TOKENS
            )

            # Create the prompt
            prompt = f"""Please answer the following question based ONLY on the provided context. 
If the context doesn't contain enough information to answer the question fully, 
say so explicitly. Do not make up or infer information that's not in the context.

Question: {question}

{context}

Answer:"""

            response = self.client.chat.completions.create(
                model=settings.OPENAI_COMPLETION_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based solely on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=settings.MAX_RESPONSE_TOKENS,
                temperature=0.3,
                top_p=1.0,
            )

            # Calculate average confidence score from chunks used
            avg_confidence = sum(chunk['score'] for chunk in context_chunks) / len(context_chunks)

            return {
                'answer': response.choices[0].message.content.strip(),
                'context_chunks': [chunk['text'] for chunk in context_chunks],
                'confidence_score': avg_confidence
            }

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise