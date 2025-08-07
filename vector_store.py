"""
Pinecone vector store integration
"""

import logging
import uuid
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from config import settings

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        """Initialize Pinecone client"""
        try:
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            self.index = self.pc.Index(settings.PINECONE_INDEX_NAME)
            logger.info("Pinecone client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {str(e)}")
            raise

    async def upsert_vectors(
        self,
        vectors: List[List[float]],
        texts: List[str],
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Upload vectors to Pinecone
        """
        try:
            # Create vector records with unique IDs and optional metadata
            records = []
            for i, (vector, text) in enumerate(zip(vectors, texts)):
                record = {
                    'id': f'chunk_{uuid.uuid4().hex}_{i}',  # Generate unique IDs
                    'values': vector,
                    'metadata': {
                        'text': text,
                        **(metadata or {})
                    }
                }
                records.append(record)

            # Upsert in batches of 100
            batch_size = 100
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                self.index.upsert(vectors=batch)

        except Exception as e:
            logger.error(f"Error upserting vectors to Pinecone: {str(e)}")
            raise

    async def query_vectors(
        self,
        query_vector: List[float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query similar vectors from Pinecone
        """
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )
            
            # Extract and return relevant chunks with scores
            matches = []
            for match in results.matches:
                matches.append({
                    'text': match.metadata['text'],
                    'score': match.score
                })
            
            return matches

        except Exception as e:
            logger.error(f"Error querying vectors from Pinecone: {str(e)}")
            raise

    async def delete_vectors(self, ids: List[str] = None) -> None:
        """
        Delete vectors from Pinecone
        """
        try:
            if ids:
                self.index.delete(ids=ids)
            else:
                self.index.delete(delete_all=True)
        except Exception as e:
            logger.error(f"Error deleting vectors from Pinecone: {str(e)}")
            raise
