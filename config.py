"""
Configuration management for the LLM Query System
"""

import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GOOGLE_API_KEY: str

    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str
    PINECONE_INDEX_NAME: str
    
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 5
    
    MAX_CONTEXT_TOKENS: int = 4000
    MAX_RESPONSE_TOKENS: int = 1000
    
    class Config:
        env_file = ".env"

settings = Settings()
