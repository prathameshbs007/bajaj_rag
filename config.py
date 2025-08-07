"""
Configuration management for the LLM Query System
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # OpenAI Configuration
    OPENAI_API_KEY: str
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-ada-002"
    OPENAI_COMPLETION_MODEL: str = "gpt-3.5-turbo"
    
    # Pinecone Configuration
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str
    PINECONE_INDEX_NAME: str
    
    # Application Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 5
    
    # Maximum token limits
    MAX_CONTEXT_TOKENS: int = 4000
    MAX_RESPONSE_TOKENS: int = 1000
    
    class Config:
        env_file = ".env"

settings = Settings()
