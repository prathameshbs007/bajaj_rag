"""
Data models and schemas for the LLM Query System
"""

from typing import List, Optional
from pydantic import BaseModel, HttpUrl

class Question(BaseModel):
    """Model for a single question"""
    text: str
    
class QueryRequest(BaseModel):
    """Model for the incoming query request"""
    document_url: HttpUrl
    questions: List[Question]

class Answer(BaseModel):
    """Model for a single answer"""
    question: str
    answer: str
    context_chunks: List[str]
    confidence_score: float

class QueryResponse(BaseModel):
    """Model for the query response"""
    document_url: HttpUrl
    answers: List[Answer]
    processing_time: float
