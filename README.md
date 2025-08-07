# LLM-Powered Intelligent Query-Retrieval System

An advanced system that leverages Large Language Models (LLM) and vector search to provide intelligent answers to questions about PDF documents.

## Features

- PDF document processing and text extraction
- Intelligent text chunking with overlap
- Vector embeddings generation using OpenAI's models
- Efficient semantic search using Pinecone vector database
- Context-aware answer generation using GPT-4
- RESTful API built with FastAPI

## Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key and environment
- Pinecone index (must be created beforehand)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd llm-query-system
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy the example environment file and fill in your credentials:
   ```bash
   cp .env.example .env
   ```

## Configuration

Edit the `.env` file with your credentials and settings:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
OPENAI_COMPLETION_MODEL=gpt-4

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here
PINECONE_INDEX_NAME=your_index_name_here

# Application Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
```

## Running the Application

1. Start the FastAPI server:
   ```bash
   uvicorn llm_query_system.main:app --reload
   ```

2. Access the API documentation at `http://localhost:8000/docs`

## API Usage

### Query Endpoint

POST `/query`

