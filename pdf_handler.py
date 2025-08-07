"""
PDF download and text extraction functionality
"""

import io
import logging
from typing import List
import httpx
import PyPDF2
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class PDFHandler:
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60)
    )
    async def download_pdf(self, url: str) -> bytes:
        """
        Download PDF from URL with retry logic
        """
        try:
            timeout = httpx.Timeout(30.0, connect=10.0)
            async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
                logger.info(f"Downloading PDF from {url}")
                response = await client.get(url)
                response.raise_for_status()
                logger.info("PDF downloaded successfully")
                return response.content
        except httpx.HTTPError as e:
            logger.error(f"HTTP error downloading PDF from {url}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error downloading PDF from {url}: {str(e)}")
            raise

    def extract_text(self, pdf_content: bytes) -> str:
        """
        Extract text content from PDF bytes
        """
        try:
            pdf_file = io.BytesIO(pdf_content)
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise

    def chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Split text into overlapping chunks
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            # If this is the last chunk, end at text_length
            if end >= text_length:
                chunks.append(text[start:text_length])
                break
            
            # Find the last period in the chunk to avoid cutting sentences
            last_period = text.rfind('.', start, end)
            if last_period != -1 and last_period > start:
                end = last_period + 1
            
            chunks.append(text[start:end])
            start = end - overlap  # Move start back by overlap amount

        return chunks
