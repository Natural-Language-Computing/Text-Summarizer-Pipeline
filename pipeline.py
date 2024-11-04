import os
from typing import List, Dict, Any
import fitz  # PyMuPDF
import torch
import dspy
from groq import Groq
from dataclasses import dataclass
import yaml
from transformers import AutoTokenizer, TextIteratorStreamer
from threading import Thread
from queue import Queue
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

@dataclass
class PipelineConfig:
    """Configuration for the summarization pipeline"""
    model_name: str = "mixtral-8x7b-32768"
    chunk_size: int = 2048
    overlap: int = 200
    max_length: int = 1024
    temperature: float = 0.7
    api_key: str = os.getenv("GROQ_API_KEY")

class PDFProcessor:
    """Handles PDF reading and text extraction"""
    
    def __init__(self, chunk_size: int = 2048, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def read_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            raise
            
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.overlap
            
        return chunks

class SummarizationModule:
    """Handles text summarization using Groq"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = Groq(api_key=config.api_key)
        
        # Initialize DSPy for prompt tuning
        dspy.settings.configure(lm=self.client)
        
    def generate_prompt(self, text: str, query: str = None) -> str:
        """Generate context-aware prompt"""
        if query:
            return f"""Please provide a focused summary of the following text, specifically addressing: {query}

Text: {text}

Summary:"""
        else:
            return f"""Please provide a comprehensive summary of the following text, highlighting the key points and main ideas:

Text: {text}

Summary:"""
            
    def summarize_chunk(self, chunk: str, query: str = None) -> str:
        """Summarize a single chunk of text"""
        try:
            prompt = self.generate_prompt(chunk, query)
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_length
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in summarization: {e}")
            raise

class SummarizationPipeline:
    """End-to-end pipeline for PDF summarization"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path) if config_path else PipelineConfig()
        self.pdf_processor = PDFProcessor(
            chunk_size=self.config.chunk_size,
            overlap=self.config.overlap
        )
        self.summarizer = SummarizationModule(self.config)
        
    def _load_config(self, config_path: str) -> PipelineConfig:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return PipelineConfig(**config_dict)
        
    def process(self, pdf_path: str, query: str = None) -> Dict[str, Any]:
        """Process PDF and generate summary"""
        try:
            # Extract text from PDF
            logger.info("Extracting text from PDF...")
            text = self.pdf_processor.read_pdf(pdf_path)
            
            # Split into chunks
            chunks = self.pdf_processor.chunk_text(text)
            logger.info(f"Split text into {len(chunks)} chunks")
            
            # Generate summaries for each chunk
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                summary = self.summarizer.summarize_chunk(chunk, query)
                chunk_summaries.append(summary)
                
            # Combine chunk summaries into final summary
            final_prompt = f"""Combine the following summaries into a coherent final summary:

Summaries:
{' '.join(chunk_summaries)}

Final Summary:"""
            
            final_summary = self.summarizer.summarize_chunk(final_prompt)
            
            return {
                "summary": final_summary,
                "chunk_summaries": chunk_summaries,
                "num_chunks": len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise

def main():
    # Example usage
    config = {
        "model_name": "mixtral-8x7b-32768",
        "chunk_size": 2048,
        "overlap": 200,
        "max_length": 1024,
        "temperature": 0.7
    }
    
    with open("config.yaml", "w") as f:
        yaml.dump(config, f)
    
    pipeline = SummarizationPipeline("config.yaml")
    result = pipeline.process(
        input("Enter path: "),
        query="What are the main conclusions and recommendations?"
    )
    
    print(f"Final Summary:\n{result['summary']}")

if __name__ == "__main__":
    main()