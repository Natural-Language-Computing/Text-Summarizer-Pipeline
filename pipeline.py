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
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

@dataclass
class PipelineConfig:
    """Configuration for the summarization pipeline"""
    model_name: str = "llama3-70b-8192"
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
        
        dspy.settings.configure(lm=self.client)
        
    def generate_prompt(self, text: str, query: str = None) -> str:
        """Generate context-aware prompt"""
        if query:
            return f""" {query} Text: {text} """
        else:
            return f"""Please provide a comprehensive summary of the following text, highlighting the key points and main ideas: Text: {text} Summary: """
    
    def evaluate_summary(self, generated_summary: str, expected_criteria: str) -> bool:
        """Evaluate the generated summary against some criteria (e.g., completeness, relevance)"""
        if expected_criteria in generated_summary:
            return True
        return False

    def tune_prompt(self, prompt: str, feedback: str, generated_summary: str, expected_criteria: str) -> str:
        """Tuning the prompt based on feedback and evaluation of the generated summary"""
        summary_is_good = self.evaluate_summary(generated_summary, expected_criteria)
        
        if not summary_is_good:
            # If summary is not good, adjust prompt for better output
            logger.info("Refining prompt based on summary feedback")
            prompt = f"{prompt}\n{feedback}\nPlease ensure the summary includes the following details: {expected_criteria}"
        else:
            logger.info("Summary meets the expected criteria")
        
        return prompt

    def summarize_chunk(self, chunk: str, query: str = None, feedback: str = None, expected_criteria: str = None) -> str:
        """Summarize a single chunk of text, incorporating feedback and evaluating summary quality"""
        try:
            prompt = self.generate_prompt(chunk, query)
            if feedback and expected_criteria:
                prompt = self.tune_prompt(prompt, feedback, "", expected_criteria)
            
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
        
    def process(self, pdf_path: str, query: str = None, feedback: str = None, expected_criteria: str = None) -> Dict[str, Any]:
        """Process PDF and generate summary"""
        
        try:
            logger.info("Extracting text from PDF...")
            text = self.pdf_processor.read_pdf(pdf_path)
            
            chunks = self.pdf_processor.chunk_text(text)
            logger.info(f"Split text into {len(chunks)} chunks")
            
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                summary = self.summarizer.summarize_chunk(chunk, query, feedback, expected_criteria)
                chunk_summaries.append(summary)
                
            final_answer = self.summarizer.summarize_chunk(" ".join(chunk_summaries), query, feedback, expected_criteria)
            
            return {
                "answer": final_answer,
                "chunk_summaries": chunk_summaries,
                "num_chunks": len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise

def main():
    # Example usage
    config = {
        "model_name": "llama3-70b-8192",
        "chunk_size": 2048,
        "overlap": 200,
        "max_length": 1024,
        "temperature": 0.7
    }
    
    with open("config.yaml", "w") as f:
        yaml.dump(config, f)
    
    pipeline = SummarizationPipeline("config.yaml")
    result = pipeline.process(
        input("Enter PDF file path: "),
        query=input("Enter query: "),
        feedback=input("Enter feedback for prompt tuning (optional): "),
        expected_criteria=input("Enter expected criteria for summarization (e.g., specific keywords): ")
    )
    
    print(f"\n{result['answer']}")

if __name__ == "__main__":
    main()
