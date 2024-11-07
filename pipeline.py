import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import dspy
import pymupdf4llm
import yaml
from dotenv import load_dotenv
from groq import Groq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
assert GROQ_API_KEY, "Please set the GROQ_API_KEY environment variable"


@dataclass
class PipelineConfig:
    """Configuration for the summarization pipeline"""

    model_name: str = "llama3-70b-8192"
    temperature: float = 0.7
    api_key: str = GROQ_API_KEY


class PDFProcessor:
    """Handles PDF reading and text extraction"""
    
    def read_pdf(file_path: str) -> list:
        """Extract text from PDF file using pymupdf4llm and convert it to Llama Index Format."""

        try:
            llama_reader = pymupdf4llm.LlamaMarkdownReader()
            doc = llama_reader.load_data(file_path)
            return doc

        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            raise

    def chunk_text(doc: list) -> List[str]:
        """Getting the text from the document of Llama Index format"""

        chunks = [page.text for page in doc]
        return chunks

class SummarizationModule:
    """Handles text summarization using Groq"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = Groq(api_key=config.api_key)
        dspy.settings.configure(lm = self.client)

    def generate_prompt(self, text: str, query: Optional[str] = None) -> str:
        """Generate context-aware prompt"""

        if query:
            return f"{query} Text: {text}"

        else:
            return f"""Please provide a comprehensive summary of the following text, highlighting the key points and main ideas: Text: {text} Summary: """

    def summarize_chunk(self, chunk: str, query: Optional[str] = None) -> str:
        """Summarize a single chunk of text"""

        try:
            prompt = self.generate_prompt(chunk, query)

            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.config.model_name,
                temperature=self.config.temperature,
            )
            output = response.choices[0].message.content
            if not output:
                raise ValueError("No output from the model")
            return output

        except Exception as e:
            logger.error(f"Error in summarization: {e}")
            raise


class SummarizationPipeline:
    """End-to-end pipeline for PDF summarization"""

    def __init__(self, config_path: str):
        self.config = (self._load_config(config_path) if config_path else PipelineConfig())
        self.pdf_processor = PDFProcessor
        self.summarizer = SummarizationModule(self.config)

    def _load_config(self, config_path: str) -> PipelineConfig:
        """Load configuration from YAML file"""

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return PipelineConfig(**config_dict)

    def process(self, pdf_path: str, query: str, expected_keyword: str) -> Dict[str, Any]:
        """Process PDF and generate summary"""

        try:
            logger.info("Extracting text from PDF...")
            doc = self.pdf_processor.read_pdf(pdf_path)
            chunks = self.pdf_processor.chunk_text(doc)
            logger.info(f"Split text into {len(chunks)} chunks")

            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                summary = self.summarizer.summarize_chunk(chunk)
                chunk_summaries.append(summary)

            final_prompt = f"{query}. Expected keywords: {expected_keyword}"
            final_answer = self.summarizer.summarize_chunk(" ".join(chunk_summaries), final_prompt)

            return {
                "answer": final_answer,
                "chunk_summaries": chunk_summaries,
                "num_chunks": len(chunks),
            }

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise


def main():
    # Example usage
    config = {
        "model_name": "llama3-70b-8192",
        "temperature": 0.7,
    }

    with open("config.yaml", "w") as f:
        yaml.dump(config, f)

    pipeline = SummarizationPipeline("config.yaml")
    result = pipeline.process(
        input("Enter PDF file path: "),
        query=input("Enter your query: "),
        expected_keyword=input("Enter expected keywords required: "),
    )

    print(f"\n{result['answer']}")


if __name__ == "__main__":
    main()
