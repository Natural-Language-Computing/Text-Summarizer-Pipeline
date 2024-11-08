import os, logging, pymupdf4llm
import textgrad as tg
from groq import Groq
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

#Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Loading GROQ_API_KEY from .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
assert GROQ_API_KEY, "Please set the GROQ_API_KEY environment variable"

class PipelineConfig:
    """Configuration for the summarization pipeline"""

    model_name: str = "mixtral-8x7b-32768"
    temperature: float = 0.7
    api_key: str = GROQ_API_KEY

class PDFProcessor:
    """Handles PDF reading and text extraction"""
    
    def read_pdf(file_path: str) -> list:
        """Extract text from PDF file using pymupdf4llm and convert it to Llama Index Format."""

        try:
            llama_reader = pymupdf4llm.LlamaMarkdownReader()
            doc = llama_reader.load_data(file_path)
            # chunks = [page.text for page in doc]
            return doc

        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            raise

class SummarizationPipeline:
    """End-to-end pipeline for PDF summarization"""

    def __init__(self):
        self.config = PipelineConfig
        self.client = Groq(api_key = self.config.api_key)
        self.pdf_processor = PDFProcessor
    
    def process(self):
        try:
            llm = tg.get_engine(f"groq-{self.config.model_name}")
            tg.set_backward_engine(tg.get_engine("groq-llama3-8b-8192"), override = True)
            model = tg.BlackboxLLM(llm)

            doc = self.pdf_processor.read_pdf('sample.pdf')

            for page in doc:
                text = page.text
                # print(text)
                system_prompt = tg.Variable(value = f"Please provide a concise(100 words) summary of the following text, highlighting the key points and main ideas:\n Text: {text}" , requires_grad = True, role_description="system_prompt")

                evaluation_instr = f"""Do not add additional information which is not provided in the text.
                If nothing is important than output "No important information found".
                """

                answer = model(system_prompt)
                optimizer = tg.TGD(parameters = [answer])

                print("Initial answer:\n", answer.value)

                loss_fn = tg.TextLoss(evaluation_instr)
                loss = loss_fn(answer)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                print("Final answer:\n", answer.value)
                print()
        
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise


def main():
    pipeline = SummarizationPipeline()
    pipeline.process()

main()