import os, logging, pymupdf4llm
import textgrad as tg
from groq import Groq
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import time  

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Loading GROQ_API_KEY from .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
assert GROQ_API_KEY, "Please set the GROQ_API_KEY environment variable"

class PipelineConfig:
    """Configuration for the summarization pipeline"""

    model_name: str = "llama-3.2-90b-text-preview"
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

class SummarizationPipeline:
    """End-to-end pipeline for PDF summarization"""

    def __init__(self):
        self.config = PipelineConfig
        self.client = Groq(api_key = self.config.api_key)
        self.pdf_processor = PDFProcessor
    
    def process(self, file_path: str) -> str:
        try:
            llm = tg.get_engine(f"groq-{self.config.model_name}")
            tg.set_backward_engine(tg.get_engine("groq-llama-3.2-11b-text-preview"), override = True)
            model = tg.BlackboxLLM(llm)

            doc = self.pdf_processor.read_pdf(file_path)

            chunk_summaries = []
            for page in doc:
                text = page.text

                system_prompt = tg.Variable(value = f"Here's a financial document. Provide a concise summary highlighting key takeaways. \nText: {text}" , requires_grad = True, role_description="system_prompt")

                evaluation_instr = f"""If nothing is important (like header, footer, introduction, title page, etc.) than just output "No important information found".
                Else, highlight the important information in key points strictly at max 5.
                Do not add any additional information apart from what is written in the text. Text: {text}\n
                """

                answer = model(system_prompt)
                optimizer = tg.TGD(parameters = [answer])

                time.sleep(3)
                loss_fn = tg.TextLoss(evaluation_instr)
                loss = loss_fn(answer)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                chunk_summaries.append(answer)
                
                time.sleep(3)  # Add a 2-second delay so as to avoid "Too Many Requests" error from the API
            
            # Now we find summary of all the chunks together
            text = " ".join([chunk.value for chunk in chunk_summaries])
            system_prompt = tg.Variable(value = f"Here's a financial document. Provide a concise summary highlighting key takeaways.\nText: {text}" , requires_grad = True, role_description="system_prompt")

            evaluation_instr = f"""Provide a concise summary of the document. Be very careful to not exclude the most important information and provide correct statistical data. Keep the summary in specific points and do not add any additional information not given in the text. """

            final_answer = model(system_prompt)
            optimizer = tg.TGD(parameters = [final_answer])

            loss_fn = tg.TextLoss(evaluation_instr)
            loss = loss_fn(final_answer)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            return final_answer.value

        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise


def main():
    pipeline = SummarizationPipeline()
    answer = pipeline.process(input("Enter the path of the PDF file: "))
    logger.info(f"Summary: \n{answer}")

main()