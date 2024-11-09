import os
import logging
import time
import random
from typing import List
from dotenv import load_dotenv
import pymupdf4llm
import textgrad as tg
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
assert GROQ_API_KEY, "Please set the GROQ_API_KEY environment variable"


class PipelineConfig:
    """Configuration for the summarization pipeline"""

    model_name: str = "llama-3.1-8b-instant"
    temperature: float = 0.7
    api_key: str = GROQ_API_KEY


class PDFProcessor:
    """Handles PDF reading and text extraction"""

    @staticmethod
    def read_pdf(file_path: str) -> List[str]:
        """Extract text from PDF file using pymupdf4llm."""
        try:
            llama_reader = pymupdf4llm.LlamaMarkdownReader()
            doc = llama_reader.load_data(file_path)
            return [page.text for page in doc]
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            raise


class SummarizationPipeline:
    """End-to-end pipeline for PDF summarization"""

    def __init__(self):
        self.config = PipelineConfig()
        self.model = self.initialize_model()

    def initialize_model(self):
        llm = tg.get_engine(f"groq-{self.config.model_name}")
        tg.set_backward_engine(
            tg.get_engine(f"groq-{self.config.model_name}"), override=True
        )
        return tg.BlackboxLLM(llm)

    def retry_with_backoff(self, func, *args, **kwargs):
        backoff_time = 5
        max_backoff_time = 60
        while True:
            try:
                return func(*args, **kwargs)
            except Exception:
                logger.warning(f"Rate limit hit, retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time = min(
                    max_backoff_time, backoff_time * 2 + random.uniform(0, 1)
                )

    def process_batch(self, batch_text: str) -> tg.Variable:
        system_prompt = tg.Variable(
            value=f"Here's a financial document. Provide a concise summary highlighting key takeaways. \nText: {batch_text}",
            requires_grad=True,
            role_description="system_prompt",
        )
        evaluation_instr = (
            "If nothing is important (like header, footer, introduction, title page, etc.) "
            "then just output 'No important information found'. Else, highlight the important "
            "information in key points strictly at max 5. Do not add any additional information "
            "apart from what is written in the text."
        )
        answer = self.retry_with_backoff(self.model, system_prompt)
        self.optimize_answer(answer, evaluation_instr)
        return answer

    def optimize_answer(self, answer: tg.Variable, evaluation_instr: str):
        optimizer = tg.TGD(parameters=[answer])
        loss_fn = tg.TextLoss(evaluation_instr)
        loss = loss_fn(answer)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def process(self, file_path: str) -> str:
        try:
            with st.spinner("Reading and processing PDF..."):
                pages = PDFProcessor.read_pdf(file_path)
                # Combine pages into batches
                batch_size = 7
                batches = [
                    " ".join(pages[i : i + batch_size])
                    for i in range(0, len(pages), batch_size)
                ]

                progress_bar = st.progress(0)
                batch_summaries = []
                for i, batch in enumerate(batches):
                    batch_summaries.append(self.process_batch(batch))
                    progress_bar.progress((i + 1) / len(batches))

                combined_text = " ".join([batch.value for batch in batch_summaries])
                final_summary = self.summarize_document(combined_text)
                return final_summary.value
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            logger.error(f"Error processing PDF: {e}")
            raise

    def summarize_document(self, text: str) -> tg.Variable:
        system_prompt = tg.Variable(
            value=f"Here's a financial document. Provide a concise summary highlighting key takeaways.\nText: {text}",
            requires_grad=True,
            role_description="system_prompt",
        )
        evaluation_instr = (
            "Provide a concise summary of the document. Be very careful to not exclude the most "
            "important information and provide correct statistical data. Keep the summary in specific "
            "points and do not add any additional information not given in the text."
        )
        final_answer = self.retry_with_backoff(self.model, system_prompt)
        self.optimize_answer(final_answer, evaluation_instr)
        return final_answer


def main():
    st.title("PDF Summarization Tool")
    st.write("Upload a PDF file to generate a summary of its contents.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with st.spinner("Saving uploaded file..."):
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

        try:
            pipeline = SummarizationPipeline()
            summary = pipeline.process(temp_path)

            st.subheader("Summary")
            st.write(summary)

        finally:
            # Cleanup temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    main()
