"""Enhanced main pipeline with quality checks"""
import argparse
import logging
from pathlib import Path

from config import Config
from document_processor import DocumentProcessor
from question_extractor import QuestionExtractor
from vector_store_manager import VectorStoreManager
from qa_engine import QAEngine
from pdf_generator import PDFGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RFPPipeline:
    """Enhanced pipeline with quality assurance"""

    def __init__(self, config: Config):
        self.config = config
        self.doc_processor = DocumentProcessor(config)
        self.question_extractor = QuestionExtractor(config)
        self.vector_manager = VectorStoreManager(config)
        self.qa_engine = QAEngine(config)
        self.pdf_generator = PDFGenerator(config)

    def build_knowledge_base(self, folder_path: str = None):
        """Build vector store from documents"""
        folder = folder_path or self.config.internal_docs_folder
        logger.info(f"Building knowledge base from: {folder}")

        try:
            documents = self.doc_processor.load_documents(folder)
            if not documents:
                raise ValueError(f"No documents found in {folder}")

            logger.info(f"Loaded {len(documents)} document pages")

            self.vector_manager.build_index(documents)
            logger.info("✓ Knowledge base built successfully")

        except Exception as e:
            logger.error(f"Failed to build knowledge base: {e}")
            raise

    # def extract_questions(self, rfp_path: str) -> list:
    #     """Extract questions with metadata"""
    #     logger.info(f"Extracting questions from: {rfp_path}")
    #
    #     try:
    #         questions = self.question_extractor.extract_from_file(rfp_path)
    #         logger.info(f"✓ Extracted {len(questions)} questions")
    #
    #         # Export extraction report
    #         if questions:
    #             report_path = f"{self.config.output_folder}/extraction_report.pdf"
    #             self.pdf_generator.generate_extraction_report(
    #                 [q.dict() for q in questions],
    #                 report_path
    #             )
    #             logger.info(f"✓ Extraction report saved to {report_path}")
    #
    #         return questions
    #     except Exception as e:
    #         logger.error(f"Failed to extract questions: {e}")
    #         raise
    def extract_questions(self, rfp_path: str) -> list[str]:
        """Extract questions from RFP document"""
        logger.info(f"Extracting questions from: {rfp_path}")

        try:
            questions = self.question_extractor.extract_from_file(rfp_path)
            logger.info(f"Extracted {len(questions)} questions")
            return questions
        except Exception as e:
            logger.error(f"Failed to extract questions: {e}")
            raise

    def answer_questions(
        self,
        questions: list,
        output_path: str = None
    ) -> list[dict]:
        """Answer questions with quality checks"""
        logger.info(f"Answering {len(questions)} questions")

        try:
            retriever = self.vector_manager.get_retriever()

            # Convert ExtractedQuestion objects to dicts
            q_items = [
                {
                    'text': q.question_text if hasattr(q, 'question_text') else str(q),
                    'metadata': {
                        'page': getattr(q, 'page_number', None),
                        'section': getattr(q, 'section', ''),
                        'type': getattr(q, 'question_type', 'question'),
                        'importance': getattr(q, 'importance', 'medium')
                    } if hasattr(q, 'page_number') else {}
                }
                for q in questions
            ]

            results = self.qa_engine.batch_answer(q_items, retriever)

            if output_path:
                self.pdf_generator.generate_qa_pdf(results, output_path)
                logger.info(f"✓ Answers saved to: {output_path}")

            return results

        except Exception as e:
            logger.error(f"Failed to answer questions: {e}")
            raise

    def run_full_pipeline(
        self,
        rfp_path: str,
        rebuild_index: bool = False,
        output_path: str = None
    ):
        """Run complete enhanced pipeline"""
        logger.info("=" * 60)
        logger.info("Starting Enhanced RFP Q&A Pipeline")
        logger.info("=" * 60)

        # Build knowledge base if needed
        if rebuild_index or not Path(self.config.index_path).exists():
            self.build_knowledge_base()

        # Extract questions
        questions = self.extract_questions(rfp_path)

        if not questions:
            logger.warning("No questions extracted")
            return []

        # Answer questions
        output = output_path or f"{self.config.output_folder}/rfp_response.pdf"
        results = self.answer_questions(questions, output)

        logger.info("=" * 60)
        logger.info("✓ Pipeline completed successfully")
        logger.info(f"  - Processed {len(questions)} questions")
        logger.info(f"  - Output: {output}")
        logger.info("=" * 60)

        return results


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Enhanced RFP Question Extraction and Answering Pipeline"
    )
    parser.add_argument("rfp_path", help="Path to RFP document")
    parser.add_argument("--rebuild-index", action="store_true",
                       help="Rebuild vector store")
    parser.add_argument("--output", help="Output PDF path")
    parser.add_argument("--data-folder", help="Internal documents folder")
    parser.add_argument("--fast", action="store_true",
                       help="Use fast mode (cheaper, less accurate)")
    parser.add_argument("--quality", action="store_true",
                       help="Use quality mode (expensive, most accurate)")

    args = parser.parse_args()

    # Load config
    config = Config.from_env()

    if args.data_folder:
        config.internal_docs_folder = args.data_folder

    if args.fast:
        config.use_fast_mode()
        logger.info("Using FAST mode")
    elif args.quality:
        config.use_quality_mode()
        logger.info("Using QUALITY mode")

    # Run pipeline
    pipeline = RFPPipeline(config)
    pipeline.run_full_pipeline(
        rfp_path=args.rfp_path,
        rebuild_index=args.rebuild_index,
        output_path=args.output
    )

if __name__ == "__main__":
    main()