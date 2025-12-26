# qa_engine.py
"""RAG-based Q&A engine with citations listed only at the end (file + page)"""

import logging
from typing import List, Dict, Tuple, Union

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


class QAEngine:
    """RAG-based Q&A engine without inline citations"""

    def __init__(self, config):
        self.config = config
        self.llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.llm_temperature
        )
        self.parser = StrOutputParser()
        self.chain = self._build_chain()

    def _build_chain(self):
        """Build LLM chain without inline citation requirement"""

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert assistant answering RFP questions.

Rules:
- Use ONLY the provided context
- Do NOT mention citation numbers like [1], [2] in the answer
- Do NOT invent information outside the context
- If information is missing, say so clearly
"""
            ),
            (
                "user",
                """Context:
{context}

Question:
{question}

Provide a clear, professional answer.
Do NOT include citations inside the answer text.
"""
            )
        ])

        return prompt | self.llm | self.parser

    def answer_single(
        self,
        question: str,
        retriever,
        question_metadata: Dict = None
    ) -> Tuple[str, List[Dict]]:
        """
        Answer a single question using RAG

        Returns:
            Tuple[str, List[Dict]]:
                - answer text (NO inline citations)
                - citation metadata (file name + page only)
        """
        try:
            results = retriever.retrieve(question)

            context_parts = []
            citations = []

            # Build numbered context for grounding (numbers NOT shown to user)
            for i, result in enumerate(results[:10], start=1):
                context_parts.append(
                    f"[{i}] {result.text}"
                )

                citations.append({
                    "id": i,
                    "file_name": result.file_name,
                    "page": result.page
                })

            context = "\n\n".join(context_parts)

            # Safety: limit context length
            max_length = 8000
            if len(context) > max_length:
                context = context[:max_length] + "\n\n[Context truncated]"

            # Generate answer (no inline citations)
            answer_text = self.chain.invoke({
                "question": question,
                "context": context
            })

            logger.info(
                f"Generated answer with {len(citations)} citation sources"
            )

            return answer_text.strip(), citations

        except Exception as e:
            logger.error(f"Failed to answer question: {e}", exc_info=True)

            return (
                f"Error generating answer: {str(e)}",
                []
            )

    def batch_answer(
        self,
        questions: List[Union[Dict, str]],
        retriever
    ) -> List[Dict]:
        """
        Answer multiple questions

        Returns:
            List[Dict] with:
            - question
            - answer (no inline citations)
            - citations (file + page)
            - metadata
        """
        logger.info(f"Answering {len(questions)} questions")

        results = []

        for i, q_item in enumerate(questions, 1):
            # Extract question text and metadata
            if isinstance(q_item, dict):
                question_text = (
                    q_item.get("text")
                    or q_item.get("question")
                    or str(q_item)
                )
                metadata = q_item.get("metadata", {})
            else:
                question_text = str(q_item)
                metadata = {}

            logger.info(
                f"Processing question {i}/{len(questions)}: "
                f"{question_text[:80]}..."
            )

            answer_text, citations = self.answer_single(
                question_text,
                retriever,
                metadata
            )

            results.append({
                "question": question_text,
                "answer": answer_text,
                "citations": citations,
                "metadata": metadata
            })

        logger.info("Batch answering completed")

        return results
