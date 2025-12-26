
# ==============================================================================
# question_extractor.py
# ==============================================================================
"""Question extraction from RFP documents"""
import logging
from typing import List, Dict
import re
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


class QuestionExtractor:
    """Extract questions and requirements from RFP documents"""
    
    def __init__(self, config):
        self.config = config
        self.llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.llm_temperature
        )
    
    def _build_extraction_prompt(self) -> ChatPromptTemplate:
        """Build prompt for extraction"""
        return ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert at analyzing Request for Proposal (RFP) documents.

Your task is to extract QUESTIONS from the document. A question is:
- Any sentence ending with a question mark (?)
- Phrases asking for information like "please provide", "please describe", "please explain"
- Instructions to bidders like "Bidders must submit...", "Vendors should include..."
- Requests for specific information or documentation

Extract ONLY clear, complete questions. Do not infer or rephrase.

Return your response as a JSON array of strings, with ONLY the JSON array and nothing else.
Example: ["Question 1 text here?", "Question 2 text here?", "Please provide your company details"]

If no questions are found, return an empty array: []"""
            ),
            (
                "user",
                """Extract all questions from this RFP text:

{text}

Remember: Return ONLY a JSON array of question strings, nothing else."""
            )
        ])
    
    def _extract_questions_from_text_chunk(self, text: str) -> List[str]:
        """Extract questions from a text chunk using LLM"""
        prompt = self._build_extraction_prompt()
        
        try:
            response = self.llm.invoke(
                prompt.format_messages(text=text)
            )
            
            response_text = response.content.strip()
            
            # Try to parse as JSON array
            try:
                # Remove markdown code blocks if present
                response_text = re.sub(r'```json\s*', '', response_text)
                response_text = re.sub(r'```\s*', '', response_text)
                
                parsed = json.loads(response_text)
                if isinstance(parsed, list):
                    return [str(q).strip() for q in parsed if q]
            except json.JSONDecodeError:
                pass
            
            # Fallback: try to find JSON array in text
            match = re.search(r'\[[\s\S]*?\]', response_text)
            if match:
                try:
                    parsed = json.loads(match.group())
                    if isinstance(parsed, list):
                        return [str(q).strip() for q in parsed if q]
                except json.JSONDecodeError:
                    pass
            
            # Last resort: extract question-like lines
            questions = []
            for line in response_text.splitlines():
                line = line.strip().strip('-•*').strip()
                if line and (
                    line.endswith("?") 
                    or line.lower().startswith(("please provide", "please submit", "bidder"))
                    or "must submit" in line.lower()
                    or "should provide" in line.lower()
                ):
                    questions.append(line)
            
            return questions
            
        except Exception as e:
            logger.error(f"Extraction failed for chunk: {e}")
            return []
    
    def extract_from_text(self, text: str, max_chunk_size: int = 4000) -> List[str]:
        """Extract questions from raw text"""
        # For very long documents, split into chunks
        if len(text) <= max_chunk_size:
            return self._extract_questions_from_text_chunk(text)
        
        # Split into manageable chunks
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1
            
            if current_length >= max_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        # Extract from each chunk
        all_questions = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            questions = self._extract_questions_from_text_chunk(chunk)
            all_questions.extend(questions)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_questions = []
        for q in all_questions:
            q_lower = q.lower()
            if q_lower not in seen:
                seen.add(q_lower)
                unique_questions.append(q)
        
        return unique_questions
    
    def extract_from_file(self, file_path: str) -> List[str]:
        """Extract questions from file"""
        from document_processor import DocumentProcessor
        
        processor = DocumentProcessor(self.config)
        text = processor.get_document_text(file_path)
        
        logger.info(f"Extracting questions from {file_path}")
        logger.info(f"Document length: {len(text)} characters")
        
        questions = self.extract_from_text(text)
        
        logger.info(f"Extracted {len(questions)} questions")
        return questions