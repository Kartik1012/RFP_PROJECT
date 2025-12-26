# config.py - SIMPLE VERSION (Backward Compatible)
"""Simple configuration - works with existing code"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Simple flat configuration (backward compatible)"""
    
    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    
    # Models
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    
    # Chunking (flat structure for backward compatibility)
    chunk_size: int = 1500
    chunk_overlap: int = 300
    separators: list = None
    
    # Retrieval
    similarity_threshold: float = 0.95
    top_k_chunks: int = 10
    fallback_top_k: int = 5
    rerank_results: bool = True
    diversity_penalty: float = 0.3
    max_context_length: int = 8000
    
    # Extraction
    extract_requirements: bool = True
    extract_metadata: bool = True
    deduplicate_similarity_threshold: float = 0.85
    min_question_length: int = 10
    max_chunk_size_for_extraction: int = 6000
    
    # Prompts
    system_prompt_style: str = "professional"
    include_confidence_scores: bool = True
    request_citations: bool = True
    answer_format: str = "comprehensive"
    include_source_quotes: bool = True
    
    # Paths
    data_folder: str = "input"
    internal_docs_folder: str = "internal"
    index_path: str = "faiss_index.pkl"
    output_folder: str = "output"
    
    # PDF Processing
    use_ocr: bool = False
    extract_tables: bool = True
    preserve_layout: bool = True
    
    # PDF Generation
    page_margin: float = 1.0
    font_size: int = 11
    line_spacing: int = 15
    include_citations: bool = True
    
    # Logging
    log_level: str = "INFO"
    verbose: bool = True
    
    def __post_init__(self):
        """Initialize defaults and validate"""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        # Set default separators if not provided
        if self.separators is None:
            self.separators = [
                "\n\n\n", "\n\n", "\n", ". ", "! ", "? ", 
                "; ", ", ", " ", ""
            ]
        
        # Create output folder
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Create nested config objects for new code compatibility
        self._create_nested_configs()
    
    def _create_nested_configs(self):
        """Create nested config objects for compatibility with enhanced modules"""
        from types import SimpleNamespace
        
        # Chunking config
        self.chunking = SimpleNamespace(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            respect_sentence_boundary=True,
            min_chunk_size=500
        )
        
        # Retrieval config
        self.retrieval = SimpleNamespace(
            similarity_threshold=self.similarity_threshold,
            top_k_chunks=self.top_k_chunks,
            rerank_results=self.rerank_results,
            diversity_penalty=self.diversity_penalty,
            max_context_length=self.max_context_length
        )
        
        # Extraction config
        self.extraction = SimpleNamespace(
            extract_requirements=self.extract_requirements,
            extract_metadata=self.extract_metadata,
            deduplicate_similarity_threshold=self.deduplicate_similarity_threshold,
            min_question_length=self.min_question_length,
            max_chunk_size_for_extraction=self.max_chunk_size_for_extraction
        )
        
        # Prompts config
        self.prompts = SimpleNamespace(
            system_prompt_style=self.system_prompt_style,
            include_confidence_scores=self.include_confidence_scores,
            request_citations=self.request_citations,
            answer_format=self.answer_format,
            include_source_quotes=self.include_source_quotes
        )
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create config from environment variables"""
        config = cls()
        
        # Allow environment overrides
        if model := os.getenv("LLM_MODEL"):
            config.llm_model = model
        if temp := os.getenv("LLM_TEMPERATURE"):
            config.llm_temperature = float(temp)
        
        return config
    
    def use_fast_mode(self):
        """Quick mode for testing"""
        self.llm_model = "gpt-4o-mini"
        self.embedding_model = "text-embedding-3-small"
        self.top_k_chunks = 5
        self.chunk_size = 1000
        self._create_nested_configs()  # Update nested configs
    
    def use_quality_mode(self):
        """High-quality mode for production"""
        self.llm_model = "gpt-4o-mini"
        self.embedding_model = "text-embedding-3-small"
        self.top_k_chunks = 15
        self.chunk_size = 2000
        self.rerank_results = True
        self._create_nested_configs()  # Update nested configs


# Alternative: If you want the full enhanced version
@dataclass
class EnhancedConfig:
    """Full enhanced configuration with nested structure"""
    
    # Import from the enhanced_config artifact if you want all features
    pass