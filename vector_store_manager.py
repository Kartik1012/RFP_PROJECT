# vector_store_manager.py
"""
Unified FAISS Vector Store + Retriever + Metadata handling
- Stores text + metadata (file_name, page)
- Returns chunks with metadata
- Supports clean citation generation
"""

import logging
import pickle
from typing import List, Tuple

import faiss
import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)

# ======================================================================
# FAISS VECTOR STORE (WITH METADATA)
# ======================================================================

class FaissVectorStore:
    def __init__(self):
        self.index = None
        self.texts: List[str] = []
        self.metadatas: List[dict] = []

    def init_index(self, dim: int):
        # Cosine similarity (normalized vectors)
        self.index = faiss.IndexFlatIP(dim)

    def add(self, embedding: list, text: str, metadata: dict):
        vec = np.array(embedding, dtype="float32")
        faiss.normalize_L2(vec.reshape(1, -1))
        self.index.add(vec.reshape(1, -1))
        self.texts.append(text)
        self.metadatas.append(metadata)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(
                (self.index, self.texts, self.metadatas),
                f
            )

    def load(self, path: str):
        with open(path, "rb") as f:
            self.index, self.texts, self.metadatas = pickle.load(f)

    def search_all(self, query_embedding):
        q_vec = np.array(query_embedding, dtype="float32")
        faiss.normalize_L2(q_vec.reshape(1, -1))

        total = self.index.ntotal
        if total == 0:
            return [], []

        scores, indices = self.index.search(q_vec.reshape(1, -1), total)
        return scores[0], indices[0]

    def search_topk(self, query_embedding, k: int):
        q_vec = np.array(query_embedding, dtype="float32")
        faiss.normalize_L2(q_vec.reshape(1, -1))
        scores, indices = self.index.search(q_vec.reshape(1, -1), k)
        return scores[0], indices[0]


# ======================================================================
# VECTOR STORE MANAGER
# ======================================================================

class VectorStoreManager:
    """Builds index and provides retriever"""

    def __init__(self, config):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.store = FaissVectorStore()

    # ---------------- Embeddings ----------------

    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.config.embedding_model,
            input=text
        )
        return response.data[0].embedding

    # ---------------- Indexing ----------------

    def build_index(self, documents: List):
        from document_processor import DocumentProcessor

        processor = DocumentProcessor(self.config)
        chunks = processor.chunk_documents(documents)

        if not chunks:
            raise ValueError("No chunks available for indexing")

        dim = len(self.get_embedding(chunks[0].page_content[:200]))
        self.store.init_index(dim)

        for i, chunk in enumerate(chunks, start=1):
            metadata = {
                "file_name": chunk.metadata.get("file_name", ""),
                "page": chunk.metadata.get("page", 0)
            }

            emb = self.get_embedding(chunk.page_content)
            self.store.add(emb, chunk.page_content, metadata)

            if i % 100 == 0:
                logger.info(f"Indexed {i}/{len(chunks)} chunks")

        self.store.save(self.config.index_path)
        logger.info(f"FAISS index saved at {self.config.index_path}")

    # ---------------- Retriever ----------------

    def get_retriever(self):
        self.store.load(self.config.index_path)
        return Retriever(self.store, self)


# ======================================================================
# RETRIEVER + CHUNK OBJECT
# ======================================================================

class ChunkWithMetadata:
    """Returned chunk with metadata"""

    def __init__(self, text: str, score: float, metadata: dict):
        self.text = text
        self.score = score
        self.metadata = metadata or {}
        self.file_name = self.metadata.get("file_name", "")
        self.page = self.metadata.get("page", 0)

    def get_citation(self) -> str:
        if self.file_name and self.page:
            return f"{self.file_name}, Page {self.page}"
        if self.file_name:
            return self.file_name
        return "Source unknown"


class Retriever:
    """FAISS-based retriever returning metadata-aware chunks"""

    def __init__(self, store: FaissVectorStore, manager: VectorStoreManager):
        self.store = store
        self.manager = manager

    def retrieve(
        self,
        query: str,
        threshold: float = None,
        fallback_k: int = None
    ) -> List[ChunkWithMetadata]:

        threshold = threshold or self.manager.config.similarity_threshold
        fallback_k = fallback_k or self.manager.config.fallback_top_k

        q_emb = self.manager.get_embedding(query)
        scores, indices = self.store.search_all(q_emb)

        results = [
            ChunkWithMetadata(
                self.store.texts[idx],
                float(score),
                self.store.metadatas[idx]
            )
            for score, idx in zip(scores, indices)
            if score >= threshold
        ]

        if not results:
            scores, indices = self.store.search_topk(q_emb, fallback_k)
            results = [
                ChunkWithMetadata(
                    self.store.texts[idx],
                    float(score),
                    self.store.metadatas[idx]
                )
                for score, idx in zip(scores, indices)
            ]

        results.sort(key=lambda x: x.score, reverse=True)
        return results
