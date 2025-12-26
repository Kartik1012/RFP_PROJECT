import faiss
import numpy as np
import pickle

class FaissVectorStore:
    def __init__(self):
        self.index = None
        self.texts = []
        self.metadatas = []

    def init_index(self, dim: int):
        # Use IndexFlatIP for cosine similarity (vectors normalized)
        self.index = faiss.IndexFlatIP(dim)

    def add(self, emb: list, text: str, metadata: dict):
        vec = np.array(emb).astype("float32")
        faiss.normalize_L2(vec.reshape(1, -1))
        self.index.add(vec.reshape(1, -1))
        self.texts.append(text)
        self.metadatas.append(metadata)

    def save(self, path="faiss_index.pkl"):
        with open(path, "wb") as f:
            pickle.dump((self.index, self.texts, self.metadatas), f)


    def load(self, path="faiss_index.pkl"):
        with open(path, "rb") as f:
            self.index, self.texts, self.metadatas = pickle.load(f)


    def search_all(self, q_emb):
        q_vec = np.array(q_emb).astype("float32")
        faiss.normalize_L2(q_vec.reshape(1, -1))
        total = self.index.ntotal
        if total == 0:
            return [], []
        scores, indices = self.index.search(q_vec.reshape(1, -1), total)
        return scores[0], indices[0]

    def search_topk(self, q_emb, k=5):
        q_vec = np.array(q_emb).astype("float32")
        faiss.normalize_L2(q_vec.reshape(1, -1))
        scores, indices = self.index.search(q_vec.reshape(1, -1), k)
        return scores[0], indices[0]

