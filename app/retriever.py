import os
import faiss
from sentence_transformers import SentenceTransformer
from .cache import cache_get, cache_set


class RAGRetriever:
    def __init__(self, doc_path: str, chunk_size: int = 400, overlap: int = 80):
        self.doc_path = doc_path
        self.chunk_size = chunk_size
        self.overlap = overlap

        # Load from the local path where the Dockerfile bakes in the model
        self.model = SentenceTransformer("./models")

        # Load and chunk docs
        self.documents, self.meta = self._load_and_chunk_documents()

        # Embed all chunks and build FAISS index
        self.embeddings = self.model.encode(self.documents)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def _load_and_chunk_documents(self) -> tuple[list[str], list[dict]]:
        docs = []
        meta = []

        for fname in os.listdir(self.doc_path):
            if not fname.endswith((".txt", ".md")):
                continue

            with open(os.path.join(self.doc_path, fname), encoding="utf-8") as f:
                text = f.read()

            for i, chunk in enumerate(self._chunk_text(text)):
                docs.append(chunk)
                meta.append({"filename": fname, "chunk_id": i})

        return docs, meta

    def _chunk_text(self, text: str) -> list[str]:
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunks.append(" ".join(words[i: i + self.chunk_size]))
            i += self.chunk_size - self.overlap  # step back by overlap to create continuity between chunks
        return chunks

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        cached = cache_get(query)
        if cached:
            return cached

        q_embed = self.model.encode([query])
        dist, idx = self.index.search(q_embed, min(top_k, len(self.documents)))

        results = [
            {
                "text": self.documents[i],
                "distance": float(dist[0][j]),
                "meta": self.meta[i],
            }
            for j, i in enumerate(idx[0])
        ]

        cache_set(query, results)
        return results
