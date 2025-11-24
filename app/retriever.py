import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from .cache import cache_get, cache_set


class RAGRetriever:
    def __init__(self, doc_path, chunk_size=400, overlap=80):
        self.doc_path = doc_path
        self.chunk_size = chunk_size
        self.overlap = overlap

        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Load and chunk docs
        self.documents, self.meta = self._load_and_chunk_documents()

        # Embed all chunks
        self.embeddings = self.model.encode(self.documents)

        # Build FAISS index
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    # For loading and chunking documents - intended for use within internal class
    def _load_and_chunk_documents(self):
        docs = []
        meta = []

        for fname in os.listdir(self.doc_path):
            if not fname.endswith(".txt") and not fname.endswith(".md"):
                continue

            text = open(os.path.join(self.doc_path, fname)).read()
            chunks = self._chunk_text(text)

            for i, chunk in enumerate(chunks):
                docs.append(chunk)
                meta.append({"filename": fname, "chunk_id": i})

        return docs, meta

    # Helper function for chunking text
    def _chunk_text(self, text):
        words = text.split()
        chunks = []

        i = 0
        while i < len(words):
            chunk = words[i : i + self.chunk_size]
            chunks.append(" ".join(chunk))
            i += self.chunk_size - self.overlap  # overlapping chunks

        return chunks

    # This searches the FAISS dataset and retrieve the top 3 chunks for context
    # It checks the cache first
    def retrieve(self, query, top_k=3):
        cached = cache_get(query)
        if cached:
            return cached

        q_embed = self.model.encode([query])
        dist, idx = self.index.search(q_embed, min(top_k, len(self.documents)))

        results = []
        for j, i in enumerate(idx[0]):
            results.append(
                {
                    "text": self.documents[i],
                    "distance": float(dist[0][j]),
                    "meta": self.meta[i],
                }
            )

        cache_set(query, results)
        return results
