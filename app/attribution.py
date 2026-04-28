from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
import re

load_dotenv()

# Loaded once at module level — both match_sentences and token_saliency share this model
_embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

client = OpenAI() if os.environ.get("OPENAI_API_KEY") else None


def _split_sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", text.strip())


def generate_llm_answer(query: str, chunks: list[dict]) -> str:
    context = "\n\n".join([c["text"] for c in chunks])

    prompt = (
        "Use ONLY the context below to answer the question.\n"
        "If something is not supported by the context, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


def match_sentences(answer: str, retrieved: list[dict]) -> list[dict]:
    sentences = _split_sentences(answer)

    doc_emb = _embed_model.encode([r["text"] for r in retrieved])
    sent_emb = _embed_model.encode(sentences)

    sims = cosine_similarity(sent_emb, doc_emb)

    results = []
    for i, sentence in enumerate(sentences):
        j = int(np.argmax(sims[i]))
        results.append({"sentence": sentence, "source_id": j, "similarity": float(sims[i][j])})
    return results


def token_saliency(answer: str) -> list[tuple[str, float]]:
    """
    Compute token-level saliency via leave-one-out embedding difference.
    Returns a list of (token, score) pairs normalized to [0, 1].
    """
    tokens = answer.split()
    if len(tokens) < 2:
        return [(tok, 1.0) for tok in tokens]

    base_emb = _embed_model.encode([" ".join(tokens)])[0]

    saliency_scores = []
    for i in range(len(tokens)):
        reduced_tokens = tokens[:i] + tokens[i + 1:]

        if not reduced_tokens:
            saliency_scores.append(1.0)
            continue

        reduced_emb = _embed_model.encode([" ".join(reduced_tokens)])[0]

        # How much the embedding shifts when this token is removed — higher means more important
        delta = 1.0 - float(cosine_similarity([base_emb], [reduced_emb])[0][0])
        saliency_scores.append(delta)

    # Normalize to [0, 1] for consistent visualization
    max_val = max(saliency_scores)
    if max_val > 0:
        saliency_scores = [s / max_val for s in saliency_scores]
    else:
        saliency_scores = [0.0] * len(tokens)

    return list(zip(tokens, saliency_scores))


def detect_hallucinations(attribution: list[dict], threshold: float = 0.35) -> list[dict]:
    """Return sentences whose similarity to any retrieved chunk falls below the threshold."""
    return [s for s in attribution if s["similarity"] < threshold]
