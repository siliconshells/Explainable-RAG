# import nltk
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

from openai import OpenAI


from dotenv import load_dotenv

load_dotenv()
import os

from typing import List

import re


def split_sentences(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


client = (
    OpenAI() if os.environ.get("OPENAI_API_KEY") else None
)  # NEW API STYLE – replaces openai.ChatCompletion


def generate_llm_answer(query, chunks):
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


def match_sentences(answer, retrieved):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # sentences = nltk.sent_tokenize(answer)
    sentences = split_sentences(answer)

    doc_emb = model.encode([r["text"] for r in retrieved])
    sent_emb = model.encode(sentences)

    sims = cosine_similarity(sent_emb, doc_emb)

    results = []
    for i, s in enumerate(sentences):
        j = np.argmax(sims[i])
        results.append({"sentence": s, "source_id": j, "similarity": float(sims[i][j])})
    return results


# def token_saliency(answer, retrieved):
#     tokens = answer.split()
#     model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

#     base = model.encode([" ".join(tokens)])
#     doc_emb = model.encode([retrieved[0]["text"]])

#     sal = []
#     for i in range(len(tokens)):
#         reduced = tokens[:i] + tokens[i + 1 :]
#         new = model.encode([" ".join(reduced)])
#         score = float(cosine_similarity(base, new))
#         sal.append((tokens[i], score))

#     return sal


def token_saliency(answer, retrieved):
    """
    Compute token-level saliency via leave-one-out embedding difference.
    Returns: list of (token, score) pairs normalized to [0,1].
    """

    tokens = answer.split()
    if len(tokens) < 2:
        return [(tok, 1.0) for tok in tokens]

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Embed the full answer
    base_emb = model.encode([" ".join(tokens)])[0]

    saliency_scores = []

    # Leave-one-token-out loop
    for i in range(len(tokens)):
        reduced_tokens = tokens[:i] + tokens[i + 1 :]

        if not reduced_tokens:
            saliency_scores.append(1.0)
            continue

        reduced_emb = model.encode([" ".join(reduced_tokens)])[0]

        # Measure change in embedding
        delta = 1 - float(cosine_similarity([base_emb], [reduced_emb])[0][0])
        saliency_scores.append(delta)

    # Normalize 0–1 for better visualization
    max_val = max(saliency_scores)
    if max_val > 0:
        saliency_scores = [s / max_val for s in saliency_scores]
    else:
        saliency_scores = [0.0] * len(tokens)

    return list(zip(tokens, saliency_scores))


def detect_hallucinations(attribution, thr=0.35):
    return [s for s in attribution if s["similarity"] < thr]
