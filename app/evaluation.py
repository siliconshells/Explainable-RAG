from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .cache import cache_get, cache_set


# Lazy-initialized so importing this module doesn't fire any API calls
_llm = None
_embeddings = None


def _components():
    global _llm, _embeddings
    if _llm is None:
        _llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
        _embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    return _llm, _embeddings


# Ragas writes one column per metric — these are the column names produced by the three
# metric classes we use, in the order we want to expose them to the UI
_COLUMN_TO_LABEL = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "llm_context_precision_without_reference": "context_precision",
}


def evaluate_rag(query: str, answer: str, contexts: list[str]) -> dict:
    """
    Run reference-free Ragas metrics on a single (query, answer, contexts) triple.
    Results are cached in Redis for 24h since the same query yields the same scores.
    """
    cache_key = f"ragas:{query}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    llm, embeddings = _components()
    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithoutReference(),
    ]

    ds = Dataset.from_dict(
        {
            "user_input": [query],
            "response": [answer],
            "retrieved_contexts": [contexts],
        }
    )

    result = evaluate(dataset=ds, metrics=metrics, llm=llm, embeddings=embeddings)
    row = result.to_pandas().iloc[0].to_dict()

    scores = {}
    for col, label in _COLUMN_TO_LABEL.items():
        value = row.get(col)
        # Ragas can return NaN if a metric fails on a sample — surface as null to the UI
        scores[label] = float(value) if value is not None and value == value else None

    cache_set(cache_key, scores, ttl=86400)
    return scores
