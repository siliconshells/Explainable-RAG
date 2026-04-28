import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.attribution import _split_sentences, match_sentences, token_saliency, detect_hallucinations


def test_split_sentences_basic():
    result = _split_sentences("Hello world. This is a test.")
    assert result == ["Hello world.", "This is a test."]


def test_split_sentences_single():
    result = _split_sentences("Just one sentence")
    assert result == ["Just one sentence"]


def test_token_saliency_returns_all_tokens():
    answer = "The model retrieves relevant chunks."
    result = token_saliency(answer)
    tokens = answer.split()
    assert len(result) == len(tokens)
    assert all(isinstance(score, float) for _, score in result)


def test_token_saliency_scores_normalized():
    result = token_saliency("GradCAM highlights important image regions for classification.")
    scores = [score for _, score in result]
    assert max(scores) <= 1.0
    assert min(scores) >= 0.0


def test_token_saliency_single_token():
    result = token_saliency("word")
    assert result == [("word", 1.0)]


def test_detect_hallucinations_filters_low_similarity():
    attribution = [
        {"sentence": "Grounded sentence.", "source_id": 0, "similarity": 0.8},
        {"sentence": "Hallucinated sentence.", "source_id": 1, "similarity": 0.2},
    ]
    result = detect_hallucinations(attribution, threshold=0.35)
    assert len(result) == 1
    assert result[0]["sentence"] == "Hallucinated sentence."


def test_detect_hallucinations_custom_threshold():
    attribution = [
        {"sentence": "A.", "source_id": 0, "similarity": 0.4},
        {"sentence": "B.", "source_id": 0, "similarity": 0.6},
    ]
    assert len(detect_hallucinations(attribution, threshold=0.5)) == 1
    assert len(detect_hallucinations(attribution, threshold=0.3)) == 0
