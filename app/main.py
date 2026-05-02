from flask import Blueprint, render_template, request, jsonify
from .retriever import RAGRetriever
from .attribution import (
    generate_llm_answer,
    match_sentences,
    token_saliency,
    detect_hallucinations,
)


main_bp = Blueprint("main", __name__)

retriever = RAGRetriever("documents/")


@main_bp.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form["query"]

        retrieved = retriever.retrieve(query)
        answer = generate_llm_answer(query, retrieved)
        attribution = match_sentences(answer, retrieved)
        saliency = token_saliency(answer)
        hallucinations = detect_hallucinations(attribution)

        return render_template(
            "index.html",
            query=query,
            answer=answer,
            retrieved=retrieved,
            attribution=attribution,
            saliency=saliency,
            hallucinations=hallucinations,
        )

    return render_template("index.html")


@main_bp.route("/evaluate", methods=["POST"])
def evaluate_route():
    """Run Ragas metrics asynchronously after the main answer has rendered."""
    # Imported inside the route so a missing/broken Ragas install never breaks the home page
    from .evaluation import evaluate_rag

    data = request.get_json(silent=True) or {}
    query = data.get("query")
    answer = data.get("answer")
    contexts = data.get("contexts")

    if not query or not answer or not contexts:
        return jsonify({"ok": False, "error": "Missing query, answer, or contexts."}), 400

    try:
        scores = evaluate_rag(query, answer, contexts)
        return jsonify({"ok": True, "scores": scores})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
