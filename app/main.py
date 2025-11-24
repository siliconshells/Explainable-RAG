from flask import Blueprint, render_template, request
from .retriever import RAGRetriever
from .attribution import (
    generate_llm_answer,
    match_sentences,
    token_saliency,
    detect_hallucinations,
)

# ✔️ Define Blueprint FIRST
main_bp = Blueprint("main", __name__)

retriever = RAGRetriever("documents/")


@main_bp.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form["query"]

        # Retrieve chunks
        retrieved = retriever.retrieve(query)

        # Generate answer
        answer = generate_llm_answer(query, retrieved)

        # Sentence Attribution
        attribution = match_sentences(answer, retrieved)

        # Token Saliency
        saliency = token_saliency(answer, retrieved)

        # Hallucination Detection
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

    # GET request → show empty page
    return render_template("index.html")
