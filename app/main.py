from flask import Blueprint, render_template, request
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
