def test_retriever(app):
    from app.retriever import RAGRetriever

    r = RAGRetriever("documents/")
    out = r.retrieve("test question")
    assert len(out) > 0
