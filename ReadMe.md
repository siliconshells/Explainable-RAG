## Explainable RAG: Understanding how LLMs use Context

### Motivation and why it matters
LLMs often cite retrieved documents but give no clue how each retrieved document chunk affects their responses, so users can't verify faithfulness. Using a RAG system with sources we control, this project asks the LLM to use only the retrieved context for its response, allowing us to evaluate how the LLMs use the context they get to provide a response.   

It builds trust and explainability into RAG systems, enabling users to see and verify how answers are grounded in evidence. This also exposes hallucination when it occurs in RAG systems.

### Deployment
This was containerized and deployed on **Google's Cloud Run** at: [x-rag.leonardeshun.com](https://x-rag.leonardeshun.com)

### How it works
Using around 4 documents, the main steps:
- Retrieve top-k chunks using Facebook AI Similarity Search (FAISS) vector "database"
dense vector representations (embeddings) was created with `all-MiniLM-L6-v2` from `sentence-transformers (HuggingFace)` to capture the semantic meaning of sentences and short paragraphs.
- Generate responses with OpenAI's `gpt-4.1-mini`.
- Attribute sentences by matching them to their sources (cosine similarity)
- Compute token-level saliency (E5 embeddings) to show level of reliance on the source
- Flag low-support sentences for hallucination detection


### Context
I uploaded 4 documents for it to use for context. If questions are asked from outside these questions, it is supposed to indicate that it doesn't have an answer. This is due to the restricted prompting template to enable me undertake this controlled experiment. The documents are on:
1. GradCAM - An article I wrote on GradCAM and the Original GradCAM Paper with images and formulae removed.
1. A Hacker's Mind - A reflection I wrote on this book by Bruce Shneier.
1. Recent emerging techniques in XAI to enhance the interpretable and understanding of AI Models for humans. by Mathew et al. - Another reflection I wrote.
In it current state, it can use any .txt or .md document dumped in the folder `documents`. Please find this in my GitHub - siliconshells.

### Some questions
1. What is Artificial intelligence
1. Tell me about a hacker's mind
1. What is XAI
1. What is GradCAM
1. What is CNN
1. What is mechanistic interpretability


### Retrieved Documents
This tab tells us which of the saved text chunks were retrieved as context for the generation LLM to generate the answer. This is limited to pick the top 3 chunks and indicate from which file the chunk comes from.


### Token-Level Saliency
Token-Level Saliency shows how much each individual token (word) contributes to a model’s prediction. It compares the question to the answer to determine this.
Think of it as a microscope for model decisions: instead of only saying why the model answered something, it highlights which exact tokens pushed the model toward its output and by how much.

### Low-Support Sentences (Possible Hallucinations)
This tab indicates sentences with very low sentence attribution, indicating that it doesn't have any real closeness in semantic meaning with any of the chunks generated and therefore probably not coming from the context. Which also suggest it could be coming out of the generation LLM's hallucination.


### Technical Features
- FAISS retrieval  
- Sentence attribution  
- Token saliency  
- Hallucination detection  
- Flask + Gunicorn + Nginx  
- Redis caching  
- Docker + docker-compose  
- CI tests  