FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image to avoid downloading it at runtime
RUN python3 -c "from sentence_transformers import SentenceTransformer; \
                model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
                model.save('./models')"

COPY . .

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:create_app()"]
