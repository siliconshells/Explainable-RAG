FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# 3. Bake in the model (Inline - No separate script needed)
# This will save the model to /app/models inside the image
RUN python3 -c "from sentence_transformers import SentenceTransformer; \
                model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
                model.save('./models')"

# 4. Copy the rest of the application code
COPY . .

# 5. Command
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:create_app()"]
