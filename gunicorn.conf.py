import os

# 1. DYNAMIC PORT (Critical)
# Cloud Run injects the PORT environment variable (usually 8080).
# If you hardcode 8000, Cloud Run will think your app failed to start.
port = os.environ.get("PORT", "8080")
bind = f"0.0.0.0:{port}"

# 2. MEMORY MANAGEMENT
# Your RAG app likely loads heavy libraries (NLTK, etc.).
# 4 workers means loading the app 4 times in RAM.
# In the free tier (512MB), this will cause an "Out of Memory" crash.
# Use 1 worker and multiple threads instead.
workers = 2
threads = 8

# 3. TIMEOUT
# Google explicitly recommends setting timeout to 0.
# This hands control of timeouts to the Cloud Run platform load balancer
# rather than Gunicorn killing the process prematurely.
timeout = 0
