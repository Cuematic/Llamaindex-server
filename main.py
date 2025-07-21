import os
from fastapi import FastAPI
from qdrant_client import QdrantClient

app = FastAPI()

qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],      # From ENV
    api_key=os.environ["QDRANT_API_KEY"]  # From ENV
)

@app.get("/")
def health_check():
    return {"status": "Secure"}
