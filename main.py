import os
from fastapi import FastAPI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer  # Lightweight alternative

app = FastAPI()

# Initialize with minimal memory footprint
@app.on_event("startup")
async def startup():
    # 1. Initialize Qdrant first (smallest memory footprint)
    app.state.qdrant = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=True  # More efficient
    )
    
    # 2. Lazy-load HuggingFace model only when needed
    app.state.embedder = lambda: SentenceTransformer(
        'sentence-transformers/all-MiniLM-L6-v2',
        device='cpu',
        cache_folder='/tmp/models'
    )

@app.get("/")
def health_check():
    return {
        "status": "OK",
        "memory": os.sys.getsizeof(app.state) / 1024  # KB
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 10000)),
        workers=1,
        limit_max_requests=100,  # Prevents memory leaks
        timeout_keep_alive=5  # Faster resource cleanup
                                  )
