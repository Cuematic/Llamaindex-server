import os
from fastapi import FastAPI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import uvicorn

app = FastAPI()

# Lightweight embedding model to save memory
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize with small segment size for memory efficiency
qdrant_client = QdrantClient(":memory:", prefer_grpc=True)
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="demo_collection",
    optimizers_config={"memmap_threshold": 10000}  # Helps with memory
)

# Load only 1 document initially for testing
try:
    documents = SimpleDirectoryReader("data", num_files_limit=1).load_data()
    index = VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store,
        show_progress=True
    )
except Exception as e:
    print(f"Error loading documents: {e}")
    index = None

@app.get("/")
def health_check():
    return {"status": "ready"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render uses 10000
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1,  # Reduce memory usage
        limit_concurrency=1  # Helps with memory
    )
