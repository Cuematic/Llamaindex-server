import os
from fastapi import FastAPI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import uvicorn

# Optional: Set up HuggingFace local embedding model instead of OpenAI
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load your documents
documents = SimpleDirectoryReader("data").load_data()

# Set up Qdrant vector store
qdrant_client = QdrantClient(":memory:")  # For demo; replace with actual Qdrant URL or local instance
vector_store = QdrantVectorStore(client=qdrant_client, collection_name="demo_collection")

# Build index
index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)

# Create FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "LlamaIndex server is running without OpenAI 🎉"}

# Run with dynamic port for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render provides PORT
    uvicorn.run("main:app", host="0.0.0.0", port=port)
