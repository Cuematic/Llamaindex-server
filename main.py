import os
from fastapi import FastAPI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

app = FastAPI()

# Initialize at startup (not module level)
@app.on_event("startup")
async def startup_event():
    # Local embedding model
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Load documents - ensure 'data' directory exists in Render
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Warning: 'data' directory was empty")
    
    global index  # Make available app-wide
    documents = SimpleDirectoryReader("data").load_data()
    
    # Use disk persistence for Render's ephemeral storage
    qdrant_client = QdrantClient(path="./qdrant_data")  # Local storage
    vector_store = QdrantVectorStore(
        client=qdrant_client, 
        collection_name="demo_collection"
    )
    index = VectorStoreIndex.from_documents(
        documents, 
        vector_store=vector_store,
        show_progress=True
    )

@app.get("/")
def health_check():
    return {
        "status": "running",
        "qdrant": "local",
        "embedding": "all-MiniLM-L6-v2"
    }

# Only for local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
