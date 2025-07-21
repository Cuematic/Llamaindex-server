import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import uvicorn

# Initialize FastAPI
app = FastAPI()

# Configure CORS (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
def initialize_components():
    # Setup embedding model
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Initialize Qdrant (in-memory for demo)
    qdrant_client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=qdrant_client, 
        collection_name="demo_collection"
    )
    
    # Load documents (create 'data' folder in your project)
    documents = SimpleDirectoryReader("data").load_data()
    
    # Create index
    index = VectorStoreIndex.from_documents(
        documents, 
        vector_store=vector_store
    )
    
    return index

# Initialize on startup
index = initialize_components()

# API Endpoints
@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "LlamaIndex Qdrant server running"}

@app.get("/query")
async def query_index(query: str):
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return {"response": str(response)}

# Run with dynamic port for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
