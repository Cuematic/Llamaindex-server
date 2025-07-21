import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import uvicorn

# Initialize FastAPI with CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lightweight configuration for Render
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load minimal data
try:
    documents = SimpleDirectoryReader("data", num_files_limit=1).load_data()
    index = VectorStoreIndex.from_documents(documents)
except Exception as e:
    print(f"Document loading error: {str(e)}")
    index = None

@app.get("/")
def health_check():
    return {"status": "ready", "message": "Server is running"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Must use 10000 for Render
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        reload=False  # Disable reload for production
    )
