import os
from fastapi import FastAPI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

app = FastAPI()

# Initialize Qdrant
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

@app.on_event("startup")
async def startup():
    # Load documents
    documents = SimpleDirectoryReader("data").load_data()
    
    # Create index
    app.state.index = VectorStoreIndex.from_documents(
        documents,
        vector_store=QdrantVectorStore(
            client=qdrant_client,
            collection_name="llamaindex"
        )
    )

@app.get("/")
def health_check():
    return {"status": "OK", "qdrant": "connected"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
