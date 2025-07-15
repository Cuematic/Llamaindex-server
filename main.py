from fastapi import FastAPI, Request
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

app = FastAPI()

# Setup Qdrant
qdrant = QdrantClient(
    url="https://YOUR_QDRANT_URL",
    api_key="YOUR_QDRANT_API_KEY"
)
documents = SimpleDirectoryReader("data").load_data()
vector_store = QdrantVectorStore(client=qdrant, collection_name="regulations")
index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)

@app.post("/query")
async def query_llama(request: Request):
    data = await request.json()
    query_engine = index.as_query_engine()
    response = query_engine.query(data["question"])
    return {"answer": str(response)}
