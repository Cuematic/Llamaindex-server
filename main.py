from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import os

app = FastAPI()

# 👇 Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Consider restricting this to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Qdrant setup
qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"]
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
