import os
from fastapi import FastAPI, Request
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

app = FastAPI()

# Load Qdrant credentials from environment variables
qdrant_url = os.environ["https://6103c076-8f1b-4be2-85fb-190b54762996.europe-west3-0.gcp.cloud.qdrant.io/"]
qdrant_api_key = os.environ["QDRANT_API_KEY"]

# Setup Qdrant
qdrant = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key
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
