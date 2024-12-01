from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import document_loader
from routers import chatbot
from routers.utils.document_loader_utils import *
from pinecone import Pinecone, ServerlessSpec
import os
import time
from contextlib import asynccontextmanager

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=embeddings.dimension,
        metric="cosine",
        spec=spec
    )
    # Wait for index to be ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)


app.include_router(document_loader.router, prefix="/documents", tags=["documents"])
app.include_router(chatbot.router, prefix="/chatbot", tags=["chatbot"])
