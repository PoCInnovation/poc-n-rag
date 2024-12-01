from fastapi import APIRouter, UploadFile, File
from routers.utils.document_loader_utils import *
import os

router = APIRouter()

@router.post("/chatbot/")
async def chatbot(query: str):
    # Run the retrieval chain and capture detailed results
    answer1_with_knowledge = retrieval_chain.invoke({"input": query, "top_k": 1}) #FIXME: Update the top_k value
    return {
        "query": query,
        "response": answer1_with_knowledge['answer'],
        "context": answer1_with_knowledge['context']
    }


@router.post("/chatbot_without_knowledge/")
async def chatbot_without_knowledge(query: str):
    # Retrieve documents directly without combining into a final answer
    results = llm.invoke(query)
    return {
        "query": query,
        "response": results.content
    }