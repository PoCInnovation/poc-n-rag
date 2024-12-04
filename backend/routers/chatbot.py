from fastapi import APIRouter, UploadFile, File
from routers.utils.document_loader_utils import *
import os

router = APIRouter()

@router.post("/chatbot/")
async def chatbot(query: str):
    # Run the retrieval chain and capture detailed result
    # generate the new query based on the actual query to be used in the retrieval

    prompt = "Update the actual question to make it more proficient in the retrieval, i want you to upgrade the quality of the question and to only write the question as the response because i'm going to call a llm directly after it"
    new_query = llm.invoke(query + "\n" + prompt).content
    answer1_with_knowledge = retrieval_chain.invoke({"input": new_query, "top_k": 1}) #FIXME: Update the top_k value
    return {
        "query": query,
        "better_query": new_query,
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