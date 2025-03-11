from fastapi import APIRouter, Query, File, UploadFile, HTTPException
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_complete, openai_embed
import pandas as pd
from fastapi.responses import Response, StreamingResponse
from typing import List
import os
from utils import row_to_text
from pdfminer.high_level import extract_text

from lightrag.utils import EmbeddingFunc, always_get_an_event_loop
import inspect


rag = LightRAG(
    working_dir="workspace",
    embedding_func=openai_embed,
    llm_model_func=gpt_4o_complete,
        addon_params={
        "insert_batch_size": 20
    }
)

router_Light_RAG = APIRouter()

"""
Upload csv file to and insert data into lightRAG
"""
@router_Light_RAG.post("/upload")
def upload_document(file: UploadFile = File(...)):
    if file.content_type == "text/csv":
        with open("workspace/data.csv", "wb") as f:
            f.write(file.file.read())

        df = pd.read_csv("workspace/data.csv")
        if 'Horodateur' in df.columns:
            df = df.drop(['Horodateur'], axis=1)

        company_list = []
        for index, row in df.iterrows():
            company_list.append(row_to_text(row))

        rag.insert(company_list)
        os.remove("workspace/data.csv")
        file.file.close()
        return {"message": "Data inserted"}

    elif file.content_type == "application/pdf":
        pdf_miner_text = extract_text(file.file)
        rag.insert([pdf_miner_text])
        return {"message": "PDF inserted"}

    elif file.content_type == "text/plain":
        rag.insert([file.file.read().decode("utf-8")])
        return {"message": "Text inserted"}
    else:
        return {"message": "File type not supported"}

@router_Light_RAG.get("/query")
async def query(query: str):
    result = await rag.aquery(query, param=QueryParam(mode="mix"))
    return Response(content=result, media_type="text/markdown")

async def print_stream(stream):
    async for chunk in stream:
        if chunk:
            print(chunk, end="", flush=True)

@router_Light_RAG.get("/stream/query")
async def stream_query(query: str):
    rag = LightRAG(
        working_dir="workspace",
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_complete,
        llm_model_max_async=4,
        addon_params={
            "insert_batch_size": 20
        }
    )

    loop = always_get_an_event_loop()

    resp = rag.aquery(
        query,
        param=QueryParam(mode="hybrid", stream=True),
    )

    if inspect.isasyncgen(resp):
        loop.run_until_complete(print_stream(resp))
    else:
        print(resp)


    return Response(content="Query completed", media_type="text/markdown")
    # result = await rag.aquery(query, param=QueryParam(mode="mix"))
    # return Response(content=result, media_type="text/markdown")

