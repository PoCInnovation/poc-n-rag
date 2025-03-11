import inspect
import os
import asyncio
from lightrag import LightRAG

# from lightrag.llm import openai_complete, openai_embed
from lightrag.utils import EmbeddingFunc, always_get_an_event_loop
from lightrag import QueryParam

from lightrag.llm.openai import openai_complete, openai_embed

# WorkingDir
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR = "workspace"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
print(f"WorkingDir: {WORKING_DIR}")


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=lambda prompt, **kwargs: openai_complete(
            prompt, **kwargs
        ),
        llm_model_name="gpt-4o",
        llm_model_max_async=4,
        embedding_func=openai_embed,
    )

    await rag.initialize_storages()

    return rag


async def print_stream(stream):
    async for chunk in stream:
        if chunk:
            print(chunk, end="", flush=True)


def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())

    resp = rag.query(
        "explain what u know",
        param=QueryParam(mode="hybrid", stream=True),
    )

    loop = always_get_an_event_loop()
    if inspect.isasyncgen(resp):
        loop.run_until_complete(print_stream(resp))
    else:
        print(resp)


if __name__ == "__main__":
    main()