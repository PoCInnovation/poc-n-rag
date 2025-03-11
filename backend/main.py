from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
from openai import OpenAI
from openai import AsyncOpenAI  # Utilisation du client asynchrone

client = AsyncOpenAI()

app = FastAPI()

async def stream_gpt_4o_mini(query: str):
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": query}],
        stream=True,
    )
    async for chunk in stream:  # Itération asynchrone sur le flux
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

@app.get('/chat/')
async def chat(query: str):
    return StreamingResponse(stream_gpt_4o_mini(query), media_type='text/event-stream')


