from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
import time
import os

os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)

index_name = "poc-n-rag"
namespace = "default"

model_name = 'multilingual-e5-large'
embeddings = PineconeEmbeddings(
    model=model_name,
    pinecone_api_key=os.environ.get('PINECONE_API_KEY')
)

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

headers_to_split_on = [
    ("##", "Header 2")
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, strip_headers=False
)

pdf_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, strip_headers=False
)


docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
    namespace=namespace
)

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
retriever=docsearch.as_retriever()

llm = ChatOpenAI(
    openai_api_key=os.environ.get('OPENAI_API_KEY'),
    model_name='gpt-4o-mini',
    temperature=0.0
)

combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
