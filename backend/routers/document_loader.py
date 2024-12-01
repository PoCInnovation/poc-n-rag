from fastapi import APIRouter, UploadFile, File
import os
from routers.utils.document_loader_utils import *
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

docsearch = PineconeVectorStore.from_existing_index(
    index_name="poc-n-rag",
    embedding=embeddings,
)

router = APIRouter()

@router.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    print(file)
    content = await file.read()
    return {"filename": file.filename, "content": content}

@router.post("/uploadMarkdown/")
async def upload_markdown(file: UploadFile = File(...)):
    markdown_document = await file.read()
    markdown_document = markdown_document.decode("utf-8")

    md_header_splits = markdown_splitter.split_text(markdown_document)

    docsearch = PineconeVectorStore.from_documents(
        documents=md_header_splits,
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace
    )
    return {"filename": file.filename, "Uploaded": True}

@router.post("/uploadPDF/")
async def upload_pdf(file: UploadFile = File(...)):
    pdf_file = await file.read()

    with open(f"temp_{file.filename}", "wb") as temp_file:
        temp_file.write(pdf_file)

    pdf_loader = PyPDFLoader(f"temp_{file.filename}")
    pdf_documents = pdf_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    pdf_splits = text_splitter.split_documents(pdf_documents)

    docsearch = PineconeVectorStore.from_documents(
        documents=pdf_splits,
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace
    )

    os.remove(f"temp_{file.filename}")
    return {"filename": file.filename, "Uploaded": True}