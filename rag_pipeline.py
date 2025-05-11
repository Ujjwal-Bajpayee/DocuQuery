import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
load_dotenv()

def load_and_split_docs(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(pages)

def create_vectorstore(docs):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding)
    return vectorstore

def create_qa_chain(vectorstore):
    llm = ChatGroq(model="llama3-8b-8192")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

def summarize_document(text, model_name="llama3-8b-8192"):
    llm = ChatGroq(model=model_name)
    prompt = f"Give a concise TL;DR summary of the following text:\n\n{text[:5000]}"
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)
