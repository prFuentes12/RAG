# utils.py

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader  # O usa otro loader si es PDF, etc.
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def get_vector_db_retriever():
    persist_path = "chroma_db"  # Persistente, no temporal
    embeddings = OpenAIEmbeddings()

    if os.path.exists(persist_path):
        db = Chroma(persist_directory=persist_path, embedding_function=embeddings)
        return db.as_retriever()

    # Cargar y trocear tus documentos
    loader = TextLoader("data/documento.txt")  # Reemplaza con tus docs
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Crear y guardar la base vectorial
    db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_path)
    db.persist()
    return db.as_retriever()
