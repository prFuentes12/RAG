import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyMuPDFLoader  
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

def get_vector_db_retriever():
    persist_path = "chroma_db"
    embeddings = OpenAIEmbeddings()

    # Verificar si existe y contiene documentos
    if os.path.exists(persist_path):
        db = Chroma(persist_directory=persist_path, embedding_function=embeddings)
        data = db.get()
        if data["documents"]:  # Solo usar si tiene datos
            print(f"Base vectorial cargada con {len(data['documents'])} documentos.")
            return db.as_retriever()
        else:
            print("La base existe pero está vacía. Se regenerará.")

    # Cargar PDFs
    pdf_files = [f for f in os.listdir("data") if f.endswith(".pdf")]
    docs = []
    for pdf_file in pdf_files:
        pdf_loader = PyMuPDFLoader(os.path.join("data", pdf_file))
        docs.extend(pdf_loader.load())

    # Trocear en chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Crear base y guardar
    db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_path)
    db.persist()
    print(f"Base vectorial generada con {len(chunks)} chunks.")
    return db.as_retriever()
