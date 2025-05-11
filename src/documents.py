from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()  # Carga las variables del .env

def mostrar_contenido_bdd():
    persist_path = "chroma_db"
    embeddings = OpenAIEmbeddings()
    
    db = Chroma(persist_directory=persist_path, embedding_function=embeddings)
    data = db.get()

    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    print(f"Total de documentos indexados: {len(documents)}\n")

    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        print(f"--- Documento {i + 1} ---")
        print(f"Contenido: {doc[:300]}...")  # Mostrar primeros 300 caracteres
        print(f"Metadata: {meta}")
        print()

if __name__ == "__main__":
    mostrar_contenido_bdd()