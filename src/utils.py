import os
import json
import time 
from checking import check_embedding_status
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma  

from tqdm import tqdm

CHROMA_PATH = "chroma_db"
JSON_PATH = "./data/UsuariosBDD.json"
BATCH_SIZE = 500  # Cantidad de documentos por lote


def cargar_usuarios_desde_json(json_path):
    with open(json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    print(f"📄 Usuarios cargados del JSON: {len(data)}") 

    docs = []
    for i, user in enumerate(data):

        # 🛠️ Procesar palabras clave (soporta string separado por comas o lista)
        palabras_raw = user['palabras_clave']
        if isinstance(palabras_raw, str):
            palabras = [p.strip() for p in palabras_raw.split(',')]
        else:
            palabras = palabras_raw

        contenido = (
            f"Perfil profesional:\n"
            f"Profesión: {user['profesion']}.\n"
            f"Especialidad: {user['especialidad']}.\n"
            f"Palabras clave en base a sus áreas de experiencia o conocimientos: {', '.join(palabras)}.\n"
            f"Biografía: {user['biografia']}.\n"
        )

        metadata = {
            "id": i,
            "nombre": user["nombre"],
            "apellido": user["apellido"],
            "profesion": user["profesion"],
            "especialidad": user["especialidad"],
            "biografia": user["biografia"],
            "palabras_clave": ', '.join(palabras)
        }


        docs.append(Document(page_content=contenido, metadata=metadata))

    return docs


def get_vector_db_retriever():
    if os.path.exists(CHROMA_PATH):
        print("✅ Existe base de datos, cargando...")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())
    else:
        print("🧠 Generando nueva base vectorial desde JSON...")
        docs = cargar_usuarios_desde_json(JSON_PATH)
        embedding = OpenAIEmbeddings()

        # Crear base con el primer lote
        initial_docs = docs[:BATCH_SIZE]
        db = Chroma.from_documents(
            documents=initial_docs,
            embedding=embedding,
            persist_directory=CHROMA_PATH
        )

        # Agregar el resto por lotes
        for i in tqdm(range(BATCH_SIZE, len(docs), BATCH_SIZE), desc="⚙️ Embedding por lotes"):
            chunk = docs[i:i + BATCH_SIZE]
            db.add_documents(chunk)
            time.sleep(4)

        print("📦 Documentos en base Chroma:", len(db.get()['documents']))
        db.persist()

    check_embedding_status(db)
    print("🧠 Proceso de embedding acabado...")
    return db.as_retriever(search_kwargs={"k": 20})
