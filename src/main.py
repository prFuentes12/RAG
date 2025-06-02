import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from utils import get_vector_db_retriever
from langsmith import traceable
from langchain.vectorstores import Chroma

load_dotenv()

# Prompt personalizado
prompt_template = """
Actúas como un sistema de recomendación de expertos.

Tienes una consulta hecha por un usuario. Dispones de una lista de profesionales con sus datos detallados (ID, nombre, profesión, especialidad, biografía y palabras clave).

Tu tarea es:
1. Analizar la consulta del usuario.
2. Evaluar qué perfiles son más relevantes y adecuados para resolver su problema.
3. Ordenarlos del más relevante al menos relevante.
4. Devuelve una lista con formato fijo. No agregues explicaciones ni texto adicional.

Formato por cada perfil:
Orden: N
ID: X
Nombre: Nombre Apellido
Profesión: ...
Especialidad: ...
Palabras Clave: ...
Biografía: ...

Consulta:
{question}

Perfiles:
{context}

"""

prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=prompt_template
)

@cl.on_chat_start
@traceable(name="Creación bdd")
def start_chat():
    retriever = get_vector_db_retriever()
    llm = ChatOpenAI(model_name="gpt-4o-mini-2024-07-18", temperature=0)
    qa_chain = LLMChain(llm=llm, prompt=prompt)

    cl.user_session.set("qa_chain", qa_chain)
    cl.user_session.set("retriever", retriever)

@cl.on_message
@traceable(name="Seleccion de usuarios")
async def handle_message(message: cl.Message):
    qa_chain = cl.user_session.get("qa_chain")
    retriever = cl.user_session.get("retriever")

    # Obtener vectorstore desde el retriever
    vectorstore: Chroma = retriever.vectorstore
    results = vectorstore.similarity_search_with_score(message.content, k=20)

    print("🔎 Documentos recuperados por similitud:", len(results))

    # Ordenar por score (menor = más similar)
    results.sort(key=lambda x: x[1])

    # Mostrar ranking por consola
    print("\n🏅 Ranking por similitud (BDD):\n")
    for i, (doc, score) in enumerate(results, 1):
        meta = doc.metadata
        print(f"{i}. ID: {meta.get('id')} | {meta.get('nombre')} {meta.get('apellido')} | Score: {score:.4f}")

    # Construir solo el contexto para el modelo
    context = ""
    for doc, _ in results:
        meta = doc.metadata
        context += (
            f"ID: {meta.get('id')}\n"
            f"Nombre: {meta.get('nombre')} {meta.get('apellido')}\n"
            f"Profesión: {meta.get('profesion')}\n"
            f"Especialidad: {meta.get('especialidad')}\n"
            f"Biografía: {meta.get('biografia')}\n"
            f"Palabras Clave: {meta.get('palabras_clave')}\n\n"       
        )

    # Ejecutar modelo
    respuesta = qa_chain.run({
        "question": message.content,
        "context": context
    })

    # Solo mostramos la respuesta del modelo al usuario
    await cl.Message(content=respuesta.strip()).send()

