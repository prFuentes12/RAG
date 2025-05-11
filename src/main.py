import chainlit as cl
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from utils import get_vector_db_retriever

load_dotenv()  # Carga tu OPENAI_API_KEY del .env

@cl.on_chat_start
def start_chat():
    retriever = get_vector_db_retriever()

    llm = ChatOpenAI(model_name="gpt-4o-mini-2024-07-18", temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True  # Para mostrar las fuentes
    )

    cl.user_session.set("qa_chain", qa_chain)

@cl.on_message
async def handle_msg(message: cl.Message):
    qa_chain = cl.user_session.get("qa_chain")
    result = qa_chain(message.content)

    respuesta = result["result"]
    docs = result["source_documents"]

    # Mostrar fuentes (si existen)
    fuentes = "\n".join([f"- {doc.metadata.get('source', 'Desconocido')}" for doc in docs])

    full_response = f"{respuesta}\n\n**Fuentes:**\n{fuentes}"

    await cl.Message(content=full_response).send()