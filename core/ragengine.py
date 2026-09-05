from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from core.vector_store import build_vector_store, get_retriver
import os


def get_llm():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set")
    return ChatMistralAI(
        model_name="mistral-small-2603",
        temperature=0,
        mistral_api_key=api_key,
    )


def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(transcript: str, persist_directory):
    vector_store = build_vector_store(transcript, persist_directory)
    retriever = get_retriver(vector_store)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant.
Answer the user's question using ONLY the provided context.
Do not make up facts or assumptions.
If the answer cannot be found in the context, respond:
\"I could not find this information in the provided context.\"
Keep answers clear, concise, and accurate.

Context:
{context}"""),
        ("human", "{question}"),
    ])
    return (
        {"context": retriever | RunnableLambda(combine_docs),
         "question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )


def ask_question(rag_chain, question: str):
    return rag_chain.invoke(question)
