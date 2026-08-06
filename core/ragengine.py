from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from core.vector_store import load_vector_store,get_retriver,build_vector_store
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

def get_llm():
    model=ChatMistralAI(model_name="mistral-small-2603",temperature=0,mistral_api_key=os.getenv("MISTRAL_API_KEY"))
    return model

def combine_docs(docs:Document):
    return "\n\n".join([i.page_content for i in docs])
    

def build_rag_chain(transcript:str):
    llm=get_llm()
    build_vector_store(transcript)
    
    vector_store=load_vector_store()
    
    retriver=get_retriver(vector_store)
    

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful AI assistant.

                Answer the user's question using ONLY the provided context.

                Instructions:
                - Use only the information from the context.
                - Do not make up facts or assumptions.
                - If the answer cannot be found in the context, respond:
                "I could not find this information in the provided context."
                - Keep the answer clear, concise, and accurate.
                - If the context contains conflicting information, mention the conflict instead of guessing.

                Context:
                {context}
                """,
                    ),
                    ("human", "{question}"),
                ])
                
    rag_chain=({"context":retriver | RunnableLambda(combine_docs),
               "question":RunnablePassthrough()
               }
               | prompt | llm |StrOutputParser()
    )
    return rag_chain
    
    
    
def ask_question(rag_chain,question):
    return rag_chain.invoke(question)

if __name__=="__main__":
    with open("transcripts/transcript.txt","r",encoding="utf-8") as f:
        file=f.read()
    chain=build_rag_chain(file)
    if chain is not None:
        print("Chain obtained ....")