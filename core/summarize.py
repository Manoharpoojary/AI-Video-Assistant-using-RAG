from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from pydantic import BaseModel
import os
from dotenv import load_dotenv
load_dotenv()


class VideoSummary(BaseModel):
    title: str
    summary: str

def get_llm():
    model=ChatMistralAI(model_name="mistral-small-2603",temperature=0,mistral_api_key=os.getenv("MISTRAL_API_KEY"))
    return model
    
def split_transcrpit(transcript:str)->list:
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
    )
    return splitter.split_text(transcript)
    
def summarize(transcript:str)->str:
    llm=get_llm()
    structured_llm = llm.with_structured_output(VideoSummary)
    summary_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Summarize the transcript accurately. Keep only important information, remove repetition, and do not invent facts."""
    ),
    ("human", "{text}"),
])
    chain=summary_prompt | llm |StrOutputParser()
    chunks=split_transcrpit(transcript)
    chunks_summary=[chain.invoke({"text":chunk}) for chunk in chunks]
    combined_sum="\n\n".join(chunks_summary)
    
    combine_summary_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Merge the summaries into one coherent summary. Remove duplicates, preserve all key information, generate a concise title, and return a title and summary."""
    ),
    ("human", "{summaries}"),
])
    complete_summary_chain=combine_summary_prompt | structured_llm 
    
    
    return complete_summary_chain.invoke({"summaries":combined_sum})

if __name__=="__main__":
    with open("transcripts/transcript.txt","r",encoding="utf-8") as f:
            file=f.read()
    print("file read successfully...")
    result=summarize(file)
    print(result)


