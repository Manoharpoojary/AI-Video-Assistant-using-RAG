from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
import os

class VideoSummary(BaseModel):
    title: str
    summary: str

def get_llm():
    api_key=os.getenv("MISTRAL_API_KEY")
    if not api_key: raise RuntimeError("MISTRAL_API_KEY is not set")
    return ChatMistralAI(model_name="mistral-small-2603",temperature=0,mistral_api_key=api_key)

def split_transcript(transcript):
    return RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=100).split_text(transcript)

def summarize(transcript):
    llm=get_llm()
    chain=ChatPromptTemplate.from_messages([
        ("system","Summarize the transcript accurately. Keep only important information, remove repetition, and do not invent facts."),
        ("human","{text}")]) | llm | StrOutputParser()
    partial=[chain.invoke({"text":chunk}) for chunk in split_transcript(transcript)]
    final_prompt=ChatPromptTemplate.from_messages([
        ("system","Merge the summaries into one coherent summary. Remove duplicates, preserve key information, generate a concise title, and return a title and summary."),
        ("human","{summaries}")])
    return (final_prompt | llm.with_structured_output(VideoSummary)).invoke({"summaries":"\n\n".join(partial)})
