from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def get_llm():
    key=os.getenv("MISTRAL_API_KEY")
    if not key: raise RuntimeError("MISTRAL_API_KEY is not set")
    return ChatMistralAI(model_name="mistral-small-2603",temperature=0,mistral_api_key=key)

action_items_prompt=ChatPromptTemplate.from_messages([("system","Extract all explicit action items. Return concise bullet points. Do not infer information. If none, return 'No action items found.'"),("human","{transcript}")])
decisions_prompt=ChatPromptTemplate.from_messages([("system","Extract all explicit decisions. Return concise bullet points. Do not infer decisions. If none, return 'No decisions found.'"),("human","{transcript}")])
questions_prompt=ChatPromptTemplate.from_messages([("system","Extract all questions from the transcript. Include answered and unanswered questions. Do not invent questions. If none, return 'No questions found.'"),("human","{transcript}")])
combine_prompt=ChatPromptTemplate.from_messages([("system","Merge the partial outputs into one final result. Remove duplicates, preserve unique information, do not add information, and keep it concise."),("human","{summaries}")])

def workflow(prompt,text):
    llm=get_llm(); chain=prompt|llm|StrOutputParser(); combine=combine_prompt|llm|StrOutputParser()
    splitter=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=200)
    outputs=[chain.invoke({"transcript":chunk}) for chunk in splitter.split_text(text)]
    return combine.invoke({"summaries":"\n\n".join(outputs)})

def extract_actions(transcript): return workflow(action_items_prompt,transcript)
def extract_decisions(transcript): return workflow(decisions_prompt,transcript)
def extract_questions(transcript): return workflow(questions_prompt,transcript)
