from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_DIR="vector_db"
COLLECTION_NAME="meeting_transcript"
EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device":"cpu"}
    )
    
def build_vector_store(transcript:str)->Chroma:
    
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=150
    )
    chunks=splitter.split_text(transcript)
    
    chunked_docs=[Document(page_content=chunk,metadata={"chunk_index":i}) for i,chunk in enumerate(chunks) ]
    
    embeddings=get_embeddings()
    vector_db=Chroma.from_documents(
        documents=chunked_docs,
        persist_directory=CHROMA_DIR,
        embedding=embeddings,
        collection_name=COLLECTION_NAME      
              )
    
    
def load_vector_store()->Chroma:
    embeddings=get_embeddings()
    vector_store=Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME   
    )
    
    return vector_store

def get_retriver(vector_store:Chroma,k:int=4):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
if __name__=="__main__":
    with open("transcripts/transcript.txt","r",encoding="utf-8") as f:
        file=f.read()
    build_vector_store(file)
    
