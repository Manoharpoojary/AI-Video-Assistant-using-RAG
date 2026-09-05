from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "meeting_transcript"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
    )


def build_vector_store(transcript: str, persist_directory) -> Chroma:
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=150)
    chunks = splitter.split_text(transcript)
    documents = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]
    return Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        persist_directory=str(persist_directory),
        collection_name=COLLECTION_NAME,
    )


def load_vector_store(persist_directory) -> Chroma:
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
    )


def get_retriver(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
