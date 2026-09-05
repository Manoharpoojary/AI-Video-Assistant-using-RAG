from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from dotenv import load_dotenv
from core.llm import get_llm, invoke_with_retry
load_dotenv()


class VideoSummary(BaseModel):
    title: str
    summary: str


class VideoAnalysis(VideoSummary):
    actions: str
    decisions: str
    questions: str

def split_transcrpit(transcript:str)->list:
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
    )
    return splitter.split_text(transcript)
    
def summarize(transcript: str) -> VideoSummary:
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
    chunks_summary = [
        invoke_with_retry(chain, {"text": chunk}, operation=f"summarizing chunk {index + 1}/{len(chunks)}")
        for index, chunk in enumerate(chunks)
    ]
    combined_sum="\n\n".join(chunks_summary)
    
    combine_summary_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Merge the summaries into one coherent summary. Remove duplicates, preserve all key information, generate a concise title, and return a title and summary."""
    ),
    ("human", "{summaries}"),
])
    complete_summary_chain=combine_summary_prompt | structured_llm 
    
    
    return invoke_with_retry(
        complete_summary_chain,
        {"summaries": combined_sum},
        operation="creating the final summary",
    )


def analyze(transcript: str) -> VideoAnalysis:
    """Produce all initial video insights with the fewest possible API calls.

    A short video now uses one request instead of separate summary, action,
    decision, and question workflows. Long transcripts are condensed per chunk
    and then merged, keeping requests proportional to their length.
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(VideoAnalysis)
    chunks = split_transcrpit(transcript)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Analyze this video transcript accurately and do not invent facts.

Return a concise title and summary. Also list only explicit action items,
decisions, and questions. For any category with none, return exactly
"No action items found.", "No decisions found.", or "No questions found.".""",
        ),
        ("human", "{text}"),
    ])

    # The common short-video case: one Mistral request total.
    if len(chunks) == 1:
        return invoke_with_retry(
            prompt | structured_llm,
            {"text": chunks[0]},
            operation="analyzing the transcript",
        )

    chunk_chain = prompt | llm | StrOutputParser()
    partial_analyses = [
        invoke_with_retry(
            chunk_chain,
            {"text": chunk},
            operation=f"analyzing chunk {index + 1}/{len(chunks)}",
        )
        for index, chunk in enumerate(chunks)
    ]
    merge_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Merge these partial transcript analyses. Remove duplicates and
return an accurate title, summary, action items, decisions, and questions.
Do not add facts not present in the partial analyses.""",
        ),
        ("human", "{text}"),
    ])
    return invoke_with_retry(
        merge_prompt | structured_llm,
        {"text": "\n\n".join(partial_analyses)},
        operation="combining the transcript analysis",
    )

if __name__=="__main__":
    with open("transcripts/transcript.txt","r",encoding="utf-8") as f:
            file=f.read()
    print("file read successfully...")
    result=summarize(file)
    print(result)


