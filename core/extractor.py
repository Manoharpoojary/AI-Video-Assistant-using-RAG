from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
from core.llm import get_llm, invoke_with_retry

from langchain_core.prompts import ChatPromptTemplate

action_items_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Extract all explicit action items from the transcript.

- Include only assigned tasks or actions.
- Return concise bullet points.
- Do not infer information.
- If none, return "No action items found."
"""
    ),
    ("human", "{transcript}"),
])


decisions_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Extract all explicit decisions from the transcript.

- Return concise bullet points.
- Do not infer decisions.
- If none, return "No decisions found."
"""
    ),
    ("human", "{transcript}"),
])


questions_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Extract all questions from the transcript.

- Include answered and unanswered questions.
- Return concise bullet points.
- Do not invent questions.
- If none, return "No questions found."
"""
    ),
    ("human", "{transcript}"),
])


combine_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Merge the partial outputs into one final result.

- Remove duplicates.
- Preserve all unique information.
- Do not add new information.
- Keep the output concise.
- If empty, return "No information found."
"""
    ),
    ("human", "{summaries}"),
])

def workflow(prompt: ChatPromptTemplate,
             combine_prompt: ChatPromptTemplate,
             text: str) -> str:

    llm = get_llm()

    chunk_chain = prompt | llm | StrOutputParser()
    combine_chain = combine_prompt | llm | StrOutputParser()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    partial_outputs = []

    for index, chunk in enumerate(chunks):
        output = invoke_with_retry(
            chunk_chain,
            {"transcript": chunk},
            operation=f"extracting information from chunk {index + 1}/{len(chunks)}",
        )
        partial_outputs.append(output)

    final_output = invoke_with_retry(
        combine_chain,
        {"summaries": "\n\n".join(partial_outputs)},
        operation="combining extracted information",
    )
    return final_output
    

def extract_actions(transcript: str):
    return workflow(
        action_items_prompt,
        combine_prompt,
        transcript,
    )


def extract_decisions(transcript: str):
    return workflow(
        decisions_prompt,
        combine_prompt,
        transcript,
    )


def extract_questions(transcript: str):
    return workflow(
        questions_prompt,
        combine_prompt,
        transcript,
    )
    
if __name__=="__main__":
    with open("transcripts/transcript.txt","r",encoding="utf-8") as f:
                file=f.read()
    print("actions===============================================")
    print(extract_actions(file))
    print("Decision==============================================")
    print(extract_decisions(file))
    print("Questions=============================================")
    print(extract_questions(file))
