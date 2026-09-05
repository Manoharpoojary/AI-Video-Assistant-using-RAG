from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from utils.audioProcessing import process_audio
from core.transcriber import transcribe_audio
from core.ragengine import build_rag_chain, ask_question
from core.summarize import analyze


def save_transcript(transcript: str, file_name: str = "transcript") -> Path:
    output_dir = Path("transcripts")
    output_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = output_dir / f"{file_name}.txt"
    transcript_path.write_text(transcript, encoding="utf-8")

    return transcript_path


def run_pipeline(source: str):

    print("\n🚀 Starting AI Video Assistant...\n")

    # Process audio
    paths = process_audio(source)

    # Transcribe
    transcript = transcribe_audio(paths, provider="")

    print(f"📝 Transcript Preview:\n{transcript[:300]}...\n")

    # Save transcript
    transcript_path = save_transcript(transcript)
    print(f"✅ Transcript saved to: {transcript_path}\n")

    # Build RAG
    rag_chain = build_rag_chain(transcript)

    analysis = analyze(transcript)
    return {
        "summary": analysis,
        "actions": analysis.actions,
        "questions": analysis.questions,
        "decisions": analysis.decisions,
        "rag_chain": rag_chain,
    }


if __name__ == "__main__":

    source = input("Enter YouTube URL or local file path: ").strip()

    result = run_pipeline(source)
    summary_data=result["summary"]

    print("\n" + "=" * 70)
    print("📋 AI VIDEO ASSISTANT RESULTS")
    print("=" * 70)

    print("\n📌 TITLE")
    print("-" * 70)
    print(summary_data.title)

    print("\n📝 SUMMARY")
    print("-" * 70)
    print(summary_data.summary)

    print("\n✅ ACTION ITEMS")
    print("-" * 70)
    print(result["actions"])

    print("\n🎯 KEY DECISIONS")
    print("-" * 70)
    print(result["decisions"])

    print("\n❓ QUESTIONS")
    print("-" * 70)
    print(result["questions"])

    print("\n" + "=" * 70)
    print("💬 Chat with your transcript")
    print("Type 'exit' to quit")
    print("=" * 70)

    rag_chain = result["rag_chain"]

    while True:

        question = input("\nYou: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("\n👋 Goodbye!")
            break

        if not question:
            continue

        answer = ask_question(rag_chain, question)

        print(f"\n🤖 Assistant:\n{answer}")
