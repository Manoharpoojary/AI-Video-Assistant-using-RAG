import os
from pathlib import Path

import whisper

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
_model = None


def load_model():
    global _model
    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL}")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded successfully")
    return _model


def transcribe_chunk_whisper(chunk_path: str, translate: bool = False) -> str:
    model = load_model()
    task = "translate" if translate else "transcribe"
    result = model.transcribe(chunk_path, task=task)
    return result["text"].strip()


def transcribe_all_whisper(chunks: list[Path], translate: bool = False) -> str:
    full_transcript = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"Transcribing chunk {i}/{len(chunks)}")
        text = transcribe_chunk_whisper(str(chunk), translate=translate)
        if text:
            full_transcript.append(text)
    print("Transcription complete")
    return " ".join(full_transcript)


def transcribe_audio(chunks: list[Path], translate: bool = False) -> str:
    return transcribe_all_whisper(chunks, translate=translate)


if __name__ == "__main__":
    print("This module provides Whisper transcription for the AI Video Assistant.")
