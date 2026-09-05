import whisper
import os

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
_model = None


def load_model():
    global _model
    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL}")
        _model = whisper.load_model(WHISPER_MODEL)
    return _model


def transcribe_chunk_whisper(chunk_path: str, translate: bool = False) -> str:
    task = "translate" if translate else "transcribe"
    return load_model().transcribe(chunk_path, task=task)["text"].strip()


def transcribe_audio(chunks, translate: bool = False) -> str:
    texts = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"Transcribing chunk {i}/{len(chunks)}")
        text = transcribe_chunk_whisper(str(chunk), translate=translate)
        if text:
            texts.append(text)
    return " ".join(texts)
