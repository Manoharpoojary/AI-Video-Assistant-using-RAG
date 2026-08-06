import whisper
import os
import requests
from pathlib import Path
from pydub import AudioSegment

SARVAM_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = "saaras:v2.5"



WHISPER_MODEL=os.getenv("WHISPER_MODEL","small")

_model=None

def load_model():
    
    global _model
    
    if _model is None:
        print("loading model")
        _model=whisper.load_model(WHISPER_MODEL)
        print("model downloaded successfully")
    return _model

def transcribe_chunk_whisper(chunk_path:str,translate : bool=False)->str:
    model=load_model()
    task="translate" if translate else "transcribe"
    result=model.transcribe(chunk_path,task=task)
    
    return result["text"]


def transcribe_chunk_sarvam(chunk_path: Path) -> str:
    SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set")
    
    headers = {
        "api-subscription-key": SARVAM_API_KEY
    }

    with open(chunk_path, "rb") as f:
        files = {
            "file": (chunk_path.name, f, "audio/wav")
        }

        data = {
            "model": SARVAM_MODEL,
            "with_diarization": "false"
        }

        response = requests.post(
            SARVAM_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        if not response.ok:
            print("Status:", response.status_code)
            print("Response:", response.text)
            return ""
        # response.raise_for_status()

    return response.json()["transcript"]


def transcribe_all_sarvam(wav_path: Path) -> str:
    
    audio = AudioSegment.from_wav(wav_path)

    chunk_length = 25 * 1000  # 30 seconds

    temp_dir = Path("downloads/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    full_transcript = ""

    for i, start in enumerate(range(0, len(audio), chunk_length)):
        chunk = audio[start:start + chunk_length]

        chunk_path = temp_dir / f"chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")

        text = transcribe_chunk_sarvam(chunk_path)

        full_transcript += text + " "

        chunk_path.unlink()

    return full_transcript


def transcribe_all_whisper(chunks:list[Path],translate:bool=False)->str:
    full_transcript=""
    for i,chunk in enumerate(chunks):
        print(f'transcribing chunk {i+1} ')
        text=transcribe_chunk_whisper(str(chunk),translate=translate)
        full_transcript+=text+"  "
    
    print("transcription complete ...")
    return full_transcript     
    
    
from pathlib import Path

def transcribe_audio(
    chunks: list[Path],
    provider: str = "whisper"
) -> str:

    full_transcript = ""

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}")

        if provider.lower() == "sarvam":
            text = transcribe_all_sarvam(chunk)
        else:
            text = transcribe_chunk_whisper(str(chunk))

        full_transcript += text + " "

    print("Transcription complete...")

    return full_transcript


if __name__=="__main__":
    # print(os.getenv("SARVAM_API_KEY"))
    result=transcribe_all_sarvam("downloads/chunks/chunk_0.wav")