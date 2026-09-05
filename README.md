# 🎥 AI Video Assistant

AI-powered video analysis using **OpenAI Whisper**, Mistral AI, Hugging Face embeddings, and ChromaDB RAG.

## Features

- YouTube URL or local audio/video upload
- Local Whisper transcription
- Summary and title generation
- Action item extraction
- Decision extraction
- Question extraction
- RAG-based chat with the transcript

## Architecture

```text
YouTube / File
      ↓
  yt-dlp + FFmpeg
      ↓
   Audio chunks
      ↓
OpenAI Whisper (local)
      ↓
   Transcript
      ├──→ Mistral → Summary / Actions / Decisions / Questions
      ↓
 Hugging Face Embeddings
      ↓
   ChromaDB
      ↓
 Retriever → Mistral → Answers
```

## Local setup

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

Install **FFmpeg** and make sure the `ffmpeg` executable is on PATH.

Create `.env`:

```env
MISTRAL_API_KEY=your_mistral_api_key
WHISPER_MODEL=small
```

Run the web app:

```bash
uvicorn app:app --reload
```

Open `http://localhost:8000`.

## Render deployment

This repository includes a `Dockerfile` and `render.yaml`.

1. Create a Render account.
2. Create a new **Blueprint** from this GitHub repository.
3. Set `MISTRAL_API_KEY` in the Render environment variables.
4. Deploy.
5. Open the generated Render URL.

`WHISPER_MODEL=small` keeps the current project behavior. For a smaller/faster deployment, this can be changed to another Whisper model through the environment variable.

## Important deployment note

Whisper inference and local Hugging Face embeddings require CPU and memory. Processing time depends heavily on the video duration and selected Whisper model. Chroma data is created per processing session in temporary storage rather than relying on repository files.
