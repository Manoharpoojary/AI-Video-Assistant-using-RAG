

```markdown
# AIvideoAssistant

AIvideoAssistant is a Python-based prototype for building an AI-powered video assistant. The project combines audio and video processing, transcription, summarization, chunking, vector storage, and retrieval-augmented generation (RAG) so that questions can be answered using content extracted from media sources.

## Project Overview

This project is designed to help you:

- ingest audio or video files,
- transcribe spoken content into text,
- split transcripts into smaller chunks,
- store chunks in a local vector database,
- retrieve relevant information,
- and generate grounded answers using an LLM.

The core idea is to move beyond relying only on the model’s built-in knowledge and instead answer questions using source content that is retrieved and referenced.

## What the Project Does

The workflow in this project is centered around an AI-assisted video understanding pipeline:

1. Audio or video input is processed.
2. Speech is transcribed into text.
3. The transcript is broken into smaller chunks.
4. The content is stored in a vector database for retrieval.
5. Relevant passages are retrieved and used to answer user questions.
6. The system can also summarize the transcript content.

This makes the project suitable for experimenting with RAG-based question answering over video content.

## Current Project Structure

```text
AIvideoAssistant/
├── .env
├── main.py
├── README.md
├── requirements.txt
├── core
│   ├── extractor.py
│   ├── ragengine.py
│   ├── summarize.py
│   ├── transcriber.py
│   └── vector_store.py
├── downloads
│   ├── What is Retrieval-Augmented Generation (RAG)？.wav
│   └── chunks
│       └── chunk_0.wav
├── transcripts
│   └── transcript.txt
├── utils
│   └── audioProcessing.py
└── vector_db
    ├── chroma.sqlite3
    └── 8178e896-a1d5-4938-a831-66933915122c
        ├── data_level0.bin
        ├── header.bin
        ├── length.bin
        └── link_lists.bin
```

## Main Components

- `main.py`: entry point for the application.
- `core/transcriber.py`: handles transcription of audio into text.
- `core/extractor.py`: processes or extracts content from media input.
- `core/vector_store.py`: stores and indexes text chunks in a vector database.
- `core/ragengine.py`: performs retrieval and answer generation.
- `core/summarize.py`: summarizes transcript content using an LLM.
- `utils/audioProcessing.py`: helper functions for audio processing.
- `downloads/`: stores downloaded or generated audio files.
- `transcripts/`: stores transcript text files.
- `vector_db/`: stores local vector database files.

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a .env file in the project root and add any required API keys or configuration values. For example:

```env
MISTRAL_API_KEY=your_key_here
```

4. Make sure audio tools such as `ffmpeg` are available if your workflow depends on audio processing features.

## Run the Project

From the project root, run:

```bash
python main.py
```

## Notes

- The project currently includes a sample transcript in `transcripts/transcript.txt`.
- Generated audio files and vector database data are stored locally in `downloads/` and `vector_db/`.
- This is a prototype project and is intended for experimentation, learning, and further development.

## Suggested Next Steps

- Add support for multiple video or audio files.
- Improve transcript chunking and retrieval quality.
- Add a chat-style interface for querying the assistant.
- Show source citations for each generated answer.
- Add evaluation tests for retrieval and answer quality.
```
