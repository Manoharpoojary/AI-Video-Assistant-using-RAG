# 🎥 AI Video Assistant

An AI-powered video assistant that converts YouTube videos or local audio/video files into structured insights using Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and vector search.

## ✨ Features

* 🎙️ Transcribes audio from YouTube videos or local files.
* 📝 Generates an accurate summary with a concise title.
* ✅ Extracts action items.
* 🎯 Identifies key decisions.
* ❓ Extracts important questions.
* 💬 Chat with the transcript using Retrieval-Augmented Generation (RAG).
* 📚 Stores transcript embeddings in ChromaDB for semantic retrieval.

---

# 🛠️ Tech Stack

### LLM

* Mistral AI (`mistral-small-2603`)

### Embeddings

* Hugging Face
* `sentence-transformers/all-MiniLM-L6-v2`

### Vector Database

* ChromaDB

### Frameworks

* LangChain
* Pydantic

### Audio Processing

* FFmpeg
* yt-dlp

### Programming Language

* Python

---

# 📂 Project Structure

```text
AIvideoAssistant/
│
├── core/
│   ├── extractor.py
│   ├── ragengine.py
│   ├── summarize.py
│   ├── transcriber.py
│   └── vector_store.py
│
├── utils/
│   └── audioProcessing.py
│
├── transcripts/
│
├── vector_db/
│
├── main.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Workflow

```text
Video / Audio
      │
      ▼
Audio Processing
      │
      ▼
Transcription
      │
      ▼
Transcript
      ├───────────────┐
      ▼               │
 Summarization        │
 Action Items         │
 Decisions            │
 Questions            │
                      ▼
              Chunking
                      ▼
                 Embeddings
                      ▼
                  ChromaDB
                      ▼
                  Retriever
                      ▼
                  Mistral AI
                      ▼
              Conversational RAG
```

---

# 🚀 Installation

Clone the repository

```bash
git clone <repository-url>
cd AIvideoAssistant
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file in the project root.

```env
MISTRAL_API_KEY=your_mistral_api_key
```

---

# ▶️ Running the Project

```bash
python main.py
```

or

```bash
uv run python main.py
```

Enter either

* A YouTube URL
* A local audio/video file path

Example

```text
Enter YouTube URL or local file path:
https://www.youtube.com/watch?v=xxxxxxxx
```

---

# 📌 Output

The assistant generates:

* Video Title
* Summary
* Action Items
* Key Decisions
* Questions

Example

```text
📌 TITLE

Retrieval-Augmented Generation

📝 SUMMARY

...

✅ ACTION ITEMS

• Build the retriever
• Update the documentation

🎯 KEY DECISIONS

• Use ChromaDB

❓ QUESTIONS

• Should we use hybrid search?
```

---

# 💬 Chat with the Transcript

After processing, ask natural language questions such as:

```text
Who was assigned the deployment?

What was the final decision?

What deadline was discussed?

Summarize the discussion about RAG.

Who asked about vector databases?
```

The assistant retrieves relevant transcript chunks using ChromaDB and answers using Mistral AI.

---

# 📚 Technologies Used

* Python
* LangChain
* Mistral AI
* Hugging Face Embeddings
* ChromaDB
* Pydantic
* Recursive Character Text Splitter
* Retrieval-Augmented Generation (RAG)

---

# 🔮 Future Improvements

* Streaming responses
* Conversation memory for follow-up questions
* Speaker diarization
* Multilingual transcription
* Hybrid retrieval (keyword + vector search)
* Web interface (Streamlit/FastAPI)
* PDF, DOCX, and Markdown export
* Meeting analytics and timeline generation

---

<!-- # 👨‍💻 Author

**Manohar Poojary**

Built as a Generative AI project to demonstrate:

* LLM Applications
* Retrieval-Augmented Generation (RAG)
* Vector Databases
* LangChain
* AI-powered Transcript Analysis -->
