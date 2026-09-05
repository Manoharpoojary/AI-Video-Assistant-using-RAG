import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from core.extractor import extract_actions, extract_decisions, extract_questions
from core.ragengine import ask_question, build_rag_chain
from core.summarize import summarize
from core.transcriber import transcribe_audio
from utils.audioProcessing import process_audio

app = FastAPI(title="AI Video Assistant", version="1.0.0")

SESSIONS: dict[str, dict] = {}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024


class QuestionRequest(BaseModel):
    session_id: str
    question: str


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Video Assistant</title>
<style>
body{font-family:Arial,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;line-height:1.5}
.card{border:1px solid #ddd;border-radius:12px;padding:20px;margin:16px 0}
input,button{padding:10px;margin:5px 0}button{cursor:pointer}
textarea{width:100%;min-height:90px;padding:10px;box-sizing:border-box}
pre{white-space:pre-wrap;background:#f6f6f6;padding:15px;border-radius:8px}
</style>
</head>
<body>
<h1>🎥 AI Video Assistant</h1>
<p>Upload an audio/video file or provide a YouTube URL. Transcription is performed locally with Whisper.</p>
<div class="card">
<form id="processForm">
<label>YouTube URL</label><br><input id="url" name="url" style="width:100%" placeholder="https://www.youtube.com/watch?v=...">
<p>or upload a file:</p><input id="file" name="file" type="file" accept="audio/*,video/*"><br>
<button type="submit">Process video</button>
</form>
<div id="status"></div>
</div>
<div id="results"></div>
<div class="card" id="chatCard" style="display:none">
<h2>💬 Chat with transcript</h2>
<textarea id="question" placeholder="What was the main decision?"></textarea><br>
<button onclick="ask()">Ask</button>
<pre id="answer"></pre>
</div>
<script>
let sessionId=null;
const status=document.getElementById('status');
document.getElementById('processForm').onsubmit=async(e)=>{
 e.preventDefault(); status.textContent='Processing... Whisper and RAG may take a while.';
 const fd=new FormData(); const url=document.getElementById('url').value.trim(); const file=document.getElementById('file').files[0];
 if(url) fd.append('url',url); if(file) fd.append('file',file);
 try{const r=await fetch('/process',{method:'POST',body:fd}); const d=await r.json(); if(!r.ok) throw new Error(d.detail||'Processing failed');
 sessionId=d.session_id; status.textContent='Done.';
 document.getElementById('results').innerHTML=`<div class="card"><h2>📌 ${escapeHtml(d.title)}</h2><h3>Summary</h3><pre>${escapeHtml(d.summary)}</pre><h3>Action Items</h3><pre>${escapeHtml(d.actions)}</pre><h3>Decisions</h3><pre>${escapeHtml(d.decisions)}</pre><h3>Questions</h3><pre>${escapeHtml(d.questions)}</pre></div>`;
 document.getElementById('chatCard').style.display='block';
 }catch(err){status.textContent='Error: '+err.message;}
};
async function ask(){const q=document.getElementById('question').value.trim();if(!q||!sessionId)return;document.getElementById('answer').textContent='Thinking...';const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sessionId,question:q})});const d=await r.json();document.getElementById('answer').textContent=d.answer||d.detail;}
function escapeHtml(s){return String(s).replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\\':'&#92;','"':'&quot;'}[c]));}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return INDEX_HTML


@app.get("/health")
def health():
    return {"status": "ok"}


def _process(source: str, work_dir: Path):
    chunks = process_audio(source, work_dir=work_dir)
    transcript = transcribe_audio(chunks)
    if not transcript.strip():
        raise RuntimeError("Whisper returned an empty transcript.")

    session_dir = work_dir / "vector_db"
    rag_chain = build_rag_chain(transcript, persist_directory=session_dir)
    summary = summarize(transcript)

    return {
        "summary": summary,
        "actions": extract_actions(transcript),
        "decisions": extract_decisions(transcript),
        "questions": extract_questions(transcript),
        "rag_chain": rag_chain,
    }


@app.post("/process")
async def process_video(
    url: str = Form(default=""),
    file: UploadFile | None = File(default=None),
):
    if not url and not file:
        raise HTTPException(status_code=400, detail="Provide a YouTube URL or upload a file.")
    if url and file:
        raise HTTPException(status_code=400, detail="Provide either a YouTube URL or a file, not both.")

    work_dir = Path(tempfile.mkdtemp(prefix="ai_video_"))
    source = url
    uploaded_path = None

    try:
        if file:
            suffix = Path(file.filename or "upload").suffix or ".bin"
            uploaded_path = work_dir / f"input{suffix}"
            total = 0
            with uploaded_path.open("wb") as out:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail="File exceeds the 500 MB upload limit.")
                    out.write(chunk)
            source = str(uploaded_path)

        result = await run_in_threadpool(_process, source, work_dir)
        session_id = uuid.uuid4().hex
        SESSIONS[session_id] = {
            "rag_chain": result["rag_chain"],
            "work_dir": work_dir,
        }

        summary = result["summary"]
        return {
            "session_id": session_id,
            "title": summary.title,
            "summary": summary.summary,
            "actions": result["actions"],
            "decisions": result["decisions"],
            "questions": result["questions"],
        }
    except HTTPException:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if uploaded_path and uploaded_path.exists():
            uploaded_path.unlink(missing_ok=True)


@app.post("/ask")
async def ask(request: QuestionRequest):
    session = SESSIONS.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Process a video first.")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    answer = await run_in_threadpool(ask_question, session["rag_chain"], request.question.strip())
    return {"answer": answer}
