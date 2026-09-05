from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path
import shutil
import tempfile
import uuid
from starlette.concurrency import run_in_threadpool

from utils.audioProcessing import process_audio
from core.transcriber import transcribe_audio
from core.ragengine import build_rag_chain, ask_question
from core.summarize import summarize
from core.extractor import extract_actions, extract_decisions, extract_questions

app = FastAPI(title="AI Video Assistant", version="1.0.0")
SESSIONS = {}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024

class QuestionRequest(BaseModel):
    session_id: str
    question: str

HTML = """
<!doctype html><html><head><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>AI Video Assistant</title><style>
body{font-family:Arial;max-width:900px;margin:40px auto;padding:20px}input,button,textarea{padding:10px;margin:6px 0}input{width:100%;box-sizing:border-box}textarea{width:100%;min-height:90px;box-sizing:border-box}button{cursor:pointer}.card{border:1px solid #ddd;border-radius:12px;padding:20px;margin:15px 0}pre{white-space:pre-wrap}
</style></head><body><h1>🎥 AI Video Assistant</h1><p>Whisper transcription + Mistral RAG.</p>
<div class='card'><form id='f'><input id='url' placeholder='YouTube URL'><p>or upload:</p><input id='file' type='file' accept='audio/*,video/*'><button>Process</button></form><p id='status'></p></div><div id='results'></div>
<div class='card' id='chat' style='display:none'><h2>💬 Chat</h2><textarea id='q' placeholder='What was the main decision?'></textarea><br><button onclick='ask()'>Ask</button><pre id='a'></pre></div>
<script>let sid=null;const $=x=>document.getElementById(x);$('f').onsubmit=async e=>{e.preventDefault();$('status').textContent='Processing with Whisper...';let fd=new FormData(),u=$('url').value.trim(),f=$('file').files[0];if(u)fd.append('url',u);if(f)fd.append('file',f);try{let r=await fetch('/process',{method:'POST',body:fd}),d=await r.json();if(!r.ok)throw Error(d.detail);sid=d.session_id;$('status').textContent='Done';$('results').innerHTML='<div class=card><h2>'+h(d.title)+'</h2><h3>Summary</h3><pre>'+h(d.summary)+'</pre><h3>Action Items</h3><pre>'+h(d.actions)+'</pre><h3>Decisions</h3><pre>'+h(d.decisions)+'</pre><h3>Questions</h3><pre>'+h(d.questions)+'</pre></div>';$('chat').style.display='block'}catch(x){$('status').textContent='Error: '+x.message}};async function ask(){let q=$('q').value.trim();if(!q||!sid)return;$('a').textContent='Thinking...';let r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid,question:q})}),d=await r.json();$('a').textContent=d.answer||d.detail}function h(s){return String(s).replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\\':'&#92;','"':'&quot;'}[c]))}</script></body></html>
"""

@app.get('/', response_class=HTMLResponse)
def home(): return HTML

@app.get('/health')
def health(): return {'status':'ok'}

def process_sync(source, work_dir):
    chunks = process_audio(source, work_dir=work_dir)
    transcript = transcribe_audio(chunks)
    if not transcript: raise RuntimeError('Whisper returned an empty transcript.')
    rag = build_rag_chain(transcript, persist_directory=work_dir/'vector_db')
    return {'summary': summarize(transcript), 'actions': extract_actions(transcript), 'decisions': extract_decisions(transcript), 'questions': extract_questions(transcript), 'rag': rag}

@app.post('/process')
async def process(url: str = Form(''), file: UploadFile | None = File(None)):
    if bool(url) == bool(file): raise HTTPException(400, 'Provide exactly one YouTube URL or file.')
    work_dir = Path(tempfile.mkdtemp(prefix='ai_video_'))
    try:
        if file:
            suffix=Path(file.filename or 'upload').suffix or '.bin'; src=work_dir/f'input{suffix}'; total=0
            with src.open('wb') as out:
                while chunk:=await file.read(1024*1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES: raise HTTPException(413, 'File exceeds 500 MB limit.')
                    out.write(chunk)
            source=str(src)
        else: source=url
        result=await run_in_threadpool(process_sync, source, work_dir)
        sid=uuid.uuid4().hex; SESSIONS[sid]={'rag':result['rag'],'work_dir':work_dir}
        s=result['summary']; return {'session_id':sid,'title':s.title,'summary':s.summary,'actions':result['actions'],'decisions':result['decisions'],'questions':result['questions']}
    except HTTPException: shutil.rmtree(work_dir,ignore_errors=True); raise
    except Exception as e: shutil.rmtree(work_dir,ignore_errors=True); raise HTTPException(500,str(e)) from e

@app.post('/ask')
async def ask(req: QuestionRequest):
    session=SESSIONS.get(req.session_id)
    if not session: raise HTTPException(404,'Session not found. Process a video first.')
    return {'answer':await run_in_threadpool(ask_question,session['rag'],req.question.strip())}
