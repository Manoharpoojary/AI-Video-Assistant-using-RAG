"""Web backend for the AI Video Assistant."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

from core.ragengine import ask_question
from main import run_pipeline


STATIC_DIR = Path(__file__).parent / "static"
app = FastAPI(title="AI Video Assistant")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class CreateJob(BaseModel):
    source: str = Field(min_length=1, max_length=2_000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)


@dataclass
class Job:
    id: str
    source: str
    status: Literal["processing", "complete", "failed"] = "processing"
    message: str = "Preparing your video…"
    result: dict | None = None
    error: str | None = None
    rag_chain: object | None = field(default=None, repr=False)


jobs: dict[str, Job] = {}
pipeline_lock = threading.Lock()


def public_job(job: Job) -> dict:
    response = {"id": job.id, "status": job.status, "message": job.message}
    if job.status == "complete" and job.result:
        response["result"] = job.result
    if job.status == "failed":
        response["error"] = job.error
    return response


def process_job(job: Job) -> None:
    # The current Chroma collection is shared, so serialize jobs to prevent one
    # video's transcript being used for another video's chat session.
    try:
        with pipeline_lock:
            job.message = "Downloading audio and creating the transcript…"
            pipeline_result = run_pipeline(job.source)
            summary = pipeline_result["summary"]
            job.result = {
                "title": summary.title,
                "summary": summary.summary,
                "actions": pipeline_result["actions"],
                "decisions": pipeline_result["decisions"],
                "questions": pipeline_result["questions"],
            }
            job.rag_chain = pipeline_result["rag_chain"]
            job.status = "complete"
            job.message = "Your video is ready."
    except Exception as error:
        job.status = "failed"
        job.error = str(error)
        job.message = "The video could not be processed."


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/jobs", status_code=202)
def create_job(request: CreateJob) -> dict:
    source = request.source.strip()
    if not source:
        raise HTTPException(status_code=422, detail="Enter a YouTube URL or local file path.")

    job = Job(id=uuid.uuid4().hex, source=source)
    jobs[job.id] = job
    threading.Thread(target=process_job, args=(job,), daemon=True).start()
    return public_job(job)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return public_job(job)


@app.post("/api/jobs/{job_id}/chat")
def chat(job_id: str, request: ChatRequest) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status != "complete" or job.rag_chain is None:
        raise HTTPException(status_code=409, detail="The video is not ready for chat yet.")

    return {"answer": ask_question(job.rag_chain, request.question.strip())}
