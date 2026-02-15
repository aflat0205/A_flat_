from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter(tags=["avatar jobs"])

# Shared job store with generate module
from .generate import _jobs


class JobStatusUpdate(BaseModel):
    status: str
    progress: Optional[float] = None
    output_url: Optional[str] = None
    error: Optional[str] = None


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    """Get avatar job status by ID."""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    return _jobs[job_id]


@router.patch("/jobs/{job_id}")
def update_job(job_id: str, update: JobStatusUpdate):
    """Update avatar job status (called by worker)."""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    job = _jobs[job_id]
    job["status"] = update.status
    if update.progress is not None:
        job["progress"] = update.progress
    if update.output_url:
        job["output_url"] = update.output_url
    if update.error:
        job["error"] = update.error
    return job


@router.get("/jobs")
def list_jobs():
    """List all avatar jobs."""
    return {"jobs": list(_jobs.values())}
