import uuid
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter(tags=["avatar generation"])

# In-memory job store (replace with DB later)
_jobs: dict[str, dict] = {}

VALID_STYLES = ["beauty-realistic", "promptable-avatar", "animated-anime"]


class GenerateAvatarRequest(BaseModel):
    asset_id: str
    style_id: str
    seed: Optional[int] = 42


@router.post("/generation")
def generate_avatar(req: GenerateAvatarRequest):
    """Queue an avatar generation job."""
    if req.style_id not in VALID_STYLES:
        raise HTTPException(400, f"Invalid style_id. Must be one of: {VALID_STYLES}")

    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "pipeline": "output_a",
        "asset_id": req.asset_id,
        "style_id": req.style_id,
        "seed": req.seed,
        "status": "queued",
        "created_at": time.time(),
        "progress": 0.0,
        "output_url": None,
        "error": None,
    }
    _jobs[job_id] = job
    return job


@router.get("/styles")
def list_styles():
    """List available avatar styles."""
    return {
        "styles": [
            {"id": s, "name": s.replace("-", " ").title()}
            for s in VALID_STYLES
        ]
    }
