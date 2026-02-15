import uuid
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter(tags=["avatar assets"])

UPLOAD_DIR = Path("app/data/uploads")
OUTPUT_DIR = Path("app/data/outputs")


@router.post("/assets/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a face-scan video for avatar generation. Returns an asset_id."""
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(400, "Only video files are accepted")

    asset_id = str(uuid.uuid4())
    ext = Path(file.filename or "video.mp4").suffix or ".mp4"
    dest = UPLOAD_DIR / f"{asset_id}{ext}"
    dest.parent.mkdir(parents=True, exist_ok=True)

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {
        "asset_id": asset_id,
        "filename": file.filename,
        "url": f"/media/uploads/{asset_id}{ext}",
    }


@router.get("/assets/{asset_id}")
def get_asset(asset_id: str):
    """Get avatar asset info by ID."""
    for p in UPLOAD_DIR.glob(f"{asset_id}.*"):
        return {"asset_id": asset_id, "url": f"/media/uploads/{p.name}", "type": "upload"}
    for p in OUTPUT_DIR.glob(f"{asset_id}.*"):
        return {"asset_id": asset_id, "url": f"/media/outputs/{p.name}", "type": "output"}
    raise HTTPException(404, "Asset not found")
