import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──
WORKER_ROOT = Path(__file__).parent
MODELS_DIR = WORKER_ROOT / "models"
DATA_DIR = WORKER_ROOT / "data"
INPUTS_DIR = DATA_DIR / "inputs"

# ── Device ──
DEVICE = os.getenv("DEVICE", "cuda")
DTYPE = "float16"  # float16 for GPU; change to float32 for CPU fallback

# ── Hugging Face ──
HF_TOKEN = os.getenv("HF_TOKEN", "")

# ── Model IDs ──
SD15_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_OPENPOSE_ID = "lllyasviel/control_v11p_sd15_openpose"
CONTROLNET_DEPTH_ID = "lllyasviel/control_v11f1p_sd15_depth"
DEPTH_MODEL_ID = "Intel/dpt-hybrid-midas"
ANIME_MODEL_ID = "stablediffusionapi/anything-v5"

# ── MediaPipe ──
FACE_LANDMARKER_PATH = MODELS_DIR / "face_landmarker.task"

# ── Backend ──
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
