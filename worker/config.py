import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────
# Volume Configuration (Vast.ai / RunPod persistent storage)
# ──────────────────────────────────────────────────────────────
# Set VOLUME_PATH to your mounted volume path (e.g., /workspace on vast.ai)
# Models will be stored here to persist across instance restarts
VOLUME_PATH = os.getenv("VOLUME_PATH", "/workspace")  # Default vast.ai volume path
VOLUME_ROOT = Path(VOLUME_PATH)

# ── Paths ──
WORKER_ROOT = Path(__file__).parent
# Store models in volume (persistent across instances)
MODELS_DIR = VOLUME_ROOT / "models"
# Store HuggingFace cache in volume (persistent)
HF_CACHE_DIR = VOLUME_ROOT / "hf_cache"
# Working data (can be local, recreated each time)
DATA_DIR = WORKER_ROOT / "data"
INPUTS_DIR = DATA_DIR / "inputs"

# Set HuggingFace cache to volume
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR / "transformers")
os.environ["DIFFUSERS_CACHE"] = str(HF_CACHE_DIR / "diffusers")

# ── Device ──
DEVICE = os.getenv("DEVICE", "cuda")
DTYPE = "float16"  # float16 for GPU; change to float32 for CPU fallback

# ──────────────────────────────────────────────────────────────
# Optimization Settings (Tier 2 Balanced - default)
# ──────────────────────────────────────────────────────────────
# Change these values directly in code instead of using export commands
# Tier 1 (conservative): USE_LCM_LORA=True, LCM_STEPS=8, KEYFRAME_INTERVAL=1, INFERENCE_WIDTH=0
# Tier 2 (balanced):     USE_LCM_LORA=True, LCM_STEPS=6, KEYFRAME_INTERVAL=5, INFERENCE_WIDTH=512
# Tier 3 (aggressive):   USE_LCM_LORA=True, LCM_STEPS=4, KEYFRAME_INTERVAL=10, INFERENCE_WIDTH=384
# Baseline (none):       USE_LCM_LORA=False, KEYFRAME_INTERVAL=1, INFERENCE_WIDTH=0

# ── LCM-LoRA Optimization ──
USE_LCM_LORA = True  # Enable LCM-LoRA (5-6x speedup)
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"
LCM_STEPS = 6  # 4-8 recommended (6 = balanced quality/speed)

# ── Keyframe Optimization ──
KEYFRAME_INTERVAL = 5  # 1=all frames (slow), 5=1 in 5 (balanced), 10=1 in 10 (fast)
INTERPOLATION_METHOD = "opencv_dis"  # Optical flow method

# ── Resolution Optimization ──
INFERENCE_WIDTH = 512   # 0=original (slow), 512=balanced, 384=fast
INFERENCE_HEIGHT = 512  # Match INFERENCE_WIDTH
ENABLE_UPSCALING = False  # Set to True to upscale back to original resolution

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
