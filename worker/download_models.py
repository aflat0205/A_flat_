import urllib.request
from huggingface_hub import snapshot_download

import config


def download_face_landmarker():
    """Download MediaPipe face_landmarker.task if not present."""
    dest = config.FACE_LANDMARKER_PATH
    if dest.exists():
        print(f"  Already exists: {dest}")
        return
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    )
    print("  Downloading face_landmarker.task ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(dest))
    print(f"  Saved to {dest}")


def download_hf_model(model_id: str, label: str):
    """Download a HuggingFace model to local cache."""
    print(f"  Downloading {label} ({model_id}) ...")
    snapshot_download(model_id, token=config.HF_TOKEN or None)
    print(f"  {label} cached.")


def main():
    print("=== Downloading required models ===\n")

    print("[1/6] MediaPipe Face Landmarker")
    download_face_landmarker()

    print("[2/6] Stable Diffusion 1.5")
    download_hf_model(config.SD15_MODEL_ID, "SD1.5")

    print("[3/6] ControlNet OpenPose")
    download_hf_model(config.CONTROLNET_OPENPOSE_ID, "ControlNet OpenPose")

    print("[4/6] ControlNet Depth")
    download_hf_model(config.CONTROLNET_DEPTH_ID, "ControlNet Depth")

    print("[5/6] MiDaS Depth Estimation")
    download_hf_model(config.DEPTH_MODEL_ID, "MiDaS DPT-Hybrid")

    print("[6/6] Anything v5 (Anime style)")
    download_hf_model(config.ANIME_MODEL_ID, "Anything v5")

    print("\n=== All models downloaded ===")


if __name__ == "__main__":
    main()
