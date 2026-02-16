import torch
from PIL import Image
from pathlib import Path
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

from pipelines.avatar.style_config import StyleConfig


def load_pipeline(style: StyleConfig, device: str, dtype_str: str):
    """Load SD1.5 + dual ControlNet pipeline for the given style."""
    import config

    dtype = torch.float16 if dtype_str == "float16" else torch.float32

    controlnet_openpose = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose",
        torch_dtype=dtype,
    )
    controlnet_depth = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth",
        torch_dtype=dtype,
    )

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        style.model_id,
        controlnet=[controlnet_openpose, controlnet_depth],
        torch_dtype=dtype,
        safety_checker=None,
    )

    # Load LCM-LoRA if enabled globally and in style config
    if config.USE_LCM_LORA and style.lcm_enabled:
        from diffusers import LCMScheduler
        pipe.load_lora_weights(config.LCM_LORA_ID)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        print(f"  Loaded LCM-LoRA ({config.LCM_LORA_ID}), using {style.lcm_steps} steps")
    else:
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        print(f"  Standard inference, using {style.num_inference_steps} steps")

    pipe.to(device)
    pipe.enable_attention_slicing()

    return pipe


def stylize_frame(
    pipe,
    style: StyleConfig,
    source_image: Image.Image,
    openpose_image: Image.Image,
    depth_image: Image.Image,
    seed: int = 42,
    target_resolution: tuple = None,
) -> Image.Image:
    """Apply style to a single frame using img2img + ControlNet."""
    import config

    original_size = source_image.size

    # Downscale inputs if target resolution specified
    if target_resolution:
        source_image = source_image.resize(target_resolution, Image.LANCZOS)
        openpose_image = openpose_image.resize(target_resolution, Image.LANCZOS)
        depth_image = depth_image.resize(target_resolution, Image.LANCZOS)

    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    prompt = style.prompt_template.replace("{face_description}", "person face")

    # Use LCM steps if enabled, otherwise fallback to num_inference_steps
    inference_steps = style.lcm_steps if (config.USE_LCM_LORA and style.lcm_enabled) else style.num_inference_steps

    result = pipe(
        prompt=prompt,
        negative_prompt=style.negative_prompt,
        image=source_image,
        control_image=[openpose_image, depth_image],
        num_inference_steps=inference_steps,
        guidance_scale=style.guidance_scale,
        strength=style.denoising_strength,
        controlnet_conditioning_scale=[
            style.controlnet_openpose_weight,
            style.controlnet_depth_weight,
        ],
        generator=generator,
    )

    output = result.images[0]

    # Upscale back to original size if needed
    if target_resolution and config.ENABLE_UPSCALING:
        output = output.resize(original_size, Image.LANCZOS)

    return output


def process_frames(
    style: StyleConfig,
    device: str,
    dtype: str,
    frame_paths: list[Path],
    openpose_paths: list[Path],
    depth_paths: list[Path],
    output_dir: Path,
    seed: int = 42,
) -> list[Path]:
    """Stylize all frames. Returns paths to styled frame images."""
    import config

    output_dir.mkdir(parents=True, exist_ok=True)
    pipe = load_pipeline(style, device, dtype)
    styled_paths = []

    # Determine target resolution from config
    target_res = None
    if config.INFERENCE_WIDTH > 0 and config.INFERENCE_HEIGHT > 0:
        target_res = (config.INFERENCE_WIDTH, config.INFERENCE_HEIGHT)
        print(f"  Using inference resolution: {target_res[0]}x{target_res[1]}")

    for i, (fp, op, dp) in enumerate(zip(frame_paths, openpose_paths, depth_paths)):
        src = Image.open(fp).convert("RGB")
        pose = Image.open(op).convert("RGB")
        depth = Image.open(dp).convert("RGB")

        styled = stylize_frame(pipe, style, src, pose, depth, seed=seed, target_resolution=target_res)
        out_path = output_dir / f"styled_{fp.stem}.png"
        styled.save(out_path)
        styled_paths.append(out_path)

        if i % 10 == 0:
            print(f"  Stylize: {i+1}/{len(frame_paths)}")

    del pipe
    torch.cuda.empty_cache()
    return styled_paths
