from dataclasses import dataclass


@dataclass(frozen=True)
class StyleConfig:
    style_id: str
    display_name: str
    # Diffusion parameters
    model_id: str
    prompt_template: str
    negative_prompt: str
    denoising_strength: float
    num_inference_steps: int
    guidance_scale: float
    # ControlNet weights
    controlnet_openpose_weight: float
    controlnet_depth_weight: float
    # Post-processing
    color_match_strength: float
    temporal_blend_frames: int
    # LCM-LoRA optimization
    lcm_enabled: bool = True
    lcm_steps: int = 6  # Override num_inference_steps if LCM enabled


STYLES: dict[str, StyleConfig] = {
    "beauty-realistic": StyleConfig(
        style_id="beauty-realistic",
        display_name="Beauty Realistic",
        model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        prompt_template=(
            "professional portrait photograph, {face_description}, "
            "beauty retouched skin, soft studio lighting, sharp focus, "
            "detailed face, natural colors, 8k uhd"
        ),
        negative_prompt=(
            "cartoon, anime, illustration, painting, drawing, "
            "deformed, ugly, blurry, bad anatomy, disfigured, "
            "poorly drawn face, mutation, extra limbs"
        ),
        denoising_strength=0.35,
        num_inference_steps=25,
        guidance_scale=7.5,
        controlnet_openpose_weight=1.2,
        controlnet_depth_weight=0.8,
        color_match_strength=0.7,
        temporal_blend_frames=2,
        lcm_enabled=True,
        lcm_steps=6,
    ),
    "promptable-avatar": StyleConfig(
        style_id="promptable-avatar",
        display_name="Promptable Avatar",
        model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        prompt_template=(
            "stylized digital portrait, {face_description}, "
            "artistic render, vibrant colors, cinematic lighting, "
            "highly detailed, trending on artstation"
        ),
        negative_prompt=(
            "cartoon, anime, illustration, painting, drawing, "
            "ugly, deformed, disfigured, poorly drawn face, "
            "bad anatomy, mutation, extra limbs, blurry"
        ),
        denoising_strength=0.55,
        num_inference_steps=30,
        guidance_scale=8.0,
        controlnet_openpose_weight=1.0,
        controlnet_depth_weight=0.7,
        color_match_strength=0.4,
        temporal_blend_frames=2,
        lcm_enabled=True,
        lcm_steps=6,
    ),
    "animated-anime": StyleConfig(
        style_id="animated-anime",
        display_name="Animated / Anime",
        model_id="stablediffusionapi/anything-v5",
        prompt_template=(
            "anime portrait, {face_description}, "
            "studio ghibli style, cel shading, clean lines, "
            "vibrant anime colors, detailed anime eyes, masterpiece"
        ),
        negative_prompt=(
            "photorealistic, photograph, 3d render, ugly, "
            "deformed, bad anatomy, poorly drawn, low quality, "
            "worst quality, blurry, extra limbs"
        ),
        denoising_strength=0.65,
        num_inference_steps=30,
        guidance_scale=8.5,
        controlnet_openpose_weight=1.0,
        controlnet_depth_weight=0.6,
        color_match_strength=0.2,
        temporal_blend_frames=3,
        lcm_enabled=True,
        lcm_steps=6,
    ),
}


def get_style(style_id: str) -> StyleConfig:
    """Get style config by ID. Raises ValueError for unknown styles."""
    if style_id not in STYLES:
        raise ValueError(
            f"Unknown style_id '{style_id}'. Valid: {list(STYLES.keys())}"
        )
    return STYLES[style_id]
