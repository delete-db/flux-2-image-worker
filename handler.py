"""
RunPod Serverless handler for FLUX.2-dev image generation.
Supports text-to-image and multi-reference image editing.
Uses 4-bit quantized model (NF4) for ~20GB VRAM, or full precision on large GPUs.
"""

import base64
import io
import os
import time
from typing import Any

import runpod
import torch
from PIL import Image

# ── Configuration ───────────────────────────────────────────

MODELS_ROOT = os.environ.get("MODELS_ROOT", "/runpod-volume/ComfyUI/models")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(MODELS_ROOT, "flux2-dev-4bit"))
USE_QUANTIZED = os.environ.get("USE_QUANTIZED", "true").lower() == "true"
USE_CPU_OFFLOAD = os.environ.get("USE_CPU_OFFLOAD", "false").lower() == "true"
COMPILE_TRANSFORMER = os.environ.get("COMPILE_TRANSFORMER", "false").lower() == "true"
DEFAULT_STEPS = int(os.environ.get("DEFAULT_STEPS", "28"))
HF_TOKEN = os.environ.get("HF_TOKEN", None)
WORKER_VERSION = "flux2-dev-v2-a100"

# ── Load Pipeline Once ──────────────────────────────────────

print(f"Worker version: {WORKER_VERSION}")
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability()
    print(
        f"  GPU: {torch.cuda.get_device_name(0)} "
        f"(sm_{cc[0]}{cc[1]}, torch {torch.__version__})"
    )
print(f"Loading FLUX.2-dev pipeline...")
print(f"  Model: {MODEL_PATH}")
print(f"  Quantized (4-bit): {USE_QUANTIZED}")
print(f"  CPU offload: {USE_CPU_OFFLOAD}")
print(f"  Compile transformer: {COMPILE_TRANSFORMER}")
print(f"  Default steps: {DEFAULT_STEPS}")

load_start = time.time()

from diffusers import Flux2Pipeline, Flux2Transformer2DModel

torch_dtype = torch.bfloat16

print("  Loading pipeline from local path...")
PIPELINE = Flux2Pipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch_dtype,
    token=HF_TOKEN,
    local_files_only=True,
)

if USE_CPU_OFFLOAD:
    PIPELINE.enable_model_cpu_offload()
    print("  CPU offload enabled")
else:
    PIPELINE.to("cuda")
    PIPELINE.vae.enable_tiling()
    print("  Loaded to CUDA directly (VAE tiling on)")

if COMPILE_TRANSFORMER and not USE_CPU_OFFLOAD:
    # Persist inductor cache on network volume so other workers skip recompile.
    os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR", "/runpod-volume/.torch_compile_cache"
    )
    print(f"  Compile cache dir: {os.environ['TORCHINDUCTOR_CACHE_DIR']}")

    # Block-level compile (PyTorch-recommended for diffusion transformers):
    # compiles repeated blocks instead of the whole model → 8–10x faster compile,
    # same runtime speedup, and avoids the CUDAGraph tensor-overwrite bug that
    # affects `mode="reduce-overhead"` with diffusers pipelines.
    def _compile_block_list(name: str):
        blocks = getattr(PIPELINE.transformer, name, None)
        if blocks is None:
            return 0
        for i, block in enumerate(blocks):
            blocks[i] = torch.compile(block, mode="default", fullgraph=False)
        return len(blocks)

    try:
        compiled = 0
        for attr in ("transformer_blocks", "single_transformer_blocks"):
            compiled += _compile_block_list(attr)
        if compiled:
            print(f"  Compiled {compiled} transformer blocks (mode=default)")
        else:
            # Fallback: compile full transformer if block structure differs
            PIPELINE.transformer = torch.compile(
                PIPELINE.transformer, mode="default", fullgraph=False
            )
            print("  Compiled full transformer (block list not found)")
    except Exception as exc:
        print(f"  torch.compile skipped: {exc}")

load_elapsed = time.time() - load_start
print(f"Pipeline loaded in {load_elapsed:.1f}s")


# ── Helpers ─────────────────────────────────────────────────

def decode_image(image_input: str) -> Image.Image:
    """Decode base64 or download URL image."""
    if image_input.startswith("http://") or image_input.startswith("https://"):
        import requests
        response = requests.get(image_input, timeout=30)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        payload = image_input.split(",", 1)[1] if "," in image_input else image_input
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")


def encode_image(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode PIL image to base64."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, quality=95)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ── Handler ─────────────────────────────────────────────────

@torch.inference_mode()
def handler(job: dict[str, Any]) -> dict[str, Any]:
    job_input = job.get("input", {})

    prompt = job_input.get("prompt", "").strip()
    if not prompt:
        return {"error": "Missing required input: prompt"}

    width = int(job_input.get("width", 1080))
    height = int(job_input.get("height", 1920))
    # Snap to multiples of 32
    width = max(64, (width // 32) * 32)
    height = max(64, (height // 32) * 32)

    seed = int(job_input.get("seed", 42))
    guidance_scale = float(job_input.get("guidance_scale", 4.0))
    num_steps = int(job_input.get("num_inference_steps", DEFAULT_STEPS))
    output_format = job_input.get("output_format", "png").upper()
    if output_format not in ("PNG", "JPEG"):
        output_format = "PNG"

    # Handle reference images (up to 10)
    input_images = []

    # Single image (backwards compatible with Flux 1 worker)
    image_input = job_input.get("image")
    if image_input:
        try:
            input_images.append(decode_image(image_input))
        except Exception as exc:
            return {"error": f"Failed to decode input image: {exc}"}

    # Multiple images
    image_inputs = job_input.get("images", [])
    for img_input in image_inputs:
        try:
            input_images.append(decode_image(img_input))
        except Exception as exc:
            return {"error": f"Failed to decode reference image: {exc}"}

    mode = f"i2i-{len(input_images)}ref" if input_images else "t2i"
    print(f"Generating {mode}: {width}x{height}, seed={seed}, guidance={guidance_scale}, steps={num_steps}")
    gen_start = time.time()

    try:
        generator = torch.Generator("cuda").manual_seed(seed)

        kwargs = {
            "prompt": prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_steps,
            "height": height,
            "width": width,
            "generator": generator,
            "max_sequence_length": 512,
        }

        # Pass reference images
        if len(input_images) == 1:
            kwargs["image"] = input_images[0]
        elif len(input_images) > 1:
            kwargs["image"] = input_images

        result = PIPELINE(**kwargs)
        output_image = result.images[0]
        image_base64 = encode_image(output_image, output_format)

        gen_elapsed = time.time() - gen_start
        print(f"Generation complete in {gen_elapsed:.1f}s")

        return {
            "image_base64": image_base64,
            "mode": mode,
            "width": output_image.width,
            "height": output_image.height,
            "generation_time_seconds": round(gen_elapsed, 1),
        }

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return {"error": str(exc)}


runpod.serverless.start({"handler": handler})
