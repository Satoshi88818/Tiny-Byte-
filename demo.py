"""
TinyByte V4 — Gradio Web Demo
Run: python demo.py --checkpoint ./checkpoints/tinybyte_v4_epoch9.pt

Provides a browser UI for:
  - Text → Image generation (byte-level and discrete VQ-VAE modes)
  - Image → Text captioning
  - Text generation / completion
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Load model on startup
# ---------------------------------------------------------------------------

_infer = None


def _get_infer(checkpoint: str, device: str, small: bool = False, medium: bool = False):
    global _infer
    if _infer is not None:
        return _infer
    from tinybyte_mm import ModelConfig, TinyByteModel, _medium_config, _small_config
    from infer import TinyByteInference

    if small:
        cfg = _small_config()
    elif medium:
        cfg = _medium_config()
    else:
        cfg = ModelConfig()

    model = TinyByteModel(cfg)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt))
    _infer = TinyByteInference(model, device)
    return _infer


# ---------------------------------------------------------------------------
# Gradio tab functions
# ---------------------------------------------------------------------------

def text_to_image_fn(
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
    cfg_decay_frac: float,
    max_bytes: int,
    temperature: float,
    top_p: float,
    use_discrete: bool,
):
    infer = _get_infer.__wrapped__ if hasattr(_get_infer, "__wrapped__") else _infer
    if infer is None:
        return None, "Model not loaded."
    try:
        img = infer.text_to_image(
            prompt,
            max_image_bytes=max_bytes,
            guidance_scale=guidance_scale,
            cfg_decay_frac=cfg_decay_frac,
            negative_prompt=negative_prompt or None,
            use_discrete=use_discrete,
        )
        if img is None:
            return None, "Generation failed — could not decode JPEG bytes."
        return img, f"Generated {max_bytes} bytes → decoded image {img.size}"
    except Exception as e:
        return None, f"Error: {e}"


def image_to_text_fn(image: Image.Image, temperature: float, max_caption: int):
    infer = _infer
    if infer is None:
        return "Model not loaded."
    if image is None:
        return "Please upload an image."
    try:
        # Save PIL image to temp file for infer.image_to_text
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        buf.seek(0)
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(buf.getvalue())
            tmp_path = tmp.name
        caption = infer.image_to_text(tmp_path, max_caption_bytes=max_caption, temperature=temperature)
        os.unlink(tmp_path)
        return caption
    except Exception as e:
        return f"Error: {e}"


def text_generate_fn(prompt: str, max_bytes: int, temperature: float, top_p: float):
    infer = _infer
    if infer is None:
        return "Model not loaded."
    try:
        return infer.generate_text(prompt, max_new_bytes=max_bytes, temperature=temperature, top_p=top_p)
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Build Gradio interface
# ---------------------------------------------------------------------------

def build_demo():
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Run: pip install gradio")
        sys.exit(1)

    with gr.Blocks(title="TinyByte V4 — Tokenizer-free Multimodal Model") as demo:
        gr.Markdown(
            """
# 🔡 TinyByte Multimodal V4
**Tokenizer-free · Byte-level · Causal Transformer · Text ↔ Image**

V4 features: VQ-VAE discrete image pathway · Mixture-of-Depth · SwiGLU · RMSNorm · Co-distilled speculative decoding
            """
        )

        with gr.Tab("Text → Image"):
            with gr.Row():
                with gr.Column():
                    t2i_prompt = gr.Textbox(label="Prompt", placeholder="A red apple on a wooden table, photorealistic", lines=2)
                    t2i_neg = gr.Textbox(label="Negative Prompt", placeholder="blurry, low quality, text", lines=1)
                    with gr.Row():
                        t2i_guidance = gr.Slider(1.0, 10.0, value=2.5, step=0.1, label="Guidance Scale")
                        t2i_decay = gr.Slider(0.1, 1.0, value=0.4, step=0.05, label="CFG Decay Fraction")
                    with gr.Row():
                        t2i_max = gr.Slider(1024, 32768, value=16384, step=1024, label="Max Bytes")
                        t2i_temp = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="Temperature")
                    with gr.Row():
                        t2i_topp = gr.Slider(0.5, 1.0, value=0.95, step=0.01, label="Top-p")
                        t2i_discrete = gr.Checkbox(label="[V4] Discrete VQ-VAE mode (faster)", value=False)
                    t2i_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    t2i_image = gr.Image(label="Generated Image", type="pil")
                    t2i_info = gr.Textbox(label="Info", interactive=False)
            t2i_btn.click(
                text_to_image_fn,
                inputs=[t2i_prompt, t2i_neg, t2i_guidance, t2i_decay, t2i_max, t2i_temp, t2i_topp, t2i_discrete],
                outputs=[t2i_image, t2i_info],
            )

        with gr.Tab("Image → Text"):
            with gr.Row():
                with gr.Column():
                    i2t_image = gr.Image(label="Input Image", type="pil")
                    with gr.Row():
                        i2t_temp = gr.Slider(0.1, 2.0, value=0.7, step=0.05, label="Temperature")
                        i2t_max = gr.Slider(32, 512, value=256, step=32, label="Max Caption Bytes")
                    i2t_btn = gr.Button("Caption", variant="primary")
                with gr.Column():
                    i2t_caption = gr.Textbox(label="Caption", lines=4, interactive=False)
            i2t_btn.click(
                image_to_text_fn,
                inputs=[i2t_image, i2t_temp, i2t_max],
                outputs=i2t_caption,
            )

        with gr.Tab("Text Generation"):
            with gr.Row():
                with gr.Column():
                    tg_prompt = gr.Textbox(label="Prompt", placeholder="Once upon a time", lines=3)
                    with gr.Row():
                        tg_max = gr.Slider(64, 2048, value=512, step=64, label="Max New Bytes")
                        tg_temp = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature")
                    tg_topp = gr.Slider(0.5, 1.0, value=0.95, step=0.01, label="Top-p")
                    tg_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    tg_output = gr.Textbox(label="Completion", lines=10, interactive=False)
            tg_btn.click(
                text_generate_fn,
                inputs=[tg_prompt, tg_max, tg_temp, tg_topp],
                outputs=tg_output,
            )

        with gr.Tab("Model Info"):
            gr.Markdown(
                """
## Architecture
- **Byte Embedding**: nn.Embedding(257, D) — 0-255 bytes + PAD
- **Learned Boundary Predictor**: 2-layer Conv1D → soft-argmax patches (+ entropy signal for MoD)
- **[V4] VQ-VAE**: CNN encoder → EMA vector quantizer (4096 codebook) → 8× shorter sequences for images
- **[V4] Mixture-of-Depth (MoD)**: Skip FFN for low-entropy patches in the deeper half of layers
- **Transformer**: N × GQA + SwiGLU FFN + RMSNorm + RoPE (causal)
- **OverlapAddDecoder**: Per-patch MLP window → fold-based overlap-add
- **Head**: Linear(D, 257) with weight tying

## V4 Changes from V3
| V3 | V4 |
|---|---|
| No discrete VAE | VQ-VAE pathway: 8-16× shorter image sequences |
| All layers process all tokens | Mixture-of-Depth: skip FFN for flat patches |
| GELU FFN | SwiGLU FFN (~10% quality improvement) |
| LayerNorm | RMSNorm (~15% faster) |
| Draft model trained separately | Co-distillation: draft trained jointly |
| No export support | ONNX + TorchScript export CLI |
| No web demo | This Gradio demo |
                """
            )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="TinyByte V4 Gradio Demo")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--small", action="store_true")
    p.add_argument("--medium", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = p.parse_args()

    # Load model before building demo
    print(f"Loading model from {args.checkpoint} on {args.device}...")
    global _infer
    _infer = _get_infer(args.checkpoint, args.device, args.small, args.medium)
    print("Model loaded.")

    demo = build_demo()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
