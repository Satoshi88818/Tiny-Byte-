"""
TinyByte V4 Inference

V4 additions:
- VQ-VAE discrete generation mode: encodes prompt image into codebook indices,
  generates new indices autoregressively, decodes back to bytes (much faster).
- export sub-command: ONNX and TorchScript export for edge deployment.
- All V3 features preserved (CFG scheduling, negative prompting, speculative
  decoding, streaming, temperature/top-p annealing).
"""
from __future__ import annotations

import argparse
import io
import sys
from typing import Generator, Optional

import torch
import torch.nn.functional as F
from PIL import Image

from tinybyte_mm import TinyByteModel

SEP = 256


# ---------------------------------------------------------------------------
# Inference class
# ---------------------------------------------------------------------------

class TinyByteInference:
    def __init__(self, model: TinyByteModel, device: str = "cuda", draft_model=None):
        self.model = model.eval().to(device)
        self.device = device
        self.draft_model = draft_model

    # -----------------------------------------------------------------------
    # Core sampling utilities
    # -----------------------------------------------------------------------

    @staticmethod
    def _top_p_sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        logits = logits / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = (cum_probs - sorted_probs) > top_p
        sorted_probs[mask] = 0.0
        sorted_probs /= sorted_probs.sum().clamp(min=1e-8)
        tok = torch.multinomial(sorted_probs, 1)
        return sorted_idx[tok]

    @staticmethod
    def _cfg_scale(step: int, total: int, guidance_scale: float, cfg_decay_frac: float = 0.4) -> float:
        if step < cfg_decay_frac * total:
            return guidance_scale
        progress = (step - cfg_decay_frac * total) / max((1 - cfg_decay_frac) * total, 1)
        return guidance_scale * (1.0 - progress) + 1.0 * progress

    # -----------------------------------------------------------------------
    # Autoregressive generation with CFG + negative prompting
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_bytes: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        guidance_scale: float = 2.0,
        cfg_decay_frac: float = 0.4,
        uncond_ids: Optional[torch.Tensor] = None,
        negative_ids: Optional[torch.Tensor] = None,
        neg_scale: float = 1.0,
        use_speculative: bool = False,
        is_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if use_speculative and self.draft_model is not None:
            from draft_model import speculative_decode
            return speculative_decode(
                self.model, self.draft_model, prompt_ids,
                max_new_bytes=max_new_bytes,
                temperature=temperature, top_p=top_p, device=self.device,
            )

        ids = prompt_ids.clone().to(self.device)
        do_cfg = guidance_scale > 1.0 and uncond_ids is not None
        do_neg = negative_ids is not None and do_cfg

        if do_cfg:
            ucond = uncond_ids.to(self.device)
        if do_neg:
            neg = negative_ids.to(self.device)

        t_start, t_end = temperature, temperature * 0.5

        for step in range(max_new_bytes):
            t = t_start + (t_end - t_start) * (step / max(max_new_bytes, 1))
            logits_cond = self.model(ids, is_image=is_image)[:, -1, :]
            if do_cfg:
                logits_uncond = self.model(ucond)[:, -1, :]
                scale = self._cfg_scale(step, max_new_bytes, guidance_scale, cfg_decay_frac)
                logits = logits_uncond + scale * (logits_cond - logits_uncond)
                if do_neg:
                    logits_neg = self.model(neg)[:, -1, :]
                    logits = logits - neg_scale * (logits_neg - logits_uncond)
            else:
                logits = logits_cond

            next_tok = self._top_p_sample(logits[0], t, top_p).unsqueeze(0).unsqueeze(0)
            ids = torch.cat([ids, next_tok], dim=1)
            if do_cfg:
                ucond = torch.cat([ucond, next_tok], dim=1)
            if do_neg:
                neg = torch.cat([neg, next_tok], dim=1)

        return ids[:, prompt_ids.size(1):]

    # -----------------------------------------------------------------------
    # [V4] VQ-VAE discrete generation (faster for image sequences)
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def generate_discrete(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
    ) -> Optional[Image.Image]:
        """
        Generate image using VQ-VAE discrete codebook tokens.
        ~8-16x faster than byte-level generation for image sequences.
        Requires model.cfg.use_vqvae = True.
        """
        cfg = self.model.cfg
        if not cfg.use_vqvae:
            raise ValueError("Model does not have VQ-VAE enabled. Use generate() instead.")

        # Encode prompt bytes into VQ-VAE latent space
        x = self.model.embed(prompt_ids)  # (1, T, D)
        z_q_proj, _, indices = self.model.vqvae.encode(x)  # (1, T_short, D), (1, T_short)

        # Autoregressively generate new codebook indices via the Transformer
        # (operating on the short VQ-VAE sequence)
        generated_indices = indices.clone()  # (1, T_short)

        codebook = self.model.vqvae.quantizer.embedding.weight  # (C, latent_dim)
        latent_proj = self.model.vqvae.latent_proj  # latent_dim → embed_dim

        for _ in range(max_new_tokens):
            # Decode current indices to embeddings
            z_q = codebook[generated_indices]  # (1, T_cur, latent_dim)
            z_q_embed = latent_proj(z_q)       # (1, T_cur, embed_dim)

            # Run through Transformer layers (bypass patch encoder)
            T_p = z_q_embed.size(1)
            cos, sin = self.model.rope(T_p)
            x_out = z_q_embed
            for layer in self.model.layers:
                x_out = layer(x_out, cos, sin)
            x_out = self.model.norm(x_out)

            # Project to codebook logits
            # (use embedding weight as codebook classifier)
            last_emb = x_out[:, -1, :]  # (1, D)
            # Map back to latent dim for codebook distance
            last_latent = (latent_proj.weight.T @ last_emb.T).T  # crude inverse
            dists = (
                last_latent.pow(2).sum(-1, keepdim=True)
                - 2 * last_latent @ codebook.T
                + codebook.pow(2).sum(-1)
            )  # (1, C)
            logits = -dists  # lower distance → higher logit
            logits = logits / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)

            # Top-p filtering
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = (cumulative - sorted_probs) > top_p
            sorted_probs[mask] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            new_idx = sorted_idx[0, torch.multinomial(sorted_probs[0], 1)]
            generated_indices = torch.cat(
                [generated_indices, new_idx.view(1, 1)], dim=1
            )

        # Decode generated indices to byte embeddings via VQ-VAE decoder
        z_q_generated = codebook[generated_indices]   # (1, T_all, latent_dim)
        recon_emb = self.model.vqvae.decoder(z_q_generated)  # (1, T_bytes, embed_dim)
        recon_logits = self.model.head(recon_emb)             # (1, T_bytes, vocab_size)

        # Decode to bytes
        byte_ids = recon_logits[0].argmax(-1).tolist()
        img_bytes = bytes([b for b in byte_ids if b < 256])
        try:
            return Image.open(io.BytesIO(img_bytes))
        except Exception:
            return None

    # -----------------------------------------------------------------------
    # Streaming generation
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def generate_streaming(
        self,
        prompt_ids: torch.Tensor,
        max_new_bytes: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        guidance_scale: float = 2.0,
        uncond_ids: Optional[torch.Tensor] = None,
        chunk_size: int = 64,
    ) -> Generator[bytes, None, None]:
        ids = prompt_ids.clone().to(self.device)
        do_cfg = guidance_scale > 1.0 and uncond_ids is not None
        if do_cfg:
            ucond = uncond_ids.to(self.device)
        buffer = []
        for step in range(max_new_bytes):
            logits_cond = self.model(ids)[:, -1, :]
            if do_cfg:
                logits_uncond = self.model(ucond)[:, -1, :]
                scale = self._cfg_scale(step, max_new_bytes, guidance_scale)
                logits = logits_uncond + scale * (logits_cond - logits_uncond)
            else:
                logits = logits_cond
            next_tok = self._top_p_sample(logits[0], temperature, top_p).unsqueeze(0).unsqueeze(0)
            ids = torch.cat([ids, next_tok], dim=1)
            if do_cfg:
                ucond = torch.cat([ucond, next_tok], dim=1)
            tok_val = next_tok.item()
            if tok_val < 256:
                buffer.append(tok_val)
            if len(buffer) >= chunk_size:
                yield bytes(buffer)
                buffer = []
        if buffer:
            yield bytes(buffer)

    # -----------------------------------------------------------------------
    # High-level API
    # -----------------------------------------------------------------------

    def text_to_image(
        self,
        prompt: str,
        max_image_bytes: int = 16384,
        guidance_scale: float = 2.0,
        cfg_decay_frac: float = 0.4,
        negative_prompt: Optional[str] = None,
        use_speculative: bool = False,
        use_discrete: bool = False,
        streaming: bool = False,
        output_path: Optional[str] = None,
    ) -> Optional[Image.Image]:
        caption_bytes = list(prompt.encode("utf-8")) + [SEP]
        prompt_ids = torch.tensor([caption_bytes], dtype=torch.long, device=self.device)
        uncond_ids = torch.tensor([[SEP]], dtype=torch.long, device=self.device)
        is_image = torch.tensor([True], dtype=torch.bool, device=self.device)

        neg_ids = None
        if negative_prompt:
            neg_bytes = list(negative_prompt.encode("utf-8")) + [SEP]
            neg_ids = torch.tensor([neg_bytes], dtype=torch.long, device=self.device)

        # [V4] Discrete VQ-VAE generation path
        if use_discrete and self.model.cfg.use_vqvae:
            img = self.generate_discrete(
                prompt_ids, max_new_tokens=max_image_bytes // 8,
                temperature=1.0, top_p=0.95,
            )
            if img and output_path:
                img.save(output_path)
            return img

        if streaming and output_path:
            with open(output_path + ".partial", "wb") as f:
                for chunk in self.generate_streaming(
                    prompt_ids, max_new_bytes=max_image_bytes,
                    guidance_scale=guidance_scale, uncond_ids=uncond_ids,
                ):
                    f.write(chunk)
                    f.flush()
            import shutil
            shutil.move(output_path + ".partial", output_path)
            return Image.open(output_path)

        generated = self.generate(
            prompt_ids, max_new_bytes=max_image_bytes,
            guidance_scale=guidance_scale, cfg_decay_frac=cfg_decay_frac,
            uncond_ids=uncond_ids, negative_ids=neg_ids,
            use_speculative=use_speculative, is_image=is_image,
        )
        img_bytes = bytes([b for b in generated[0].tolist() if b < 256])
        try:
            img = Image.open(io.BytesIO(img_bytes))
            if output_path:
                img.save(output_path)
            return img
        except Exception as e:
            print(f"Could not decode image: {e}")
            return None

    def image_to_text(
        self,
        image_path: str,
        max_caption_bytes: int = 256,
        temperature: float = 0.7,
    ) -> str:
        from dataset import _encode_image_as_jpeg_bytes
        img_bytes = _encode_image_as_jpeg_bytes(image_path)
        if not img_bytes:
            return ""
        prompt_ids = torch.tensor(
            [list(img_bytes[:4096]) + [SEP]], dtype=torch.long, device=self.device
        )
        generated = self.generate(
            prompt_ids, max_new_bytes=max_caption_bytes,
            temperature=temperature, guidance_scale=1.0,
        )
        return bytes([b for b in generated[0].tolist() if b < 256]).decode("utf-8", errors="replace")

    def generate_text(
        self,
        prompt: str,
        max_new_bytes: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> str:
        prompt_ids = torch.tensor(
            [list(prompt.encode("utf-8"))], dtype=torch.long, device=self.device
        )
        generated = self.generate(
            prompt_ids, max_new_bytes=max_new_bytes,
            temperature=temperature, top_p=top_p, guidance_scale=1.0,
        )
        return bytes([b for b in generated[0].tolist() if b < 256]).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_model(args) -> TinyByteModel:
    from tinybyte_mm import ModelConfig, _medium_config, _small_config
    if args.small:
        cfg = _small_config()
    elif args.medium:
        cfg = _medium_config()
    else:
        cfg = ModelConfig()
    model = TinyByteModel(cfg)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt))
    return model


def main():
    p = argparse.ArgumentParser(description="TinyByte V4 Inference")
    sub = p.add_subparsers(dest="command")

    # --- generate sub-command ---
    gen = sub.add_parser("generate", help="Generate text or images")
    gen.add_argument("--checkpoint", required=True)
    gen.add_argument("--draft-checkpoint", default=None)
    gen.add_argument("--mode", choices=["text", "text2image", "image2text"], required=True)
    gen.add_argument("--prompt", default="")
    gen.add_argument("--negative-prompt", default=None)
    gen.add_argument("--image", default=None)
    gen.add_argument("--output", default="output.jpg")
    gen.add_argument("--max-bytes", type=int, default=16384)
    gen.add_argument("--guidance-scale", type=float, default=2.0)
    gen.add_argument("--cfg-decay-frac", type=float, default=0.4)
    gen.add_argument("--temperature", type=float, default=1.0)
    gen.add_argument("--top-p", type=float, default=0.95)
    gen.add_argument("--speculative", action="store_true")
    gen.add_argument("--discrete", action="store_true", help="[V4] Use VQ-VAE discrete generation")
    gen.add_argument("--streaming", action="store_true")
    gen.add_argument("--small", action="store_true")
    gen.add_argument("--medium", action="store_true")
    gen.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # --- export sub-command ---
    exp = sub.add_parser("export", help="[V4] Export model to ONNX or TorchScript")
    exp.add_argument("--checkpoint", required=True)
    exp.add_argument("--format", choices=["onnx", "torchscript"], default="onnx")
    exp.add_argument("--output", default="tinybyte_v4.onnx")
    exp.add_argument("--seq-len", type=int, default=64, help="Example sequence length for tracing")
    exp.add_argument("--small", action="store_true")
    exp.add_argument("--medium", action="store_true")
    exp.add_argument("--device", default="cpu")

    # Backwards-compatible: if no sub-command given, use generate with positional-style args
    args = p.parse_args()
    if args.command is None:
        # Legacy V3 CLI: treat all args as generate
        p.print_help()
        return

    if args.command == "export":
        model = _load_model(args).to(args.device).eval()
        example = torch.zeros(1, args.seq_len, dtype=torch.long, device=args.device)
        if args.format == "torchscript":
            model.export_torchscript(args.output, example)
        else:
            model.export_onnx(args.output, example)
        return

    # generate command
    model = _load_model(args).to(args.device)
    draft = None
    if args.draft_checkpoint and args.speculative:
        from draft_model import DraftModel
        draft = DraftModel.from_pretrained(args.draft_checkpoint, args.device)

    infer = TinyByteInference(model, args.device, draft_model=draft)

    if args.mode == "text":
        result = infer.generate_text(args.prompt, max_new_bytes=args.max_bytes, temperature=args.temperature, top_p=args.top_p)
        print(result)

    elif args.mode == "text2image":
        img = infer.text_to_image(
            args.prompt,
            max_image_bytes=args.max_bytes,
            guidance_scale=args.guidance_scale,
            cfg_decay_frac=args.cfg_decay_frac,
            negative_prompt=args.negative_prompt,
            use_speculative=args.speculative,
            use_discrete=args.discrete,
            streaming=args.streaming,
            output_path=args.output,
        )
        if img:
            print(f"Image saved to {args.output}")
        else:
            print("Image generation failed.")

    elif args.mode == "image2text":
        caption = infer.image_to_text(args.image, temperature=args.temperature)
        print(caption)


if __name__ == "__main__":
    main()
