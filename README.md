
TinyByte Multimodal V4

**Tokenizer-free byte-level causal Transformer for text and image generation.**

A compact, efficient multimodal foundation model that processes **raw bytes** directly — no BPE tokenizer, no modality-specific encoders. TinyByte V4 unifies text and JPEG image generation in a single autoregressive sequence, using learned patching, discrete VAE compression for images, and Mixture-of-Depth for compute efficiency.

Built with first-principles reasoning: bytes are the universal, lossless representation of data. Why force artificial vocabularies when the model can learn natural boundaries and allocate compute intelligently?

## Key Features (V4)

- **True byte-level modeling** — Vocabulary of 257 (0-255 bytes + PAD/SEP)
- **Learned boundary predictor** — 1D CNN discovers semantic patches from entropy (reused for routing)
- **VQ-VAE pathway for images** — 8× shorter discrete latent sequences for vision (codebook size 4096)
- **Mixture-of-Depth (MoD)** — Skip expensive FFN layers on low-entropy (smooth) patches; full compute on complex regions
- **Modern backbone upgrades**:
  - SwiGLU FFN (better gradient flow)
  - RMSNorm (faster, no mean subtraction)
  - Grouped-Query Attention (GQA)
  - Rotary embeddings (RoPE) + byte-level positional offsets within patches
- **OverlapAddDecoder** — Reconstructs variable-length byte streams from fixed patch outputs
- **Multimodal curriculum** — Progressive image size & JPEG quality ramp-up
- **JPEG header auxiliary loss** — Stronger supervision on critical image structure
- **Co-distillation** — Joint training of a lightweight draft model for high-quality speculative decoding
- **Production-ready** — FSDP, bf16, torch.compile, WandB logging, HDF5 caching, WebDataset support, HF-style save/load, TorchScript/ONNX export

## Model Configurations

| Config   | Embed Dim | Layers | Heads (GQA KV) | Params (approx) | VQ-VAE | MoD | Use Case                  |
|----------|-----------|--------|----------------|-----------------|--------|-----|---------------------------|
| Small    | 256       | 6      | 4 (4)          | ~30M            | No     | No  | Prototyping / Edge       |
| Medium   | 512       | 12     | 8 (4)          | ~150M           | Yes    | Yes | Research / Fine-tuning   |
| Default  | 1024      | 24     | 16 (4)         | ~600M+          | Yes    | Yes | Full-scale multimodal    |

## Architecture Highlights

- **EntropyPatchEncoder**: Fixed stride fallback or learned boundaries via soft segmentation.
- **TransformerLayer with MoD**: Attention always runs; FFN routed only to high-entropy tokens (capacity annealed from 1.0 during warmup).
- **VQ-VAE**: Lightweight CNN encoder → Vector Quantizer (EMA updates) → latent projection. Auxiliary byte reconstruction loss keeps the pathway grounded.
- **Speculative Decoding**: Draft model (6L/256D) trained via co-distillation (hard CE + softened KL) for high acceptance rates.

## Training

See `train.py`. Supports:
- Mixed text + image batches (`is_image` flag)
- WebDataset (LAION-COCO, CC12M style) or JSONL + local images
- HDF5 pre-tokenized cache for fast iteration
- Aesthetic score weighting
- MoD capacity annealing
- Co-distillation of draft model (`--draft-codistill`)

Example command (medium config):
```bash
python train.py \
  --medium \
  --batch-size 4 \
  --grad-accum 4 \
  --epochs 10 \
  --lr 3e-4 \
  --bf16 \
  --wandb \
  --draft-codistill \
  --wds-path ./data/shards
```

Monitor: `main_loss`, `vq_loss`, `aux_recon_loss`, `trim_frac` (high trim = excessive padding — adjust curriculum or max_seq_len).

## Inference & Speculative Decoding

```python
from tinybyte_mm import TinyByteModel
from draft_model import DraftModel, speculative_decode

model = TinyByteModel.from_pretrained("checkpoints/tinybyte_v4_epochX")
draft = DraftModel.from_pretrained("checkpoints/draft_v4_epochX")

prompt = torch.tensor(list("A beautiful photo of ".encode("utf-8")), dtype=torch.long).unsqueeze(0)

generated = speculative_decode(
    model, draft, prompt,
    max_new_bytes=2048,
    draft_steps=6,
    temperature=0.9,
    top_p=0.95,
    verbose=True
)

print(bytes(generated[0].tolist()).decode("utf-8", errors="replace"))
```

The draft model dramatically accelerates generation while maintaining quality thanks to co-distillation.

## Dataset Preparation

- `dataset.py` handles JPEG byte encoding + caption mixing + SEP token.
- Curriculum automatically ramps image resolution and JPEG quality.
- Supports pure-text, caption-only, and full multimodal samples.
- Optional VQ-VAE index caching and HDF5 pre-tokenization for speed.

## Saving & Export

Models save in Hugging Face-compatible format (`model.pt` + `config.json` + README.md).

```python
model.save_pretrained("my_tinybyte_v4")
draft_model.save_pretrained("my_draft_v4")

# Export for edge
model.export_onnx("tinybyte.onnx")
model.export_torchscript("tinybyte.ts")
```

## Why TinyByte?

Most multimodal models bolt vision encoders onto tokenized LLMs. TinyByte starts from first principles:

- **Universality**: One architecture for text, images (JPEG bytes), and future modalities (audio, binary, etc.).
- **Efficiency**: Learned patches + VQ-VAE + MoD reduce effective sequence length and FLOPs on uniform regions.
- **Robustness**: No tokenizer artifacts, no out-of-vocab issues, better handling of noisy or mixed-format data.
- **Simplicity**: No separate vision tower, no cross-attention hacks — just bytes in, bytes out.


## Roadmap / Future Ideas

- Hierarchical VQ or multi-scale patching for higher-resolution images
- Full CFG scheduling + negative prompting
- Video/audio byte support
- Larger scaling experiments (1B+ params)
- True variable-length dynamic patching beyond current center prediction

## License

MIT 



---

**Built as a minimal yet powerful foundation for the tokenizer-free future.**

Start small, scale with patches, generate with bytes.
```

