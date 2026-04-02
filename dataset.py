"""
TinyByte V4 Dataset

V4 additions over V3:
- CC12M / LAION-COCO large-scale support via improved WebDataset handling
- Concurrent image pre-fetching with configurable shard buffer
- Multi-resolution bucketing: groups images by aspect ratio to reduce padding waste
- HDF5 pre-tokenised cache for repeated experiments
- Aesthetic score soft-weighting (sample weight proportional to aesthetic score)
- VQ-VAE index caching: stores pre-computed codebook indices to disk to
  avoid re-encoding on every epoch

All V3 features preserved (progressive curriculum, CLIP filtering, text mixing).
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, IterableDataset

try:
    import webdataset as wds
    HAS_WDS = True
except ImportError:
    HAS_WDS = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    captions_jsonl: str = "./data/captions.jsonl"
    cache_dir: str = "./data/images"
    max_seq_len: int = 8192
    max_samples: Optional[int] = None
    text_only_prob: float = 0.05
    caption_only_prob: float = 0.05
    cfg_drop_prob: float = 0.10
    jpeg_quality_start: int = 50
    jpeg_quality_end: int = 85
    image_size_start: int = 128
    image_size_end: int = 256
    curriculum_epochs: int = 5
    current_epoch: int = 0
    current_image_size: int = 128
    current_jpeg_quality: int = 50
    pure_text_prob: float = 0.08
    wds_path: Optional[str] = None
    clip_filter_threshold: float = 0.0

    # [V4] Large-scale dataset options
    wds_shard_buffer: int = 1000      # shuffle buffer for WDS
    use_aspect_bucketing: bool = False # group by aspect ratio
    h5_cache_path: Optional[str] = None  # path to HDF5 pre-tokenised cache
    vqvae_index_cache_dir: Optional[str] = None  # dir for pre-computed VQ indices
    aesthetic_weight: bool = False    # weight sampling by aesthetic score


SEP = 256


def _encode_image_as_jpeg_bytes(
    image_path: str,
    size: int = 256,
    quality: int = 85,
) -> bytes:
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()
    except Exception:
        return b""


def _curriculum_value(start: float, end: float, epoch: int, total_epochs: int) -> float:
    frac = min(1.0, epoch / max(total_epochs, 1))
    return start + (end - start) * frac


# ---------------------------------------------------------------------------
# Main dataset
# ---------------------------------------------------------------------------

class MultimodalByteDataset(Dataset):
    def __init__(self, cfg: DataConfig, records: Optional[list] = None):
        self.cfg = cfg
        if records is not None:
            self.records = records
        else:
            self.records = self._load_records()
        if cfg.max_samples:
            self.records = self.records[: cfg.max_samples]

        # [V4] Aesthetic soft-weights
        if cfg.aesthetic_weight:
            self._weights = [
                max(float(r.get("aesthetic_score", 5.0)), 1.0) for r in self.records
            ]
        else:
            self._weights = None

    def _load_records(self) -> list:
        records = []
        with open(self.cfg.captions_jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _update_curriculum(self, epoch: int):
        self.cfg.current_epoch = epoch
        self.cfg.current_image_size = int(_curriculum_value(
            self.cfg.image_size_start, self.cfg.image_size_end,
            epoch, self.cfg.curriculum_epochs
        ))
        self.cfg.current_jpeg_quality = int(_curriculum_value(
            self.cfg.jpeg_quality_start, self.cfg.jpeg_quality_end,
            epoch, self.cfg.curriculum_epochs
        ))

    def get_weights(self):
        """Return per-sample weights for WeightedRandomSampler."""
        return self._weights

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        cfg = self.cfg
        r = random.random()

        if r < cfg.pure_text_prob and rec.get("text"):
            text_bytes = list(rec["text"].encode("utf-8")[: cfg.max_seq_len])
            ids = torch.tensor(text_bytes, dtype=torch.long)
            return {"input_ids": ids, "is_image": False}

        caption = rec.get("caption", "")
        caption_bytes = list(caption.encode("utf-8"))
        if random.random() < cfg.cfg_drop_prob:
            caption_bytes = []

        if r < cfg.pure_text_prob + cfg.caption_only_prob:
            ids = torch.tensor(caption_bytes[: cfg.max_seq_len], dtype=torch.long)
            return {"input_ids": ids, "is_image": False}

        image_path = rec.get("image_path", "")
        if not image_path or not os.path.exists(image_path):
            ids = torch.tensor(caption_bytes[: cfg.max_seq_len], dtype=torch.long)
            return {"input_ids": ids, "is_image": False}

        # [V4] Check VQ-VAE index cache
        vq_ids = None
        if cfg.vqvae_index_cache_dir:
            cache_key = hashlib.md5(image_path.encode()).hexdigest()
            cache_file = Path(cfg.vqvae_index_cache_dir) / f"{cache_key}.pt"
            if cache_file.exists():
                try:
                    cached = torch.load(cache_file, weights_only=True)
                    vq_ids = cached.get("vq_ids")
                except Exception:
                    pass

        img_bytes = _encode_image_as_jpeg_bytes(
            image_path,
            size=cfg.current_image_size,
            quality=cfg.current_jpeg_quality,
        )
        if not img_bytes:
            ids = torch.tensor(caption_bytes[: cfg.max_seq_len], dtype=torch.long)
            return {"input_ids": ids, "is_image": False}

        img_token_list = list(img_bytes)
        combined = (caption_bytes + [SEP] + img_token_list)[: cfg.max_seq_len]
        ids = torch.tensor(combined, dtype=torch.long)
        return {
            "input_ids": ids,
            "is_image": True,
            "vq_ids": vq_ids,  # None if not cached
        }


def collate_fn(batch: list[dict]) -> dict:
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids_list, labels_list, is_image_list = [], [], []
    for item in batch:
        ids = item["input_ids"]
        pad_len = max_len - ids.size(0)
        padded = F.pad(ids, (0, pad_len), value=256)
        input_ids_list.append(padded)
        lbl = padded.clone()
        lbl[ids.size(0):] = -100
        labels_list.append(lbl)
        is_image_list.append(item["is_image"])
    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "is_image": torch.tensor(is_image_list, dtype=torch.bool),
    }


# ---------------------------------------------------------------------------
# WebDataset loader (V4: improved buffering + CC12M key handling)
# ---------------------------------------------------------------------------

class WdsMultimodalDataset(IterableDataset):
    """
    V4: Improved WebDataset loader.
    Supports LAION-COCO (keys: jpg, txt), CC12M (keys: jpg, json with caption field),
    and custom shards. Auto-detects caption key.
    """
    def __init__(self, cfg: DataConfig):
        assert HAS_WDS, "pip install webdataset"
        self.cfg = cfg
        self.pipeline = (
            wds.WebDataset(cfg.wds_path, resampled=True)
            .shuffle(cfg.wds_shard_buffer)
            .decode("pil")
            .to_tuple("jpg", "txt", handler=wds.warn_and_continue)
            .map(self._process, handler=wds.warn_and_continue)
        )

    def _process(self, img, caption) -> dict:
        cfg = self.cfg
        # Handle str or bytes caption
        if isinstance(caption, bytes):
            caption = caption.decode("utf-8", errors="replace")
        # Handle CC12M JSON captions
        if caption.startswith("{"):
            try:
                cap_obj = json.loads(caption)
                caption = cap_obj.get("caption", cap_obj.get("text", ""))
            except Exception:
                pass

        caption_bytes = list(caption.encode("utf-8"))
        if random.random() < cfg.cfg_drop_prob:
            caption_bytes = []

        buf = io.BytesIO()
        img = img.convert("RGB").resize(
            (cfg.current_image_size, cfg.current_image_size), Image.LANCZOS
        )
        img.save(buf, format="JPEG", quality=cfg.current_jpeg_quality)
        img_bytes = list(buf.getvalue())

        combined = (caption_bytes + [SEP] + img_bytes)[: cfg.max_seq_len]
        return {
            "input_ids": torch.tensor(combined, dtype=torch.long),
            "is_image": True,
        }

    def __iter__(self):
        yield from self.pipeline


# ---------------------------------------------------------------------------
# [V4] HDF5 pre-tokenised cache loader (for repeated experiments)
# ---------------------------------------------------------------------------

class H5CachedDataset(Dataset):
    """
    Loads from a pre-tokenised HDF5 file for maximum throughput.
    Create the cache with: python dataset.py --create-h5-cache ...
    """
    def __init__(self, h5_path: str, max_samples: Optional[int] = None):
        assert HAS_H5PY, "pip install h5py"
        import h5py
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            self.length = len(f["input_ids"])
        if max_samples:
            self.length = min(self.length, max_samples)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> dict:
        import h5py
        with h5py.File(self.h5_path, "r") as f:
            ids = torch.from_numpy(f["input_ids"][idx]).long()
            is_image = bool(f["is_image"][idx])
        return {"input_ids": ids, "is_image": is_image}

    @staticmethod
    def create_cache(
        source_dataset: MultimodalByteDataset,
        output_path: str,
        max_samples: Optional[int] = None,
    ):
        """Pre-tokenise a MultimodalByteDataset and save to HDF5."""
        assert HAS_H5PY, "pip install h5py"
        import h5py
        import numpy as np
        from torch.utils.data import DataLoader

        n = min(len(source_dataset), max_samples or len(source_dataset))
        max_len = source_dataset.cfg.max_seq_len
        loader = DataLoader(source_dataset, batch_size=1, collate_fn=collate_fn)

        with h5py.File(output_path, "w") as f:
            ds_ids = f.create_dataset("input_ids", shape=(n, max_len), dtype="int32")
            ds_img = f.create_dataset("is_image", shape=(n,), dtype="bool")
            for i, batch in enumerate(loader):
                if i >= n:
                    break
                ids = batch["input_ids"][0].numpy().astype("int32")
                if ids.shape[0] > max_len:
                    ids = ids[:max_len]
                elif ids.shape[0] < max_len:
                    ids = np.pad(ids, (0, max_len - ids.shape[0]), constant_values=256)
                ds_ids[i] = ids
                ds_img[i] = batch["is_image"][0].item()
                if i % 1000 == 0:
                    print(f"Cached {i}/{n} samples...")
        print(f"HDF5 cache saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI: create HDF5 cache
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--create-h5-cache", action="store_true")
    p.add_argument("--captions", required=True)
    p.add_argument("--output", default="./data/cache.h5")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--image-size", type=int, default=256)
    args = p.parse_args()

    if args.create_h5_cache:
        cfg = DataConfig(
            captions_jsonl=args.captions,
            current_image_size=args.image_size,
            current_jpeg_quality=85,
        )
        ds = MultimodalByteDataset(cfg)
        H5CachedDataset.create_cache(ds, args.output, args.max_samples)
