#!/usr/bin/env python3
"""
Real-world fine-tuning for the PARSeq+CTC plate model, with synthetic replay.

Goal: teach the model real texture, sensor noise, glare and true motion blur —
things a synthetic generator cannot produce — WITHOUT losing what the synthetic
set taught it (charset, 1/2/3-line layouts, glyph shapes).

THE ONE IDEA THAT MATTERS: REPLAY
---------------------------------
We never train on real-only, not even for a few hundred steps. Fine-tuning a
converged model on a narrow new distribution is the textbook recipe for
catastrophic forgetting. Instead every batch is a blend: each sample is drawn
from the real pool with probability `p_real`, otherwise from the original
synthetic set. The old distribution stays in the gradient, so forgetting is not
a risk to monitor — it is structurally prevented.

Within the real half, sources are sampled by weight (and CCPD is balanced
equally across its subsets), so 200k CCPD images cannot drown out 2k Indian
ones.

WHAT ELSE IS DIFFERENT FROM NORMAL TRAINING
-------------------------------------------
* LR is 1/10 of the original peak, with a warmup. The first few dozen steps at
  full LR are where the damage would happen.
* Nothing is frozen. Texture and blur are LOW-level statistics learned in the
  conv stem — freezing the stem would freeze the exact thing we want to adapt.
* Degradation augmentation is halved for real images. They already contain the
  real version of blur/noise/jpeg; stacking synthetic degradation on top pushes
  them back off-distribution. Geometric augmentation stays at full strength.
* Two validation numbers, always: real accuracy (are we improving?) and
  synthetic accuracy (are we forgetting?). A checkpoint is only saved if real
  improved AND synthetic has not dropped more than --forget-budget.
* Real val/test accuracy is reported PER SOURCE, because an average hides the
  case where one source is quietly poisoning another.

USAGE
-----
  # 1. see what the model does today, before touching it
  python finetune_real.py eval --ckpt parseq_ctc.pt --synthetic-root synthv3_dataset

  # 2. blended fine-tune
    python parseq_ctc_ocr_finetune.py finetune --ckpt parseq_ctc.pt --out parseq_ctc_real.pt \
        --synthetic-root synthv3_dataset

  # 3. mine unlabelled photos: agreement -> pseudo-labels, disagreement -> review
  python finetune_real.py scan --ckpt parseq_ctc_real.pt --images /path/to/unlabelled

Requires parseq_ctc_ocr.py in the same directory (the model is imported, not
redefined, so the two files cannot drift apart).
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import random
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from simple_parseq import (Config, PARSeqCTC, collate, edit_distance,
                                load_model, pick_device, preprocess, read_manifest)
except ImportError as e:
    raise SystemExit("finetune_real.py needs parseq_ctc_ocr.py in the same "
                     f"directory (import failed: {e})")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x


# =====================================================================
# 1. CONFIG
# =====================================================================
@dataclass
class FTConfig:
    # blending
    p_real: float = 0.35            # probability a training sample is real
    w_ccpd: float = 1.0             # relative weight within the real pool
    w_cropped: float = 1.0
    w_indian: float = 1.0
    w_pseudo: float = 1.0

    # schedule
    steps: int = 3000
    warmup: int = 200
    batch_size: int = 32
    lr: float = 7e-5                # ~1/10 of the original peak LR
    weight_decay: float = 0.01
    eval_every: int = 250
    forget_budget: float = 0.02     # max allowed synthetic-accuracy regression

    # real-image handling
    box_jitter: float = 0.06        # random box expand/shift: real detectors are loose
    degrade_scale: float = 0.5      # halve blur/noise/jpeg probabilities
    ccpd_trim: float = 0.15         # crop off the Chinese province character

    # evaluation
    val_cap: int = 300              # per-source val images used during training
    synth_val_cap: int = 512
    seed: int = 42


SOURCES = ("ccpd", "cropped", "indian", "pseudo")


@dataclass
class Sample:
    path: str
    text: str
    source: str
    box: tuple | None = None        # (x1, y1, x2, y2) in original image coords


# =====================================================================
# 2. INDEXING THE REAL SOURCES
# =====================================================================
# CCPD encodes everything in the filename. Label tables come from the CCPD repo
# (note: no letter "I", and "O" sits at the end).
CCPD_ALPHABETS = list("ABCDEFGHJKLMNPQRSTUVWXYZO")
CCPD_ADS = list("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789O")
CCPD_SUBSETS = ["ccpd_base", "ccpd_blur", "ccpd_challenge", "ccpd_db",
                "ccpd_fn", "ccpd_rotate", "ccpd_tilt", "ccpd_weather"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def looks_like_plate(text: str) -> bool:
    """
    Guard against label noise. Some Pascal-VOC files put a class name
    ("licence_plate") in the <name> field instead of the plate text; that
    survives charset filtering as "LICENCEPLATE". Every real plate label
    contains at least one digit, so this filter is cheap and safe.
    """
    return len(text) >= 4 and any(c.isdigit() for c in text)


def index_ccpd(root, tok, trim: float) -> dict[str, list[Sample]]:
    """One list per CCPD subset, so we can balance across them later."""
    out = {}
    for subset in CCPD_SUBSETS:
        d = os.path.join(root, subset)
        if not os.path.isdir(d):
            continue
        entries = []
        for name in os.listdir(d):
            if not name.endswith(".jpg"):
                continue
            f = name.rsplit(".", 1)[0].split("-")
            if len(f) < 5:
                continue
            try:
                tl, br = f[2].split("_")
                x1, y1 = (int(v) for v in tl.split("&"))
                x2, y2 = (int(v) for v in br.split("&"))
                idx = [int(v) for v in f[4].split("_")]
                # idx = [province, letter, ads x5]. The province is a Chinese
                # glyph with no place in our charset, so it is dropped from the
                # label -- and trimmed off the image below, so that picture and
                # label still agree. Without the trim the model would be taught
                # "skip the leading glyph", which would be a disaster on Indian
                # plates. Set --ccpd-trim 0 to disable and compare.
                text = CCPD_ALPHABETS[idx[1]] + "".join(CCPD_ADS[i] for i in idx[2:])
            except (ValueError, IndexError):
                continue
            text = tok.clean(text)
            if not looks_like_plate(text) or x2 - x1 < 8 or y2 - y1 < 8:
                continue
            if trim > 0:
                x1 = int(x1 + trim * (x2 - x1))
            entries.append(Sample(os.path.join(d, name), text, "ccpd", (x1, y1, x2, y2)))
        out[subset] = entries
    return out


def index_cropped(root, tok) -> list[Sample]:
    """car_licence3: already cropped, filename stem is the label."""
    entries = []
    for split in ("train", "valid", "val", "test"):
        d = os.path.join(root, split)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if not name.endswith(IMG_EXTS):
                continue
            text = tok.clean(name.rsplit(".", 1)[0].split("_")[0])
            if looks_like_plate(text):
                entries.append(Sample(os.path.join(d, name), text, "cropped", None))
    return entries


def index_indian(root, tok) -> list[Sample]:
    """indian_plate2: Pascal-VOC sidecar holds the label and the box."""
    entries = []
    for xml_path in glob.glob(os.path.join(root, "**", "*.xml"), recursive=True):
        img_path = next((xml_path[:-4] + e for e in IMG_EXTS
                         if os.path.exists(xml_path[:-4] + e)), None)
        if img_path is None:
            continue
        try:
            objects = ET.parse(xml_path).getroot().findall("object")
        except ET.ParseError:
            continue
        for obj in objects:
            text = tok.clean(obj.findtext("name", ""))
            if not looks_like_plate(text):
                continue
            bb = obj.find("bndbox")
            box = None
            if bb is not None:
                try:
                    box = (int(float(bb.findtext("xmin"))), int(float(bb.findtext("ymin"))),
                           int(float(bb.findtext("xmax"))), int(float(bb.findtext("ymax"))))
                except (TypeError, ValueError):
                    box = None
            entries.append(Sample(img_path, text, "indian", box))
            break       # one plate per image
    return entries


def index_pseudo(jsonl_path, tok) -> list[Sample]:
    """Pseudo-labels produced by `scan` (both heads agreed, high confidence)."""
    entries = []
    if not jsonl_path or not os.path.exists(jsonl_path):
        return entries
    base = os.path.dirname(os.path.abspath(jsonl_path))
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            p = d["path"] if os.path.isabs(d["path"]) else os.path.join(base, d["path"])
            text = tok.clean(d["text"])
            if looks_like_plate(text) and os.path.exists(p):
                entries.append(Sample(p, text, "pseudo", None))
    return entries


# =====================================================================
# 3. DETERMINISTIC SPLIT
# =====================================================================
def bucket(key: str) -> int:
    """Stable 0-99 bucket from a path. No state files, identical every run, and
    a given image can never drift between train and test."""
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 100


def split_samples(samples: list[Sample]) -> dict[str, list[Sample]]:
    """90 / 5 / 5. The test slice is never trained on and never tuned against."""
    out = {"train": [], "val": [], "test": []}
    for s in samples:
        b = bucket(s.path)
        out["train" if b < 90 else "val" if b < 95 else "test"].append(s)
    return out


# =====================================================================
# 4. REAL-IMAGE LOADING
# =====================================================================
def augment_real(img: np.ndarray, degrade: float) -> np.ndarray:
    """
    Geometry at full strength (this is what stops a small real set being
    memorised); degradation scaled down (real photos already contain the real
    thing, so piling synthetic blur on top just moves them off-distribution).
    """
    h, w = img.shape[:2]

    if random.random() < 0.8:                                   # perspective / tilt
        m = 0.05
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + np.float32([[random.uniform(-m, m) * w,
                                 random.uniform(-m, m) * h] for _ in range(4)])
        img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                                  borderMode=cv2.BORDER_REPLICATE)
    if random.random() < 0.6:                                   # contrast / brightness
        img = cv2.convertScaleAbs(img, alpha=random.uniform(0.8, 1.25),
                                  beta=random.uniform(-25, 25))
    if random.random() < 0.25:                                  # gamma
        g = random.uniform(0.7, 1.4)
        lut = np.clip((np.arange(256) / 255.0) ** g * 255, 0, 255).astype(np.uint8)
        img = cv2.LUT(img, lut)

    if random.random() < 0.20 * degrade:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if random.random() < 0.20 * degrade:                        # resolution loss
        s = random.uniform(0.5, 0.85)
        small = cv2.resize(img, (max(4, int(w * s)), max(4, int(h * s))),
                           interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if random.random() < 0.20 * degrade:
        noise = np.random.normal(0, random.uniform(3, 9), img.shape)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.20 * degrade:
        _, enc = cv2.imencode(".jpg", img,
                              [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(40, 85)])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if random.random() < 0.12:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    return img


def load_real(s: Sample, cfg: Config, ft: FTConfig, train: bool) -> torch.Tensor:
    """Read, crop to the plate, augment (train only), preprocess exactly as the
    original training pipeline did."""
    img = cv2.imread(s.path, cv2.IMREAD_COLOR)
    if img is None:
        return torch.zeros(3, cfg.img_h, cfg.img_w)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if s.box is not None:
        H, W = img.shape[:2]
        x1, y1, x2, y2 = s.box
        if train and ft.box_jitter > 0:
            # A real detector's box is never tight. Jittering it here means the
            # model sees the crops it will actually get at inference.
            bw, bh = x2 - x1, y2 - y1
            j = ft.box_jitter
            x1 += int(random.uniform(-j, j) * bw)
            x2 += int(random.uniform(-j, j) * bw)
            y1 += int(random.uniform(-j, j) * bh)
            y2 += int(random.uniform(-j, j) * bh)
        x1, x2 = max(0, min(x1, x2)), min(W, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(H, max(y1, y2))
        if x2 - x1 > 4 and y2 - y1 > 4:
            img = img[y1:y2, x1:x2]

    if train:
        img = augment_real(img, ft.degrade_scale)
    return preprocess(img, cfg)


class RealDataset(Dataset):
    """Plain list of real samples — used for val and test."""

    def __init__(self, samples, cfg, ft, train=False):
        self.samples, self.cfg, self.ft, self.train = samples, cfg, ft, train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return load_real(s, self.cfg, self.ft, self.train), s.text


class BlendDataset(Dataset):
    """
    The replay mixer. Length is virtual (steps x batch), so "one epoch" here
    means "the step budget", not "one pass over the data".

    Each item: real with probability p_real (source chosen by weight, CCPD
    balanced across its subsets), otherwise a synthetic sample.
    """

    def __init__(self, real_pools: dict, ccpd_subsets: dict, synth_samples,
                 synth_root, cfg: Config, ft: FTConfig, length: int):
        self.pools = {k: v for k, v in real_pools.items() if v}
        self.ccpd_subsets = {k: v for k, v in ccpd_subsets.items() if v}
        self.synth = synth_samples
        self.synth_root = synth_root
        self.cfg, self.ft, self.length = cfg, ft, length

        weights = {"ccpd": ft.w_ccpd, "cropped": ft.w_cropped,
                   "indian": ft.w_indian, "pseudo": ft.w_pseudo}
        self.names = [n for n in self.pools if weights[n] > 0]
        self.weights = [weights[n] for n in self.names]

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        use_real = self.names and random.random() < self.ft.p_real
        if use_real:
            name = random.choices(self.names, weights=self.weights, k=1)[0]
            if name == "ccpd" and self.ccpd_subsets:
                # equal weight per subset: base/blur/challenge/db/... all matter
                subset = random.choice(list(self.ccpd_subsets))
                s = random.choice(self.ccpd_subsets[subset])
            else:
                s = random.choice(self.pools[name])
            return load_real(s, self.cfg, self.ft, True), s.text

        path, text = random.choice(self.synth)
        img = cv2.imread(os.path.join(self.synth_root, path), cv2.IMREAD_COLOR)
        if img is None:
            return torch.zeros(3, self.cfg.img_h, self.cfg.img_w), text
        from parseq_ctc_ocr import augment as augment_synth
        img = augment_synth(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return preprocess(img, self.cfg), text


# =====================================================================
# 5. DATA ASSEMBLY + EVALUATION
# =====================================================================
def build_sources(args, tok, ft: FTConfig):
    """Index every source, split it, and print what was found."""
    print("indexing real sources ...")
    ccpd_subsets = index_ccpd(args.ccpd, tok, ft.ccpd_trim) if args.ccpd else {}
    pools = {
        "ccpd": [s for v in ccpd_subsets.values() for s in v],
        "cropped": index_cropped(args.cropped, tok) if args.cropped else [],
        "indian": index_indian(args.indian, tok) if args.indian else [],
        "pseudo": index_pseudo(args.pseudo, tok),
    }
    splits = {name: split_samples(v) for name, v in pools.items()}
    # Pseudo-labels are the model's OWN output. Validating against them would be
    # circular and would inflate the real mean, so they are training-only.
    splits["pseudo"] = {"train": pools["pseudo"], "val": [], "test": []}

    for name, v in pools.items():
        if not v:
            continue
        sp = splits[name]
        ex = ", ".join(s.text for s in v[:3])
        print(f"  {name:<8} {len(v):>7d}  (train {len(sp['train'])}, "
              f"val {len(sp['val'])}, test {len(sp['test'])})   e.g. {ex}")
    if ccpd_subsets:
        print("  ccpd subsets: " + ", ".join(f"{k.replace('ccpd_','')}={len(v)}"
                                             for k, v in ccpd_subsets.items()))
    print("  ^ sanity-check those example labels before training.\n")

    # train pools keep only the train bucket; CCPD stays split by subset
    train_pools = {n: splits[n]["train"] for n in pools}
    train_ccpd = {k: [s for s in v if bucket(s.path) < 90] for k, v in ccpd_subsets.items()}
    return pools, splits, train_pools, train_ccpd


def load_synthetic(args, cfg: Config, tok):
    """
    Reproduces the ORIGINAL train/val split exactly (same seed, same shuffle,
    same 5% slice), so replay never trains on what we validate against.
    """
    samples = read_manifest(args.synthetic_root, args.manifest, cfg, tok)
    random.Random(cfg.seed).shuffle(samples)
    n_val = max(1, int(len(samples) * cfg.val_split)) if len(samples) > 20 else 0
    val, train = samples[:n_val], samples[n_val:]
    print(f"synthetic: train {len(train)}, val {len(val)}")
    return train, val


class SynthValDataset(Dataset):
    def __init__(self, samples, root, cfg):
        self.samples, self.root, self.cfg = samples, root, cfg

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, text = self.samples[i]
        img = cv2.imread(os.path.join(self.root, path), cv2.IMREAD_COLOR)
        if img is None:
            return torch.zeros(3, self.cfg.img_h, self.cfg.img_w), text
        return preprocess(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), self.cfg), text


@torch.no_grad()
def accuracy(model, loader, device, want_ctc=True):
    """-> (word acc, char acc, ctc word acc, n)."""
    model.eval()
    exact = ctc_exact = n = 0
    char = 0.0
    for images, labels in loader:
        images = images.to(device)
        preds, _ = model.predict(images)
        ctc = model.predict_ctc(images) if want_ctc else [""] * len(labels)
        for p, c, g in zip(preds, ctc, labels):
            exact += int(p == g)
            ctc_exact += int(c == g)
            char += 1.0 - edit_distance(p, g) / max(len(g), len(p), 1)
            n += 1
    d = max(n, 1)
    return exact / d, char / d, ctc_exact / d, n


def make_loader(ds, batch_size, workers, shuffle=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, collate_fn=collate,
                      persistent_workers=workers > 0)


def eval_all(model, real_loaders, synth_loader, device):
    """Per-source real accuracy plus synthetic accuracy, as one report."""
    rows = {}
    for name, ld in real_loaders.items():
        rows[name] = accuracy(model, ld, device)
    if synth_loader is not None:
        rows["synthetic"] = accuracy(model, synth_loader, device)
    return rows


def print_report(rows, prefix=""):
    print(f"{prefix}{'source':<12}{'word':>8}{'char':>8}{'ctc':>8}{'n':>8}")
    for name, (w, c, x, n) in rows.items():
        print(f"{prefix}{name:<12}{w:>8.4f}{c:>8.4f}{x:>8.4f}{n:>8d}")


def real_mean(rows) -> float:
    """Unweighted mean over real sources — every source counts the same, so a
    big one cannot mask a regression in a small one."""
    vals = [v[0] for k, v in rows.items() if k != "synthetic"]
    return float(np.mean(vals)) if vals else 0.0


# =====================================================================
# 6. FINE-TUNE
# =====================================================================
def finetune(args):
    ft = FTConfig(p_real=args.p_real, steps=args.steps, warmup=args.warmup,
                  batch_size=args.batch_size, lr=args.lr,
                  eval_every=args.eval_every, forget_budget=args.forget_budget,
                  ccpd_trim=args.ccpd_trim, degrade_scale=args.degrade_scale,
                  box_jitter=args.box_jitter, val_cap=args.val_cap,
                  w_ccpd=args.w_ccpd, w_cropped=args.w_cropped,
                  w_indian=args.w_indian, w_pseudo=args.w_pseudo)
    random.seed(ft.seed)
    np.random.seed(ft.seed)
    torch.manual_seed(ft.seed)

    device = args.device or pick_device()
    model, cfg = load_model(args.ckpt, device)     # cfg comes from the ckpt, so
    tok = model.tok                                # preprocessing always matches
    print(f"device={device}  img={cfg.img_h}x{cfg.img_w}  keep_aspect={cfg.keep_aspect}  "
          f"ctc_weight={cfg.ctc_weight}")

    pools, splits, train_pools, train_ccpd = build_sources(args, tok, ft)
    if not any(train_pools.values()):
        raise SystemExit("no real training data found — check --ccpd/--cropped/--indian")
    synth_train, synth_val = load_synthetic(args, cfg, tok)

    # loaders
    rng = random.Random(ft.seed)
    real_val_loaders = {}
    for name in SOURCES:
        v = splits[name]["val"]
        if not v:
            continue
        if len(v) > ft.val_cap:
            v = rng.sample(v, ft.val_cap)
        real_val_loaders[name] = make_loader(RealDataset(v, cfg, ft), ft.batch_size,
                                             args.num_workers)
    sv = synth_val if len(synth_val) <= ft.synth_val_cap else rng.sample(synth_val, ft.synth_val_cap)
    synth_loader = make_loader(SynthValDataset(sv, args.synthetic_root, cfg),
                               ft.batch_size, args.num_workers)

    train_ld = make_loader(
        BlendDataset(train_pools, train_ccpd, synth_train, args.synthetic_root,
                     cfg, ft, ft.steps * ft.batch_size),
        ft.batch_size, args.num_workers, shuffle=False)

    # ---- baseline BEFORE any update: this is the number to beat ----
    print("\nbaseline (before fine-tuning):")
    base = eval_all(model, real_val_loaders, synth_loader, device)
    print_report(base, "  ")
    base_real, base_synth = real_mean(base), base["synthetic"][0]
    print(f"  real mean {base_real:.4f}   synthetic {base_synth:.4f}\n")

    # ---- optimiser: low LR, warmup, cosine. Nothing frozen. ----
    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or "embed" in n_ or "pos_" in n_ else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": ft.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=ft.lr, betas=(0.9, 0.99))

    def lr_at(step):
        if step < ft.warmup:
            return (step + 1) / max(ft.warmup, 1)
        t = (step - ft.warmup) / max(ft.steps - ft.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # best_real tracks the best OBSERVED score, not "better than baseline". If it
    # were seeded with base_real, a run that never beats the starting point would
    # write no checkpoint at all and leave you with nothing to inspect.
    best_real, saved_any, step = -1.0, False, 0
    run_p = run_c = seen = 0.0
    bar = tqdm(train_ld, total=ft.steps, desc="finetune")
    model.train()
    for images, labels in bar:
        images = images.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            loss, p_loss, c_loss = model.loss(images, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        run_p += p_loss.item()
        run_c += c_loss.item()
        seen += 1
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(parseq=f"{p_loss.item():.3f}", ctc=f"{c_loss.item():.3f}",
                            lr=f"{sched.get_last_lr()[0]:.2e}")

        if step % ft.eval_every == 0 or step == ft.steps:
            rows = eval_all(model, real_val_loaders, synth_loader, device)
            r, s = real_mean(rows), rows["synthetic"][0]
            drop = base_synth - s
            print(f"\nstep {step}: parseq={run_p / seen:.4f} ctc={run_c / seen:.4f}  "
                  f"real={r:.4f} (base {base_real:.4f})  "
                  f"synthetic={s:.4f} (base {base_synth:.4f}, drop {drop:+.4f})")
            print_report(rows, "  ")
            run_p = run_c = seen = 0.0

            # guarded checkpoint: improving on real is not enough on its own
            if r > best_real and drop <= ft.forget_budget:
                best_real, saved_any = r, True
                torch.save({"model": model.state_dict(), "config": asdict(cfg)}, args.out)
                print(f"  saved -> {args.out}  (real {r:.4f})")
            elif drop > ft.forget_budget:
                print(f"  ! forgetting: synthetic dropped {drop:.4f} > "
                      f"{ft.forget_budget:.4f}. Not saving. Lower --p-real or --lr.")
            model.train()
        if step >= ft.steps:
            break

    print(f"\ndone. best real mean {best_real:.4f} (baseline {base_real:.4f})")
    if not saved_any:
        print(f"NOTHING SAVED: every checkpoint exceeded the forgetting budget. "
              f"{args.ckpt} is untouched. Lower --p-real or --lr and retry.")
    elif best_real <= base_real:
        print(f"Fine-tuning did not beat the baseline. {args.out} is the best "
              f"observed run, but prefer {args.ckpt} unless per-source numbers "
              f"say otherwise.")
    else:
        print(f"-> {args.out}. Now run `eval --split test` with the new checkpoint.")


# =====================================================================
# 7. EVAL (frozen test split)
# =====================================================================
def eval_cmd(args):
    device = args.device or pick_device()
    model, cfg = load_model(args.ckpt, device)
    ft = FTConfig(ccpd_trim=args.ccpd_trim)
    pools, splits, _, _ = build_sources(args, model.tok, ft)

    which = args.split
    loaders = {}
    for name in SOURCES:
        v = splits[name][which]
        if v:
            if args.cap and len(v) > args.cap:
                v = random.Random(ft.seed).sample(v, args.cap)
            loaders[name] = make_loader(RealDataset(v, cfg, ft), args.batch_size,
                                        args.num_workers)
    synth_loader = None
    if args.synthetic_root:
        _, synth_val = load_synthetic(args, cfg, model.tok)
        if args.cap and len(synth_val) > args.cap:
            synth_val = random.Random(ft.seed).sample(synth_val, args.cap)
        synth_loader = make_loader(SynthValDataset(synth_val, args.synthetic_root, cfg),
                                   args.batch_size, args.num_workers)

    print(f"\n{which} split:")
    rows = eval_all(model, loaders, synth_loader, device)
    print_report(rows, "  ")
    print(f"  real mean {real_mean(rows):.4f}")


# =====================================================================
# 8. SCAN — pseudo-labels and an active-learning queue
# =====================================================================
@torch.no_grad()
def scan(args):
    """
    Two heads decoding independently is a free ensemble signal:
      agree + confident  -> pseudo-label, cheap training data in the real domain
      disagree or unsure -> the images worth your hand-labelling time
    Feed the pseudo-labels back with --pseudo and repeat. That is the loop that
    compounds.
    """
    device = args.device or pick_device()
    model, cfg = load_model(args.ckpt, device)
    ft = FTConfig()

    paths = []
    for ext in IMG_EXTS:
        paths += glob.glob(os.path.join(args.images, "**", f"*{ext}"), recursive=True)
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit(f"no images under {args.images}")
    print(f"scanning {len(paths)} images ...")

    samples = [Sample(p, "", "scan", None) for p in paths]
    ld = make_loader(RealDataset(samples, cfg, ft), args.batch_size, args.num_workers)

    keep, review = [], []
    i = 0
    for images, _ in tqdm(ld, desc="scan"):
        images = images.to(device)
        preds, confs = model.predict(images)
        ctc = model.predict_ctc(images)
        for p, c, conf in zip(preds, ctc, confs):
            path = paths[i]
            i += 1
            agree = (p == c) and len(p) >= 4
            if agree and conf >= args.conf:
                keep.append({"path": path, "text": p, "conf": round(conf, 4)})
            else:
                review.append({"path": path, "parseq": p, "ctc": c,
                               "conf": round(conf, 4), "agree": int(agree)})

    with open(args.out_pseudo, "w") as f:
        for r in keep:
            f.write(json.dumps(r) + "\n")
    review.sort(key=lambda r: (r["agree"], r["conf"]))     # least certain first
    with open(args.out_review, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "parseq", "ctc", "conf", "agree"])
        w.writeheader()
        w.writerows(review)

    n = len(paths)
    print(f"\npseudo-labels : {len(keep):6d}  ({len(keep) / n:.1%})  -> {args.out_pseudo}")
    print(f"needs review  : {len(review):6d}  ({len(review) / n:.1%})  -> {args.out_review}")
    print("Label the top of the review file first — those are worth many random "
          "images each. Then re-run finetune with --pseudo and the new labels.")


# =====================================================================
# 9. CLI
# =====================================================================
ROOT_DEFAULT = "/Users/rzt/Documents/ocr_env"


def add_source_args(p, root):
    p.add_argument("--ccpd", default=os.path.join(root, "CCPD2019"))
    p.add_argument("--cropped", default=os.path.join(root, "licenceplate_images/car_licence3"))
    p.add_argument("--indian", default=os.path.join(root, "licenceplate_images/indian_plate2"))
    p.add_argument("--pseudo", default=None, help="pseudo_labels.jsonl from `scan`")
    p.add_argument("--ccpd-trim", type=float, default=0.15,
                   help="fraction of plate width trimmed from the left to remove "
                        "the Chinese province glyph (0 disables)")
    p.add_argument("--manifest", default="manifest.jsonl")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default=None)


def main():
    ap = argparse.ArgumentParser(description="Real-data fine-tuning with synthetic replay")
    sub = ap.add_subparsers(dest="cmd", required=True)
    root = os.environ.get("OCR_ROOT", ROOT_DEFAULT)

    f = sub.add_parser("finetune")
    f.add_argument("--ckpt", default="parseq_ctc.pt")
    f.add_argument("--out", default="parseq_ctc_real.pt")
    f.add_argument("--synthetic-root", required=True)
    f.add_argument("--steps", type=int, default=10000)
    f.add_argument("--warmup", type=int, default=200)
    f.add_argument("--batch-size", type=int, default=32)
    f.add_argument("--lr", type=float, default=1e-4)
    f.add_argument("--p-real", type=float, default=0.5)
    f.add_argument("--w-ccpd", type=float, default=1.0)
    f.add_argument("--w-cropped", type=float, default=1.0)
    f.add_argument("--w-indian", type=float, default=1.0)
    f.add_argument("--w-pseudo", type=float, default=1.0)
    f.add_argument("--eval-every", type=int, default=200)
    f.add_argument("--forget-budget", type=float, default=0.05)
    f.add_argument("--degrade-scale", type=float, default=0.5)
    f.add_argument("--box-jitter", type=float, default=0.06)
    f.add_argument("--val-cap", type=int, default=300)
    add_source_args(f, root)
    f.set_defaults(func=finetune)

    e = sub.add_parser("eval")
    e.add_argument("--ckpt", default="parseq_ctc.pt")
    e.add_argument("--synthetic-root", default=None)
    e.add_argument("--split", choices=["val", "test"], default="test")
    e.add_argument("--batch-size", type=int, default=32)
    e.add_argument("--cap", type=int, default=500)
    add_source_args(e, root)
    e.set_defaults(func=eval_cmd)

    s = sub.add_parser("scan")
    s.add_argument("--ckpt", default="parseq_ctc_real.pt")
    s.add_argument("--images", required=True, help="directory of unlabelled crops")
    s.add_argument("--conf", type=float, default=0.90)
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--out-pseudo", default="pseudo_labels.jsonl")
    s.add_argument("--out-review", default="review.csv")
    s.add_argument("--num-workers", type=int, default=4)
    s.add_argument("--device", default=None)
    s.set_defaults(func=scan)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


# # 1. what does the model do on real data today? (this is your before-number)
# python parseq_ctc_ocr_finetune.py eval --ckpt parseq_ctc.pt --split val \
#     --synthetic-root synthv3_dataset

# # 2. blended fine-tune
# python parseq_ctc_ocr_finetune.py finetune --ckpt parseq_ctc.pt --out parseq_ctc_real.pt \
#     --synthetic-root synthv3_dataset

# # 3. honest final number, on the frozen test split
# python parseq_ctc_ocr_finetune.py eval --ckpt parseq_ctc_real.pt --split test \
#     --synthetic-root synthv3_dataset

# # 4. mine unlabelled photos
# python parseq_ctc_ocr_finetune.py scan --ckpt parseq_ctc_real.pt --images /path/to/unlabelled