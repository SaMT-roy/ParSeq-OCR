#!/usr/bin/env python3
"""
PARSeq + auxiliary CTC head — single-file OCR for 1/2/3-line number plates.

WHY TWO HEADS
-------------
A pure autoregressive decoder can drive its loss down by modelling
p(next char | previous chars) and barely reading the image — plates are highly
structured, so the label prior alone gets you far. The result fits the training
set and hallucinates plausible plates on unseen ones.

The CTC head prevents that. It is forced to emit characters in raster order
along the flattened feature grid, which is only possible if the encoder
produces per-character, positionally-localised features. It supplies the
monotonic alignment prior that makes CNN+LSTM+CTC generalise so well, for the
price of one linear layer.

The PARSeq decoder then handles what CTC cannot: 2- and 3-line layouts, where
it attends anywhere in 2D and does not care about reading order at all. It is
also trained with Permutation Language Modelling, so no single scan order can
become a crutch, and it refines its output with a bidirectional cloze pass.

    total loss = parseq_loss + ctc_weight * ctc_loss     (decode with PARSeq)

MULTI-LINE + CTC
----------------
Flatten the encoder grid in raster order and every token of line 1 precedes
every token of line 2, so a valid monotonic CTC path exists (this is the
OrigamiNet idea). The failure mode is a line firing in two row bands at once
("GJ06GJ06" — CTC only collapses *adjacent* duplicates), so the grid is pooled
down to `ctc_bands` rows before the CTC head, leaving little room to double
fire.

LABEL ORDER
-----------
The CTC head requires ground-truth text in raster reading order: a plate
reading "GJ06" / "AB1234" must be stored as "GJ06AB1234". If ctc_loss plateaus
high while parseq_loss falls, your labels are not in reading order — set
--ctc-weight 0 and the model still trains (as plain PARSeq).

USAGE
-----
  python parseq_ctc_ocr.py train    --root DATASET_DIR --manifest manifest.jsonl
  python parseq_ctc_ocr.py predict  --ckpt parseq_ctc.pt --image plate.png
  python parseq_ctc_ocr.py diagnose --ckpt parseq_ctc.pt   # is it reading pixels?

Manifest: one JSON object per line, {"color": "images/0001.png", "text": "GJ06AB1234"}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from itertools import permutations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x


# =====================================================================
# 1. CONFIG
# =====================================================================
@dataclass
class Config:
    # ---- data ----
    charset: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    img_h: int = 96                 # taller than single-line setups: a 3-line
    img_w: int = 128                # plate needs vertical room to survive /8
    keep_aspect: bool = True        # letterbox; preserves the 1/2/3-line layout cue
    max_label_length: int = 12
    img_key: str = "color"
    text_key: str = "text"

    # ---- model ----
    embed_dim: int = 128
    enc_depth: int = 4
    enc_heads: int = 4
    dec_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    # ---- permutation language modelling ----
    max_gen_perms: int = 6          # K in the paper
    perm_mirrored: bool = True
    refine_iters: int = 1           # cloze refinement passes at inference

    # ---- auxiliary CTC head ----
    ctc_weight: float = 0.2         # 0.0 disables it entirely
    ctc_bands: int = 4              # grid rows kept before the CTC head

    # ---- training ----
    epochs: int = 40
    batch_size: int = 64
    lr: float = 7e-4
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    val_split: float = 0.05
    num_workers: int = 4
    seed: int = 42


# =====================================================================
# 2. TOKENIZER
# =====================================================================
class Tokenizer:
    """
    Shared id space for both heads:
        0        -> [EOS] for the decoder, <blank> for CTC (same slot, both
                    mean "nothing to emit", so one head size fits both)
        1 .. N   -> charset
        N+1      -> [BOS]  (decoder input only)
        N+2      -> [PAD]  (decoder input only, ignored by the loss)
    The head predicts only ids 0..N.
    """

    def __init__(self, charset: str, max_label_length: int):
        self.itos = ["[E]"] + list(charset) + ["[B]", "[P]"]
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.eos_id = 0
        self.blank_id = 0
        self.bos_id = len(self.itos) - 2
        self.pad_id = len(self.itos) - 1
        self.num_classes = len(charset) + 1
        self.max_label_length = max_label_length

    def clean(self, text: str) -> str:
        text = "".join(c for c in text.upper() if c in self.stoi)
        return text[: self.max_label_length]

    def encode(self, labels: list[str], device=None) -> torch.Tensor:
        """-> (B, L+2): [BOS] chars... [EOS] [PAD]...  (L = longest in batch)."""
        L = max(1, max(len(s) for s in labels))
        rows = []
        for s in labels:
            ids = [self.bos_id] + [self.stoi[c] for c in s] + [self.eos_id]
            ids += [self.pad_id] * (L + 2 - len(ids))
            rows.append(ids)
        return torch.tensor(rows, dtype=torch.long, device=device)

    def encode_ctc(self, labels: list[str], device=None):
        """-> flat targets (sum of lengths,) and their lengths (B,)."""
        flat = [self.stoi[c] for s in labels for c in s]
        lens = [len(s) for s in labels]
        return (torch.tensor(flat, dtype=torch.long, device=device),
                torch.tensor(lens, dtype=torch.long, device=device))

    def decode(self, ids: list[int]) -> str:
        """Decoder output: stop at the first [EOS]."""
        out = []
        for i in ids:
            if i == self.eos_id:
                break
            if i < len(self.itos) and i not in (self.bos_id, self.pad_id):
                out.append(self.itos[i])
        return "".join(out)

    def decode_ctc(self, ids: list[int]) -> str:
        """CTC output: collapse adjacent duplicates, then drop blanks."""
        out, prev = [], -1
        for i in ids:
            if i != prev and i != self.blank_id:
                out.append(self.itos[i])
            prev = i
        return "".join(out)


# =====================================================================
# 3. DATA
# =====================================================================
def augment(img: np.ndarray) -> np.ndarray:
    """Geometric + photometric augmentation. On a synthetic-only training set
    this is worth more than any architectural change."""
    h, w = img.shape[:2]

    if random.random() < 0.7:                      # perspective / tilt
        m = 0.06
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + np.float32([[random.uniform(-m, m) * w,
                                 random.uniform(-m, m) * h] for _ in range(4)])
        M = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if random.random() < 0.7:                      # contrast / brightness
        img = cv2.convertScaleAbs(img, alpha=random.uniform(0.7, 1.3),
                                  beta=random.uniform(-30, 30))
    if random.random() < 0.3:                      # gamma
        g = random.uniform(0.6, 1.6)
        lut = np.clip((np.arange(256) / 255.0) ** g * 255, 0, 255).astype(np.uint8)
        img = cv2.LUT(img, lut)

    r = random.random()                            # capture degradation
    if r < 0.25:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    elif r < 0.40:                                 # resolution loss
        s = random.uniform(0.35, 0.8)
        small = cv2.resize(img, (max(4, int(w * s)), max(4, int(h * s))),
                           interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    elif r < 0.55:                                 # motion blur
        k = random.choice([5, 7])
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1.0 / k
        img = cv2.filter2D(img, -1, kern.T if random.random() < 0.5 else kern)

    if random.random() < 0.3:                      # sensor noise
        noise = np.random.normal(0, random.uniform(4, 14), img.shape)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.3:                      # jpeg artefacts
        _, enc = cv2.imencode(".jpg", img,
                              [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(25, 75)])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if random.random() < 0.15:                     # grayscale cameras
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    return img


def preprocess(img: np.ndarray, cfg: Config) -> torch.Tensor:
    """RGB uint8 HWC -> float CHW in [-1, 1], sized (img_h, img_w)."""
    H, W = cfg.img_h, cfg.img_w
    if cfg.keep_aspect:
        h, w = img.shape[:2]
        s = min(W / w, H / h)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA
                             if s < 1 else cv2.INTER_LINEAR)
        canvas = np.zeros((H, W, 3), np.uint8)
        y, x = (H - nh) // 2, (W - nw) // 2
        canvas[y:y + nh, x:x + nw] = resized
        img = canvas
    else:
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    x = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
    return torch.from_numpy(x.transpose(2, 0, 1))


class PlateDataset(Dataset):
    def __init__(self, root, samples, cfg: Config, train: bool):
        self.root, self.samples, self.cfg, self.train = root, samples, cfg, train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, text = self.samples[i]
        img = cv2.imread(os.path.join(self.root, path), cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((self.cfg.img_h, self.cfg.img_w, 3), np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.train:
            img = augment(img)
        return preprocess(img, self.cfg), text


def collate(batch):
    imgs, labels = zip(*batch)
    return torch.stack(imgs), list(labels)


def read_manifest(root, manifest, cfg: Config, tok: Tokenizer):
    samples = []
    with open(os.path.join(root, manifest)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            text = tok.clean(str(d[cfg.text_key]))
            if text:
                samples.append((d[cfg.img_key], text))
    return samples


# =====================================================================
# 4. BUILDING BLOCKS
# =====================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden, dropout=0.0):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden)
        self.w_val = nn.Linear(dim, hidden)
        self.w_out = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w_out(F.silu(self.w_gate(x)) * self.w_val(x)))


class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.hd, self.p = heads, dim // heads, dropout
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v,
                                           dropout_p=self.p if self.training else 0.0)
        return self.drop(self.proj(o.transpose(1, 2).reshape(B, N, D)))


class EncoderBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, dropout):
        super().__init__()
        self.n1, self.n2 = RMSNorm(dim), RMSNorm(dim)
        self.attn = SelfAttention(dim, heads, dropout)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class ConvStem(nn.Module):
    """Downsamples by 8. Convolutions learn strokes and edges (with locality
    and translation equivariance for free); the transformer learns layout."""

    def __init__(self, dim, in_ch=3):
        super().__init__()
        c1, c2 = dim // 4, dim // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, c1, 3, 2, 1, bias=False), nn.GroupNorm(8, c1), nn.GELU(),
            nn.Conv2d(c1, c1, 3, 1, 1, bias=False), nn.GroupNorm(8, c1), nn.GELU(),
            nn.Conv2d(c1, c2, 3, 2, 1, bias=False), nn.GroupNorm(8, c2), nn.GELU(),
            nn.Conv2d(c2, c2, 3, 1, 1, bias=False), nn.GroupNorm(8, c2), nn.GELU(),
            nn.Conv2d(c2, dim, 3, 2, 1, bias=False), nn.GroupNorm(8, dim), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    """Image -> visual tokens. No [CLS]: PARSeq cross-attends to every patch,
    and the CTC head needs the full grid."""

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.img_h % 8 == 0 and cfg.img_w % 8 == 0, "img size must be /8"
        self.grid = (cfg.img_h // 8, cfg.img_w // 8)
        self.stem = ConvStem(cfg.embed_dim)
        self.pos = nn.Parameter(torch.zeros(1, self.grid[0] * self.grid[1], cfg.embed_dim))
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            EncoderBlock(cfg.embed_dim, cfg.enc_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.enc_depth)
        )
        self.norm = RMSNorm(cfg.embed_dim)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        x = self.stem(x).flatten(2).transpose(1, 2)       # (B, gh*gw, D), raster order
        x = self.drop(x + self.pos)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    PARSeq decoder layer, two streams:
      query   = position queries — "which character sits at slot i?"
      content = the characters decoded so far
    The permutation mask applies to query->content attention, so one layer is
    enough (as in the paper) and the content stream never needs updating.
    """

    def __init__(self, dim, heads, mlp_ratio, dropout):
        super().__init__()
        self.norm_q, self.norm_c = RMSNorm(dim), RMSNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = RMSNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio), dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, query, content, memory, query_mask=None, content_pad_mask=None):
        c = self.norm_c(content)
        q = query + self.drop(self.self_attn(self.norm_q(query), c, c,
                                            attn_mask=query_mask,
                                            key_padding_mask=content_pad_mask,
                                            need_weights=False)[0])
        q = q + self.drop(self.cross_attn(self.norm1(q), memory, memory,
                                          need_weights=False)[0])
        return q + self.mlp(self.norm2(q))


# =====================================================================
# 5. MODEL
# =====================================================================
class PARSeqCTC(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = Tokenizer(cfg.charset, cfg.max_label_length)
        T = cfg.max_label_length + 1                      # chars + [EOS]

        self.encoder = Encoder(cfg)
        self.text_embed = nn.Embedding(len(self.tok.itos), cfg.embed_dim)
        self.pos_queries = nn.Parameter(torch.zeros(1, T, cfg.embed_dim))
        self.decoder = DecoderLayer(cfg.embed_dim, cfg.dec_heads, cfg.mlp_ratio, cfg.dropout)
        self.head = nn.Linear(cfg.embed_dim, self.tok.num_classes)
        self.ctc_head = nn.Linear(cfg.embed_dim, self.tok.num_classes)
        self.drop = nn.Dropout(cfg.dropout)

        nn.init.trunc_normal_(self.pos_queries, std=0.02)
        nn.init.trunc_normal_(self.text_embed.weight, std=0.02)

    # ---------------- auxiliary CTC branch ----------------
    def ctc_logits(self, memory: torch.Tensor) -> torch.Tensor:
        """
        (B, gh*gw, D) -> (B, ctc_bands*gw, C).
        Pool the grid vertically so each text line lands in roughly one band,
        then read it out in raster order: line 1 left->right, line 2, line 3.
        """
        B, _, D = memory.shape
        gh, gw = self.encoder.grid
        g = memory.transpose(1, 2).reshape(B, D, gh, gw)
        g = F.adaptive_avg_pool2d(g, (self.cfg.ctc_bands, gw))
        return self.ctc_head(g.flatten(2).transpose(1, 2))

    def ctc_loss(self, memory, labels: list[str]) -> torch.Tensor:
        logits = self.ctc_logits(memory)
        B, T, _ = logits.shape
        # CTC is numerically fragile in fp16 — always run it in fp32.
        logp = logits.float().log_softmax(-1).transpose(0, 1)      # (T, B, C)
        # MPS has no aten::_ctc_loss kernel. Run this one loss on the CPU:
        # .cpu() is differentiable, so gradients flow back to the device, and
        # the tensor is tiny (T*B*C ~ 150k floats) so the copy is free.
        if logp.device.type == "mps":
            logp = logp.cpu()
        targets, tgt_len = self.tok.encode_ctc(labels, logp.device)
        in_len = torch.full((B,), T, dtype=torch.long, device=logp.device)
        nll = F.ctc_loss(logp, targets, in_len, tgt_len,
                         blank=self.tok.blank_id, zero_infinity=True, reduction="sum")
        # Normalise per input frame, NOT per character (torch's reduction='mean'
        # divides by target length, which starts around 240 here and would swamp
        # the decoder's cross-entropy). Per frame both losses start near
        # log(num_classes), so ctc_weight means what it looks like it means.
        return (nll / (B * T)).to(memory.device)

    # ---------------- PARSeq decode step ----------------
    def decode(self, tgt_in, memory, query=None, query_mask=None, pad_mask=None):
        """tgt_in: (B, L) decoder input beginning with [BOS]."""
        B, L = tgt_in.shape
        null_ctx = self.text_embed(tgt_in[:, :1])          # [BOS]: no position info
        rest = self.text_embed(tgt_in[:, 1:]) + self.pos_queries[:, : L - 1]
        content = self.drop(torch.cat([null_ctx, rest], dim=1))
        if query is None:
            query = self.pos_queries[:, :L].expand(B, -1, -1)
        return self.decoder(self.drop(query), content, memory, query_mask, pad_mask)

    # ---------------- permutation language modelling ----------------
    def gen_perms(self, n_chars: int, device) -> torch.Tensor:
        """
        (K, n_chars+2) orderings over decoder positions, where position 0 is
        [BOS] and position n_chars+1 is [EOS].
        Row 0 is always left-to-right; row 1 is its exact reverse.
        """
        if n_chars <= 1:
            return torch.arange(n_chars + 2, device=device).unsqueeze(0)

        mirrored = self.cfg.perm_mirrored
        max_perms = math.factorial(n_chars) // (2 if mirrored else 1)
        k = max(1, min(self.cfg.max_gen_perms // (2 if mirrored else 1), max_perms))

        perms = [torch.arange(n_chars, device=device)]                # left-to-right
        if n_chars <= 6:                                              # sample w/o replacement
            pool = torch.tensor(list(permutations(range(n_chars))), device=device)
            keep = ~(pool == perms[0]).all(-1) & ~(pool.flip(-1) == perms[0]).all(-1)
            pool = pool[keep]
            if len(pool):
                idx = torch.randperm(len(pool), device=device)[: k - 1]
                perms += list(pool[idx])
        else:
            perms += [torch.randperm(n_chars, device=device) for _ in range(k - 1)]
        perms = torch.stack(perms)

        if mirrored:                                                  # interleave reverses
            perms = torch.stack([perms, perms.flip(-1)], dim=1).reshape(-1, n_chars)

        bos = perms.new_zeros(len(perms), 1)
        eos = perms.new_full((len(perms), 1), n_chars + 1)
        perms = torch.cat([bos, perms + 1, eos], dim=1)
        if len(perms) > 1:      # true right-to-left order predicts [EOS] first
            perms[1, 1:] = n_chars + 1 - torch.arange(n_chars + 1, device=device)
        return perms

    @staticmethod
    def query_mask_from_perm(perm: torch.Tensor) -> torch.Tensor:
        """
        Boolean mask (True = may not attend). The query for position p sees
        every position decoded BEFORE p, and never p itself — that would leak
        the answer.
        """
        T = perm.shape[0]
        mask = torch.zeros(T, T, dtype=torch.bool, device=perm.device)
        for i in range(T):
            mask[perm[i], perm[i + 1:]] = True
        mask.fill_diagonal_(True)
        return mask[1:, :-1]            # queries 1..T-1, keys 0..T-2

    # ---------------- training ----------------
    def loss(self, images, labels: list[str]):
        """-> (total, parseq_loss, ctc_loss) for logging."""
        tok = self.tok
        memory = self.encoder(images)
        tgt = tok.encode(labels, images.device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        # nothing may attend to [PAD], or to anything at/after [EOS]
        pad_mask = (tgt_in == tok.pad_id) | (tgt_in == tok.eos_id)

        perms = self.gen_perms(tgt.shape[1] - 2, images.device)
        total, numel = 0.0, 0
        n = (tgt_out != tok.pad_id).sum().item()
        for i, perm in enumerate(perms):
            qm = self.query_mask_from_perm(perm)
            logits = self.head(self.decode(tgt_in, memory, query_mask=qm, pad_mask=pad_mask))
            total = total + n * F.cross_entropy(
                logits.flatten(end_dim=1), tgt_out.flatten(),
                ignore_index=tok.pad_id, label_smoothing=self.cfg.label_smoothing)
            numel += n
            if i == 1:
                # only the two canonical orders supervise [EOS]; otherwise the
                # model would learn to stop from partial, unordered context
                tgt_out = torch.where(tgt_out == tok.eos_id, tok.pad_id, tgt_out)
                n = (tgt_out != tok.pad_id).sum().item()
        p_loss = total / max(numel, 1)

        if self.cfg.ctc_weight > 0:
            c_loss = self.ctc_loss(memory, labels)
            return p_loss + self.cfg.ctc_weight * c_loss, p_loss.detach(), c_loss.detach()
        zero = torch.zeros((), device=images.device)
        return p_loss, p_loss.detach(), zero

    # ---------------- inference ----------------
    @torch.no_grad()
    def forward(self, images) -> torch.Tensor:
        """Decoder logits (B, max_label_length+1, num_classes)."""
        tok, cfg = self.tok, self.cfg
        B, dev = images.shape[0], images.device
        T = cfg.max_label_length + 1
        memory = self.encoder(images)
        queries = self.pos_queries[:, :T].expand(B, -1, -1)
        causal = torch.ones(T, T, dtype=torch.bool, device=dev).triu(1)

        # pass 1: autoregressive, left to right
        tgt_in = torch.full((B, T), tok.pad_id, dtype=torch.long, device=dev)
        tgt_in[:, 0] = tok.bos_id
        chunks = []
        for i in range(T):
            j = i + 1
            out = self.decode(tgt_in[:, :j], memory,
                              query=queries[:, i:j], query_mask=causal[i:j, :j])
            p = self.head(out)                                    # (B, 1, C)
            chunks.append(p)
            if j < T:
                tgt_in[:, j] = p.argmax(-1).squeeze(1)
                if (tgt_in == tok.eos_id).any(-1).all():
                    break
        logits = torch.cat(chunks, dim=1)
        if logits.shape[1] < T:                                   # pad the early exit
            pad = logits.new_zeros(B, T - logits.shape[1], logits.shape[-1])
            pad[..., tok.eos_id] = 1e4
            logits = torch.cat([logits, pad], dim=1)

        # pass 2+: cloze refinement, every slot re-decided with full context
        if cfg.refine_iters:
            cloze = causal.clone()
            cloze[torch.ones(T, T, dtype=torch.bool, device=dev).triu(2)] = False
            bos = torch.full((B, 1), tok.bos_id, dtype=torch.long, device=dev)
            for _ in range(cfg.refine_iters):
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                pad_mask = (tgt_in == tok.eos_id).int().cumsum(-1) > 0
                logits = self.head(self.decode(tgt_in, memory, query=queries,
                                               query_mask=cloze, pad_mask=pad_mask))
        return logits

    @torch.no_grad()
    def predict(self, images):
        """-> (texts, confidences). Mean per-character probability."""
        probs = self.forward(images).softmax(-1)
        conf, ids = probs.max(-1)
        texts, scores = [], []
        for seq, c in zip(ids.tolist(), conf.tolist()):
            t = self.tok.decode(seq)
            texts.append(t)
            scores.append(float(np.mean(c[: len(t) + 1])))
        return texts, scores

    @torch.no_grad()
    def predict_ctc(self, images) -> list[str]:
        """Greedy CTC decode. Useful as a cross-check: if this head is right
        and the decoder is wrong, the decoder is ignoring the image."""
        ids = self.ctc_logits(self.encoder(images)).argmax(-1)
        return [self.tok.decode_ctc(s) for s in ids.tolist()]


# =====================================================================
# 6. TRAIN / EVAL
# =====================================================================
def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@torch.no_grad()
def evaluate(model, loader, device):
    """-> (word acc, char acc, CTC-head word acc)."""
    model.eval()
    exact = ctc_exact = total = 0
    char = 0.0
    use_ctc = model.cfg.ctc_weight > 0
    for images, labels in loader:
        images = images.to(device)
        preds, _ = model.predict(images)
        ctc_preds = model.predict_ctc(images) if use_ctc else [""] * len(labels)
        for p, cp, g in zip(preds, ctc_preds, labels):
            exact += int(p == g)
            ctc_exact += int(cp == g)
            char += 1.0 - edit_distance(p, g) / max(len(g), 1)
            total += 1
    n = max(total, 1)
    return exact / n, char / n, ctc_exact / n


def train(args):
    cfg = Config(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                 img_h=args.img_h, img_w=args.img_w,
                 max_label_length=args.max_label_length,
                 ctc_weight=args.ctc_weight, keep_aspect=not args.stretch,
                 num_workers=args.num_workers)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = args.device or pick_device()
    tok = Tokenizer(cfg.charset, cfg.max_label_length)
    samples = read_manifest(args.root, args.manifest, cfg, tok)
    random.Random(cfg.seed).shuffle(samples)
    n_val = max(1, int(len(samples) * cfg.val_split)) if len(samples) > 20 else 0
    train_s, val_s = samples[n_val:], samples[:n_val]
    print(f"device={device}  train={len(train_s)}  val={len(val_s)}  "
          f"grid={cfg.img_h // 8}x{cfg.img_w // 8}  ctc_weight={cfg.ctc_weight}")

    train_ld = DataLoader(PlateDataset(args.root, train_s, cfg, True),
                          batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                          num_workers=cfg.num_workers, collate_fn=collate,
                          persistent_workers=cfg.num_workers > 0)
    val_ld = DataLoader(PlateDataset(args.root, val_s, cfg, False),
                        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                        collate_fn=collate) if val_s else None

    model = PARSeqCTC(cfg).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or "embed" in name or "pos_" in name else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=cfg.lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * max(len(train_ld), 1),
        pct_start=0.075, cycle_momentum=False)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best, first_ctc = -1.0, None
    for epoch in range(cfg.epochs):
        model.train()
        run_p = run_c = 0.0
        bar = tqdm(train_ld, desc=f"epoch {epoch + 1}/{cfg.epochs}")
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
            run_p += p_loss.item()
            run_c += c_loss.item()
            if hasattr(bar, "set_postfix"):
                bar.set_postfix(parseq=f"{p_loss.item():.3f}", ctc=f"{c_loss.item():.3f}",
                                lr=f"{sched.get_last_lr()[0]:.2e}")

        nb = max(len(train_ld), 1)
        ep_p, ep_c = run_p / nb, run_c / nb
        msg = f"epoch {epoch + 1}: parseq={ep_p:.4f} ctc={ep_c:.4f}"
        score = -ep_p
        if val_ld:
            word, char, ctc_word = evaluate(model, val_ld, device)
            msg += f"  val_word={word:.4f} val_char={char:.4f} val_ctc_word={ctc_word:.4f}"
            score = word
        print(msg)

        # label-order sanity check: CTC needs raster reading order
        if cfg.ctc_weight > 0:
            first_ctc = ep_c if first_ctc is None else first_ctc
            if epoch == 4 and ep_c > 0.8 * first_ctc:
                print("  ! ctc loss is barely moving. Labels are probably not in "
                      "raster reading order (top line first) — or the plates need "
                      "more vertical resolution. Try --ctc-weight 0 to isolate.")

        if score >= best:
            best = score
            torch.save({"model": model.state_dict(), "config": asdict(cfg)}, args.ckpt)
            print(f"  saved -> {args.ckpt}")
    print("done.")


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = Config(**ck["config"])
    model = PARSeqCTC(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg


def predict(args):
    device = args.device or pick_device()
    model, cfg = load_model(args.ckpt, device)
    for path in args.image:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"{path}: unreadable")
            continue
        x = preprocess(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cfg).unsqueeze(0).to(device)
        text, conf = model.predict(x)
        line = f"{os.path.basename(path)}: {text[0]}  (conf {conf[0]:.3f})"
        if cfg.ctc_weight > 0:
            line += f"  [ctc: {model.predict_ctc(x)[0]}]"
        print(line)


@torch.no_grad()
def diagnose(args):
    """
    Is the decoder reading pixels, or generating plates from a memorised prior?
    Feed it images containing no text at all. A grounded model emits short,
    low-confidence junk. A model that has learned the label distribution emits
    confident, well-formed plates — the failure that makes an AR decoder fit
    the training set and collapse on unseen data.
    """
    device = args.device or pick_device()
    model, cfg = load_model(args.ckpt, device)
    n = 32
    tests = {
        "uniform noise": torch.rand(n, 3, cfg.img_h, cfg.img_w, device=device) * 2 - 1,
        "black":         torch.full((n, 3, cfg.img_h, cfg.img_w), -1.0, device=device),
        "white":         torch.full((n, 3, cfg.img_h, cfg.img_w), 1.0, device=device),
    }
    print("Blank-input test — a grounded model should produce short, "
          "low-confidence, low-variety output.")
    print("Only meaningful on a converged model: an undertrained one emits "
          "nothing and looks 'healthy' for the wrong reason.\n")
    worst = 0.0
    for name, x in tests.items():
        texts, confs = model.predict(x)
        mean_len = float(np.mean([len(t) for t in texts]))
        mean_conf = float(np.mean(confs))
        plausible = float(np.mean([len(t) >= 6 for t in texts]))
        worst = max(worst, plausible * mean_conf)
        print(f"  {name:<14} mean_len={mean_len:4.1f}  mean_conf={mean_conf:.3f}  "
              f"plate-like={plausible:.2f}  e.g. {texts[:3]}")
    print()
    if worst > 0.5:
        print("VERDICT: the decoder is hallucinating from the label prior. Raise "
              "--ctc-weight, add augmentation, or add training data variety.")
    elif worst > 0.2:
        print("VERDICT: borderline. Some prior-driven generation is happening.")
    else:
        print("VERDICT: healthy — output collapses without image evidence.")


# =====================================================================
# 7. CLI
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description="PARSeq + auxiliary CTC OCR")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--root", required=True)
    t.add_argument("--manifest", default="manifest.jsonl")
    t.add_argument("--ckpt", default="parseq_ctc.pt")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--img-h", type=int, default=96)
    t.add_argument("--img-w", type=int, default=128)
    t.add_argument("--max-label-length", type=int, default=12)
    t.add_argument("--ctc-weight", type=float, default=0.2)
    t.add_argument("--stretch", action="store_true",
                   help="disable aspect-preserving letterbox")
    t.add_argument("--num-workers", type=int, default=4)
    t.add_argument("--device", default='mps')
    t.set_defaults(func=train)

    p = sub.add_parser("predict")
    p.add_argument("--ckpt", default="parseq_ctc.pt")
    p.add_argument("--image", nargs="+", required=True)
    p.add_argument("--device", default=None)
    p.set_defaults(func=predict)

    d = sub.add_parser("diagnose")
    d.add_argument("--ckpt", default="parseq_ctc.pt")
    d.add_argument("--device", default=None)
    d.set_defaults(func=diagnose)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


# python parseq_ctc_ocr.py train    --root /Users/rzt/Documents/ocr_env/synthv3_dataset   
# python parseq_ctc_ocr.py predict  --ckpt parseq_ctc.pt --image plate.png
# python parseq_ctc_ocr.py diagnose --ckpt parseq_ctc.pt