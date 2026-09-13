# ocr — PARSeq + CTC number plate reader

Two files that do one thing: read a licence plate off a crop, including the awkward 2- and 3-line ones that most OCR stacks quietly fail on.

## Why this exists

Off-the-shelf scene-text OCR is built around a single horizontal line of text. Indian plates aren't that — a lot of them are stacked two or three lines, and the crop you get from a detector is loose, tilted, motion-blurred and shot at night. And there's no large labelled set of them, so training starts synthetic and has to survive the jump to real photos.

So this folder is a from-scratch model (no `timm`, no `mmocr`, no pretrained backbone) built around two decisions, both of which have long comments in the code explaining what they're defending against.

### 1. Two heads, one encoder

A plain autoregressive decoder can cheat. Plate text is so structured that the model can learn `p(next char | previous chars)`, drive the loss to the floor, and barely look at the image — then confidently hallucinate plausible-looking plates on anything unseen. That's the failure mode this whole design is fighting.

The fix is an **auxiliary CTC head**: one extra `nn.Linear` on the encoder grid. CTC has to emit characters in raster order, which is only possible if the encoder produces per-character, positionally localised features. It can't cheat with a language prior. It costs almost nothing and it forces the encoder to actually read pixels.

The **PARSeq decoder** then does what CTC can't — attend anywhere in 2D, so multi-line layouts and reading order stop being a problem. It's trained with Permutation Language Modelling so no single scan order becomes a crutch, and it does a bidirectional cloze refinement pass at inference.

```
total_loss = parseq_loss + ctc_weight * ctc_loss     # decode with PARSeq
```

CTC is the training-time conscience; PARSeq is what you actually decode with. `--ctc-weight 0` falls back to plain PARSeq and still trains.

The multi-line trick is the OrigamiNet idea: flatten the grid in raster order and every token of line 1 precedes every token of line 2, so a monotonic CTC path exists. The failure mode (one line firing across two row bands, giving `GJ06GJ06` — CTC only collapses *adjacent* duplicates) is handled by pooling the grid down to `ctc_bands` rows first.

### 2. Fine-tune with replay, never real-only

`parseq_ctc_ocr_finetune.py` exists because the obvious move — take the synthetic-trained checkpoint and fine-tune on real plates — is the textbook recipe for catastrophic forgetting. So it never happens here, not even for a few hundred steps. **Every batch is a blend**: each sample is real with probability `p_real` (0.35), synthetic otherwise. The old distribution stays in the gradient, so forgetting is structurally prevented rather than monitored.

Everything else follows from that:

- LR at ~1/10 the original peak, with warmup — the first few dozen steps at full LR are where the damage happens.
- **Nothing is frozen.** Texture and blur are low-level statistics living in the conv stem; freezing the stem would freeze the exact thing you're trying to adapt.
- Degradation augmentation is **halved for real images** — they already contain the real blur/noise/JPEG; stacking synthetic degradation on top pushes them back off-distribution. Geometric augmentation stays at full strength, since that's what stops a small real set being memorised.
- Two validation numbers, always: real accuracy (improving?) and synthetic accuracy (forgetting?). A checkpoint saves only if real went up **and** synthetic didn't drop more than `--forget-budget`.
- Real accuracy is reported **per source**, because an average hides one dataset quietly poisoning another.
- Within the real pool, sources are weight-sampled and CCPD is balanced across its subsets, so 200k Chinese images can't drown out 2k Indian ones.

## Files

| File | What it is |
|---|---|
| `parseq_ctc_ocr.py` | The model, tokenizer, augmentation, training loop and CLI — all in one file, deliberately. `train` / `predict` / `diagnose`. |
| `parseq_ctc_ocr_finetune.py` | Real-world adaptation on top of a synthetic checkpoint. `eval` / `finetune` / `scan`. Imports the model rather than redefining it, so the two can't drift apart. |
| `parseq_ctc_ocr.ipynb` | The notebook the main script was written in — same code, cell by cell. Kept for interactive poking; the `.py` is the source of truth. |

## Running it

```bash
# train on synthetic
python parseq_ctc_ocr.py train --root DATASET_DIR --manifest manifest.jsonl

# read one plate
python parseq_ctc_ocr.py predict --ckpt parseq_ctc.pt --image plate.png

# is it actually reading pixels, or reciting the label prior?
python parseq_ctc_ocr.py diagnose --ckpt parseq_ctc.pt
```

Manifest is JSONL, one object per line: `{"color": "images/0001.png", "text": "GJ06AB1234"}`

```bash
# baseline before touching anything
python parseq_ctc_ocr_finetune.py eval --ckpt parseq_ctc.pt --synthetic-root synthv3_dataset

# blended fine-tune on real data
python parseq_ctc_ocr_finetune.py finetune --ckpt parseq_ctc.pt --out parseq_ctc_real.pt \
    --synthetic-root synthv3_dataset

# mine unlabelled photos: heads agree -> pseudo-label, heads disagree -> human review
python parseq_ctc_ocr_finetune.py scan --ckpt parseq_ctc_real.pt --images /path/to/unlabelled
```

That last one is the other reason the CTC head earns its keep: two independent decoders on one encoder give you a free agreement signal for auto-labelling.

Requires `torch`, `opencv-python`, `numpy` (and `tqdm`, optionally). Runs on CUDA, MPS or CPU — `pick_device()` sorts it out.

## Data sources it knows how to index

The fine-tune script ingests three real datasets in their native, annoying formats rather than making you convert them:

- **CCPD** — everything encoded in the filename. Note `ccpd_trim`: the leading Chinese province glyph isn't in the charset, so it's dropped from the label *and* cropped off the image, so picture and label still agree. Without the trim you'd be teaching the model "skip the first glyph", which would wreck it on Indian plates.
- **car_licence3** — already cropped, filename stem is the label.
- **indian_plate2** — Pascal-VOC sidecar XML for label and box.
- **pseudo** — whatever `scan` produced.

`looks_like_plate()` guards against real label noise: some VOC files put the class name in `<name>`, which survives charset filtering as `LICENCEPLATE`. Every genuine plate has a digit, so the filter is cheap.

Splits are hashed from the file path (`bucket()`), 90/5/5 — no state file, identical every run, and an image can never drift between train and test.

## Gotchas

- **Label order matters.** The CTC head needs ground truth in raster reading order: a plate reading `GJ06` over `AB1234` must be stored as `GJ06AB1234`. If `ctc_loss` plateaus high while `parseq_loss` falls, your labels aren't in reading order.
- `parseq_ctc_ocr_finetune.py` imports `from simple_parseq import ...` — a leftover from when the main file was called `simple_parseq.py`. Rename the import to `parseq_ctc_ocr` (or symlink) or it exits on startup.
- CTC runs in fp32 always (it's numerically fragile in fp16) and is forced onto CPU under MPS, which has no `aten::_ctc_loss` kernel. `.cpu()` is differentiable and the tensor is tiny, so the copy is free.
- Input is 96×128 letterboxed — taller than typical single-line OCR setups, because a 3-line plate needs vertical room to survive the /8 downsample. Aspect ratio is preserved on purpose: the letterbox shape is itself the cue for how many lines there are.
