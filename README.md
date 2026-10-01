---
library_name: pytorch
tags:
  - image-watermarking
  - image-hashing
  - dinov2
license: other
---

# CoverLock

CoverLock is a watermark-agnostic image-binding module. It derives a stable
256-bit code from an image and XORs that code with a watermark payload. The
watermark encoder and decoder remain unchanged.

This repository provides the final CoverLock S/B/L/G models, training code,
and a small inference API. CoverLock is independent of the watermark backend:
the same image code can wrap the payload of CIN, MBRS, VINE, or another binary
message-based watermark system.

> **License notice:** replace `license: other` above and add a `LICENSE` file
> before making the repository public after choosing the project license. DINOv2 and any
> downstream watermark implementation retain their own licenses.

## Checkpoints

| Name | Frozen backbone | Layer | Training step | File |
|---|---|---:|---:|---|
| CoverLock-S | DINOv2 ViT-S/14 | 11 | 205,000 | `weights/coverlock_s.pt` |
| CoverLock-B | DINOv2 ViT-B/14 | 11 | 200,000 | `weights/coverlock_b.pt` |
| CoverLock-L | DINOv2 ViT-L/14 | 23 | 200,000 | `weights/coverlock_l.pt` |
| CoverLock-G | DINOv2 ViT-G/14 | 39 | 200,000 | `weights/coverlock_g.pt` |

The exact 200k S checkpoint was not retained by the original checkpoint
rotation. The nearest retained S checkpoint (205k), used in the paper's scale
comparison, is provided and labeled with its exact step.

The checkpoints contain only the small CoverLock head. The frozen DINOv2
backbone is downloaded through PyTorch Hub on first use.

## Installation

```bash
python -m pip install -e .
```

Python 3.10+ and PyTorch 2.1+ are recommended.

## Generate an image code

```bash
coverlock encode \
  --checkpoint weights/coverlock_s.pt \
  --image path/to/image.jpg
```

Use the code as an XOR mask around an existing watermark system:

```python
import torch
from coverlock import CoverLock

model = CoverLock.from_checkpoint("weights/coverlock_s.pt", device="cuda")
image_code = model.encode_pil(image)           # bool tensor, shape [1, 256]
embedded_payload = torch.logical_xor(message, image_code)

# decoded_payload is produced by the unchanged watermark decoder.
recovered_message = torch.logical_xor(decoded_payload, image_code)
```

For shorter messages, select the same deterministic bit indices at embedding
and verification time. Do not silently truncate differently across systems.

## Training

The training command is shared by all scales:

```bash
bash scripts/train.sh s /path/to/training/images outputs/coverlock-s
```

Replace `s` with `b`, `l`, or `g`. The defaults reproduce the reference setup:
224 px images, 256-bit codes, global batch size 256, bfloat16, and 200k steps.
Override portable settings with environment variables, for example:

```bash
GPU_IDS=0,1 NPROC=2 STEPS=10000 BATCH_SIZE=128 \
  bash scripts/train.sh s /path/to/images outputs/debug
```

The dataset argument is any directory recursively containing JPG, PNG, WebP,
or BMP images. Training uses a frozen DINOv2 backbone. Only the CoverLock head
is optimized.

## Checkpoint format

The checkpoints contain:

```text
format_version, model_type, variant, step, config, state_dict
```

They contain no optimizer, dataset path, run directory, hostname, or other
machine-specific state.

## Repository contents

```text
coverlock/       model, inference API, losses, and trainer
scripts/         portable S/B/L/G training launcher
tests/           lightweight model-head tests
weights/         inference-only checkpoints
```

The repository deliberately excludes optimizer states, training logs,
intermediate checkpoints, dataset paths, and machine-specific configuration.

## Citation

Please add the paper's final BibTeX entry here when it becomes available.

## Publishing

The four checkpoint files are each below GitHub's per-file size limit, so this
repository can be pushed with standard Git:

```bash
git init
git add .
git commit -m "Add CoverLock S/B/L/G"
```

Then create the GitHub and Hugging Face repositories and push this directory.
No credentials or remote repository names are embedded here.
