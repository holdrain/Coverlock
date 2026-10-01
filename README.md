# CoverLock: Content-Bound Watermarking with Keyed Orthogonal Hashing

Official PyTorch implementation of **CoverLock**, a watermark-agnostic module
for binding a binary watermark payload to its cover image.

CoverLock extracts a robust, image-dependent bit string with a frozen DINOv2
backbone and a lightweight learned head. The bit string masks the payload with
XOR before watermark embedding and unmasks the decoded payload during
verification. This design leaves the underlying watermark encoder and decoder
unchanged and can be used with binary message-based systems such as CIN, MBRS,
and VINE.

> The paper and citation will be linked here when they are publicly available.

## Method

Given an image `x` and a binary payload `m`, CoverLock computes an image code
`b(x)` and embeds the bound payload

```text
m_bound = m XOR b(x).
```

After the watermarked image is received, the watermark decoder recovers the
bound payload and CoverLock recomputes the code from the received image:

```text
m_recovered = m_bound_decoded XOR b(x_received).
```

The CoverLock head pools intermediate DINOv2 patch features, maps them to a
normalized embedding, and applies a fixed keyed orthogonal projection. During
training, the DINOv2 backbone remains frozen; only the lightweight head is
optimized.

## Model Zoo

| Model | Frozen backbone | Layer | Training step | Checkpoint |
|---|---|---:|---:|---|
| CoverLock-S | DINOv2 ViT-S/14 | 11 | 200k | [`coverlock_s.pt`](weights/coverlock_s.pt) |
| CoverLock-B | DINOv2 ViT-B/14 | 11 | 200k | [`coverlock_b.pt`](weights/coverlock_b.pt) |
| CoverLock-L | DINOv2 ViT-L/14 | 23 | 200k | [`coverlock_l.pt`](weights/coverlock_l.pt) |
| CoverLock-G | DINOv2 ViT-G/14 | 39 | 200k | [`coverlock_g.pt`](weights/coverlock_g.pt) |

All released checkpoints produce 256-bit codes and contain only the CoverLock
head. DINOv2 weights are downloaded through PyTorch Hub on first use.

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/holdrain/coverlock.git
cd coverlock
python -m pip install -e .
```

Python 3.10 or later and PyTorch 2.1 or later are recommended. A CUDA-capable
GPU is recommended for the larger backbones.

## Quick Start

Generate the 256-bit CoverLock code for an image:

```bash
coverlock encode \
  --checkpoint weights/coverlock_s.pt \
  --image path/to/image.jpg \
  --device cuda
```

The command prints the code as a string of `0` and `1` values. Use
`--device cpu` when CUDA is unavailable.

The same operation is available through the Python API:

```python
from PIL import Image
from coverlock import CoverLock

model = CoverLock.from_checkpoint(
    "weights/coverlock_s.pt",
    device="cuda",
)
image = Image.open("path/to/image.jpg")
image_code = model.encode_pil(image)  # bool tensor with shape [1, 256]
```

## Watermark Integration

CoverLock wraps the payload passed to an existing binary watermark system:

```python
import torch

# `message` is the original bool payload.
image_code = model.encode_pil(cover_image)
bound_payload = torch.logical_xor(message, image_code)

# Embed `bound_payload` with the unchanged watermark encoder.
# Decode the possibly distorted watermarked image with the unchanged decoder.
received_code = model.encode_pil(received_image)
recovered_message = torch.logical_xor(decoded_payload, received_code)
```

For payloads shorter than 256 bits, select a fixed subset of code bits and use
the same indices during embedding and verification.

## Training

The launcher supports all four model scales:

```bash
bash scripts/train.sh {s|b|l|g} DATA_DIR OUTPUT_DIR
```

For example, train CoverLock-S with the reference defaults:

```bash
bash scripts/train.sh s /path/to/training/images outputs/coverlock-s
```

`DATA_DIR` is searched recursively for JPG, JPEG, PNG, WebP, and BMP images.
The default configuration uses 224 x 224 images, 256-bit codes, a global batch
size of 256, bfloat16 precision, four GPUs, and 200,000 optimization steps.

Portable settings can be overridden with environment variables:

```bash
GPU_IDS=0,1 NPROC=2 BATCH_SIZE=128 STEPS=10000 \
  bash scripts/train.sh s /path/to/training/images outputs/coverlock-s-debug
```

Set `PYTHON_BIN` to select a specific Python interpreter. The output directory
must not already exist. Checkpoints are saved every 5,000 steps by default.

## Testing

Run the lightweight head and loss tests without downloading a DINOv2 backbone:

```bash
python -m pip install pytest
pytest -q
```

## Checkpoint Format

Each checkpoint is a PyTorch dictionary with the following fields:

```text
format_version, model_type, variant, step, config, state_dict
```

Released files contain the CoverLock head and its inference configuration. They
do not contain optimizer state, training data paths, logs, hostnames, or other
machine-specific state.

## Repository Structure

```text
coverlock/       model, inference API, losses, and distributed trainer
scripts/         training launcher for CoverLock-S/B/L/G
tests/           lightweight unit tests
weights/         released inference checkpoints
```

## Citation

The BibTeX entry will be added after the paper is publicly released.

## Acknowledgements

CoverLock uses [DINOv2](https://github.com/facebookresearch/dinov2) as its frozen
visual backbone. Please follow the licenses and citation requirements of
DINOv2 and the watermark backend used in your application.
