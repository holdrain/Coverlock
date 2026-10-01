# CoverLock

Official implementation of **CoverLock** from the paper
[*Residual Transferability in Neural Image Watermarking*](https://arxiv.org/abs/2609.32241).

CoverLock is a plug-and-play defense that binds a binary watermark payload to
its cover image without modifying the watermark encoder or decoder. It derives
a robust image-dependent code and applies it to the payload through XOR.

## Installation

```bash
git clone https://github.com/holdrain/coverlock.git
cd coverlock

conda create -n coverlock python=3.10 -y
conda activate coverlock
pip install -e .
```

The first inference or training run downloads the frozen DINOv2 backbone
through PyTorch Hub.

## Inference with a Watermarking Method

CoverLock can be placed around any watermarking method that embeds and decodes
a binary message. Compute the image code, XOR it with the message before
embedding, and repeat the operation after decoding:

```python
import torch
from coverlock import CoverLock

coverlock = CoverLock.from_checkpoint(
    "weights/coverlock_s.pt",
    device="cuda",
)

# Sender: bind the payload to the cover image.
cover_code = coverlock.encode_pil(cover_image)
bound_message = torch.logical_xor(message.bool(), cover_code)
watermarked_image = watermark_encoder(cover_image, bound_message)

# Receiver: let the watermark decoder produce binary message bits, then recover
# the original payload.
decoded_message = watermark_decoder(received_image)
received_code = coverlock.encode_pil(received_image)
recovered_message = torch.logical_xor(decoded_message, received_code)
```

`message`, `bound_message`, and `decoded_message` must use the same bit length.
For watermarking methods with fewer than 256 message bits, use the same fixed
subset of CoverLock bits at both the sender and receiver.

To generate the image code directly from the command line:

```bash
coverlock encode \
  --checkpoint weights/coverlock_s.pt \
  --image path/to/image.jpg \
  --device cuda
```

## Training

Place the training images under one directory. Images are discovered
recursively, and JPG, JPEG, PNG, WebP, and BMP files are supported.

Start training with:

```bash
bash scripts/train.sh {s|b|l|g} DATA_DIR OUTPUT_DIR
```

For example:

```bash
bash scripts/train.sh s /path/to/training/images outputs/coverlock-s
```

The default configuration trains for 200k steps with four GPUs and a global
batch size of 256. These settings can be overridden when needed:

```bash
GPU_IDS=0,1 NPROC=2 BATCH_SIZE=128 STEPS=200000 \
  bash scripts/train.sh s /path/to/training/images outputs/coverlock-s
```
