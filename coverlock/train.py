"""Distributed trainer used for all CoverLock scales."""

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from .data import ImagePairs
from .losses import contrastive_loss, hash_loss
from .model import CoverLockHead, FrozenDINOExtractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("s", "b", "l", "g"), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--message-bits", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--hash-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if args.precision == "bf16" and device.type != "cuda":
        parser.error("bf16 requires CUDA; use --precision fp32")
    if args.batch_size % world_size:
        parser.error("batch size must be divisible by world size")
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    if distributed:
        dist.barrier()
    dataset = ImagePairs(args.data, args.image_size, args.seed)
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed, drop_last=True) if distributed else None
    loader = DataLoader(dataset, batch_size=args.batch_size // world_size,
                        shuffle=sampler is None, sampler=sampler, drop_last=True,
                        num_workers=args.num_workers, pin_memory=True)
    iterator = iter(loader)
    extractor = FrozenDINOExtractor(args.backbone, args.layer).to(device).eval()
    head = CoverLockHead(extractor.output_dim, args.message_bits, args.hidden,
                         args.embedding_dim, args.hash_seed, "orthogonal").to(device)
    raw_head = head
    if distributed:
        head = DistributedDataParallel(head, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    weights = dict(stability=2.0, independence=1.0, balance=5.0,
                   variance=5.0, decorrelation=0.05, binary=0.05)
    config = dict(architecture="contrastive_projection", backbone=args.backbone,
                  layer=args.layer, image_size=args.image_size,
                  message_bits=args.message_bits, hidden=args.hidden,
                  embedding_dim=args.embedding_dim, hash_seed=args.hash_seed,
                  projection_mode="orthogonal")
    if rank == 0:
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    progress = tqdm(range(1, args.steps + 1), disable=rank != 0)
    epoch = 0
    for step in progress:
        try:
            clean, attacked = next(iterator)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            iterator = iter(loader)
            clean, attacked = next(iterator)
        clean, attacked = clean.to(device), attacked.to(device)
        with torch.autocast("cuda", torch.bfloat16, enabled=args.precision == "bf16"):
            with torch.no_grad():
                clean_features = extractor(clean)
                attacked_features = extractor(attacked)
            clean_logits, clean_embedding = head(clean_features, return_embeddings=True)
            attacked_logits, attacked_embedding = head(attacked_features, return_embeddings=True)
        clean_logits, attacked_logits = clean_logits.float(), attacked_logits.float()
        clean_embedding, attacked_embedding = clean_embedding.float(), attacked_embedding.float()
        if distributed:
            clean_logits = torch.cat(all_gather(clean_logits))
            attacked_logits = torch.cat(all_gather(attacked_logits))
            clean_embedding = torch.cat(all_gather(clean_embedding))
            attacked_embedding = torch.cat(all_gather(attacked_embedding))
        regularization, _ = hash_loss(clean_logits, attacked_logits, weights)
        loss = contrastive_loss(clean_embedding, attacked_embedding, args.temperature) + 0.25 * regularization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if rank == 0:
            progress.set_postfix(loss=f"{loss.item():.4f}")
            if step % args.save_every == 0 or step == args.steps:
                torch.save(dict(format_version=1, model_type="coverlock",
                                variant=args.variant, step=step, config=config,
                                state_dict=raw_head.state_dict()),
                           output / f"coverlock_{args.variant}_step{step}.pt")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
