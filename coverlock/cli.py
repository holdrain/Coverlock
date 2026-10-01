import argparse
from pathlib import Path

from PIL import Image

from .model import CoverLock


def main():
    parser = argparse.ArgumentParser(prog="coverlock")
    sub = parser.add_subparsers(dest="command", required=True)
    encode = sub.add_parser("encode", help="generate the CoverLock code for one image")
    encode.add_argument("--checkpoint", required=True)
    encode.add_argument("--image", required=True)
    encode.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "encode":
        model = CoverLock.from_checkpoint(args.checkpoint, args.device)
        code = model.encode_pil(Image.open(Path(args.image)))[0].cpu().tolist()
        print("".join("1" if bit else "0" for bit in code))


if __name__ == "__main__":
    main()

