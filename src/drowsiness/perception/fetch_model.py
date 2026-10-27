"""Download the MediaPipe Face Landmarker model and verify it.

    python -m drowsiness.perception.fetch_model [--dest models/face_landmarker.task]

The model is gitignored (never commit weights); every machine fetches its own copy.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from collections.abc import Sequence
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_SIZE = 3_758_596
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
DEFAULT_DEST = Path("models/face_landmarker.task")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_model(dest: Path = DEFAULT_DEST, url: str = MODEL_URL) -> Path:
    """Download to `dest` unless a verified copy is already there. Raises on mismatch."""
    if dest.is_file() and sha256_of(dest) == MODEL_SHA256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as f:
            while chunk := response.read(1 << 16):
                f.write(chunk)
        size, digest = tmp.stat().st_size, sha256_of(tmp)
        if size != MODEL_SIZE or digest != MODEL_SHA256:
            raise OSError(f"downloaded model failed verification (size {size}, sha256 {digest})")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the Face Landmarker model.")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args(argv)
    try:
        path = fetch_model(args.dest)
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"model ready: {path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
