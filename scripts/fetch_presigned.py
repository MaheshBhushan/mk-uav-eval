"""Download a dataset version from a presigned-URL manifest. Needs no credentials.

Usage: uv run python scripts/fetch_presigned.py <manifest path or URL> --dest data/
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mkuav.storage import fetch_manifest, load_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest")
    parser.add_argument("--dest", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    count = fetch_manifest(load_manifest(args.manifest), args.dest)
    print(f"fetched {count} files into {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
