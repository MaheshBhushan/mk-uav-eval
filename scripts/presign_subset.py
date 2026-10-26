"""Sign the CI subset stored in MinIO into a manifest of short-lived URLs.

Run by a maintainer who holds MinIO credentials. CI consumes the manifest
with scripts/fetch_presigned.py and never sees the credentials.

Env: MINIO_ENDPOINT (e.g. http://localhost:9000), MINIO_ACCESS_KEY,
MINIO_SECRET_KEY, optional MINIO_BUCKET (default mk-uav-eval).

Usage: uv run python scripts/presign_subset.py --version data-v2 \
           --expires 3600 --out subset-manifest.json
Objects are expected at s3://$MINIO_BUCKET/subsets/<version>/<path>, mirroring
data/ci_subset/. Upload them with `mc mirror data/ci_subset minio/<bucket>/subsets/<version>`.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mkuav.storage import build_manifest, presign_get  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", default="data-v2")
    parser.add_argument("--subset-dir", type=Path, default=ROOT / "data" / "ci_subset")
    parser.add_argument("--expires", type=int, default=3600)
    parser.add_argument("--out", type=Path, default=Path("subset-manifest.json"))
    args = parser.parse_args()

    missing = [v for v in ("MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY")
               if not os.environ.get(v)]
    if missing:
        print(f"missing env: {', '.join(missing)}", file=sys.stderr)
        return 2
    bucket = os.environ.get("MINIO_BUCKET", "mk-uav-eval")
    files = {
        p.relative_to(args.subset_dir).as_posix():
            f"subsets/{args.version}/{p.relative_to(args.subset_dir).as_posix()}"
        for p in args.subset_dir.rglob("*") if p.is_file()
    }

    def sign(key: str) -> str:
        return presign_get(os.environ["MINIO_ENDPOINT"], bucket, key,
                           os.environ["MINIO_ACCESS_KEY"], os.environ["MINIO_SECRET_KEY"],
                           expires=args.expires)

    manifest = build_manifest(files, sign, args.expires)
    args.out.write_text(json.dumps(manifest, indent=1))
    print(f"signed {len(files)} objects, expires {manifest['expires_at']} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
