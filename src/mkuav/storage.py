"""MinIO/S3 dataset storage: SigV4 presigned GET URLs and manifest download.

A maintainer with MinIO credentials signs every object of a dataset version
into a JSON manifest of short-lived URLs. CI downloads the manifest and the
objects it lists without holding any credentials. Stdlib only, so the CI
fetch step needs neither boto3 nor DVC.
"""
import datetime as dt
import hashlib
import hmac
import json
import urllib.parse
import urllib.request
from pathlib import Path

MAX_EXPIRES = 7 * 24 * 3600  # SigV4 hard limit


def _quote(value: str, safe: str = "-_.~") -> str:
    return urllib.parse.quote(value, safe=safe)


def presign_get(endpoint: str, bucket: str, key: str, access_key: str, secret_key: str,
                expires: int = 900, region: str = "us-east-1",
                now: dt.datetime | None = None) -> str:
    """Return a path-style SigV4 presigned GET URL valid for `expires` seconds."""
    if not 1 <= expires <= MAX_EXPIRES:
        raise ValueError(f"expires must be in [1, {MAX_EXPIRES}], got {expires}")
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"endpoint must be an http(s) URL, got {endpoint!r}")
    now = (now or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    datestamp = amz_date[:8]
    scope = f"{datestamp}/{region}/s3/aws4_request"
    # Path-style for MinIO; an empty bucket means virtual-host style (bucket in endpoint).
    prefix = "/" + _quote(bucket) if bucket else ""
    path = prefix + "/" + _quote(key.lstrip("/"), safe="-_.~/")
    query = {
        "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
        "X-Amz-Credential": f"{access_key}/{scope}",
        "X-Amz-Date": amz_date,
        "X-Amz-Expires": str(expires),
        "X-Amz-SignedHeaders": "host",
    }
    canonical_query = "&".join(f"{_quote(k)}={_quote(v)}" for k, v in sorted(query.items()))
    canonical_request = "\n".join([
        "GET", path, canonical_query, f"host:{parsed.netloc}\n", "host", "UNSIGNED-PAYLOAD",
    ])
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256", amz_date, scope,
        hashlib.sha256(canonical_request.encode()).hexdigest(),
    ])
    key_bytes = ("AWS4" + secret_key).encode()
    for part in (datestamp, region, "s3", "aws4_request"):
        key_bytes = hmac.new(key_bytes, part.encode(), hashlib.sha256).digest()
    signature = hmac.new(key_bytes, string_to_sign.encode(), hashlib.sha256).hexdigest()
    return (f"{parsed.scheme}://{parsed.netloc}{path}?{canonical_query}"
            f"&X-Amz-Signature={signature}")


def build_manifest(files: dict[str, str], sign, expires: int,
                   now: dt.datetime | None = None) -> dict:
    """Build a manifest from {relative_path: object_key}; `sign(key)` returns a URL."""
    now = (now or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
    expires_at = now + dt.timedelta(seconds=expires)
    return {
        "expires_at": expires_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": {rel: sign(key) for rel, key in sorted(files.items())},
    }


def _safe_target(dest: Path, rel: str) -> Path:
    target = (dest / rel).resolve()
    if not target.is_relative_to(dest.resolve()):
        raise ValueError(f"manifest path escapes destination: {rel!r}")
    return target


def fetch_manifest(manifest: dict, dest: Path, opener=urllib.request.urlopen,
                   now: dt.datetime | None = None) -> int:
    """Download every manifest entry under `dest`. Returns the file count."""
    now = (now or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
    expires_at = dt.datetime.strptime(manifest["expires_at"], "%Y-%m-%dT%H:%M:%SZ")
    if expires_at.replace(tzinfo=dt.timezone.utc) <= now:
        raise RuntimeError(f"manifest expired at {manifest['expires_at']}; re-sign it")
    files = manifest["files"]
    targets = {rel: _safe_target(dest, rel) for rel in files}
    for rel, url in files.items():
        if urllib.parse.urlsplit(url).scheme not in ("http", "https"):
            raise ValueError(f"refusing non-http URL for {rel!r}")
        targets[rel].parent.mkdir(parents=True, exist_ok=True)
        with opener(url, timeout=60) as resp:
            targets[rel].write_bytes(resp.read())
    return len(files)


def load_manifest(source: str) -> dict:
    """Load a manifest from a local path or an http(s) URL."""
    if urllib.parse.urlsplit(source).scheme in ("http", "https"):
        with urllib.request.urlopen(source, timeout=60) as resp:
            return json.load(resp)
    return json.loads(Path(source).read_text())
