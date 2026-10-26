import datetime as dt
import io

import pytest

from mkuav.storage import build_manifest, fetch_manifest, presign_get

NOW = dt.datetime(2013, 5, 24, tzinfo=dt.timezone.utc)


def test_presign_matches_aws_sigv4_reference_vector():
    # Example from the AWS S3 "Authenticating Requests: Using Query Parameters" docs.
    url = presign_get("https://examplebucket.s3.amazonaws.com", "", "test.txt",
                      "AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                      expires=86400, now=NOW)
    assert url.endswith(
        "X-Amz-Signature=aeeed9bbccd4d02ee5c0109b86d86835f995330da4c265957d157751f604d404")


def test_presign_path_style_for_minio():
    url = presign_get("http://localhost:9000", "mk-uav-eval", "subsets/data-v2/a b.jpg",
                      "ak", "sk", expires=600, now=NOW)
    assert url.startswith("http://localhost:9000/mk-uav-eval/subsets/data-v2/a%20b.jpg?")
    assert "X-Amz-Expires=600" in url and "X-Amz-Credential=ak%2F20130524" in url


@pytest.mark.parametrize("expires", [0, 7 * 24 * 3600 + 1])
def test_presign_rejects_out_of_range_expiry(expires):
    with pytest.raises(ValueError):
        presign_get("http://localhost:9000", "b", "k", "ak", "sk", expires=expires)


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_fetch_manifest_writes_files(tmp_path):
    manifest = build_manifest({"icg/x.png": "k1", "visdrone_val/y.txt": "k2"},
                              lambda k: f"http://minio/{k}", 3600, now=NOW)
    count = fetch_manifest(manifest, tmp_path, opener=lambda u, timeout: _Resp(u.encode()),
                           now=NOW)
    assert count == 2
    assert (tmp_path / "icg/x.png").read_bytes() == b"http://minio/k1"


def test_fetch_manifest_rejects_expired_and_traversal(tmp_path):
    manifest = build_manifest({"a": "k"}, lambda k: "http://m/k", 60, now=NOW)
    with pytest.raises(RuntimeError):
        fetch_manifest(manifest, tmp_path, now=NOW + dt.timedelta(minutes=2))
    bad = build_manifest({"../evil": "k"}, lambda k: "http://m/k", 60, now=NOW)
    with pytest.raises(ValueError):
        fetch_manifest(bad, tmp_path, now=NOW)
