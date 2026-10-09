"""Download pinned public CSV copies and retain the exact retrieval provenance.

The Kaggle competition endpoint requires authentication. This script accesses
the separately published public dataset, without attempting to bypass that gate.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

MIRROR = "megancrenshaw/home-credit-default-risk"
VERSION = 2
FILES = {
    "application_train.csv": "home-credit-default-risk/application_train.csv",
    "application_test.csv": "home-credit-default-risk/application_test.csv",
    "sample_submission.csv": "home-credit-default-risk/sample_submission.csv",
    "bureau.csv": "bureau.csv",
    "HomeCredit_columns_description.csv": "HomeCredit_columns_description.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "HomeCreditAcademicReplication/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw = root / "data/raw"
    provenance = root / "data/provenance"
    raw.mkdir(parents=True, exist_ok=True)
    provenance.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "data/source_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if existing.get("dataset_version") == VERSION and set(existing.get("files", {})) == set(FILES):
            if all((raw/name).is_file()
                   and (raw/name).stat().st_size == details["bytes"]
                   and sha256(raw/name) == details["sha256"]
                   for name, details in existing["files"].items()):
                print("All pinned inputs already exist and hashes match; preserving original provenance.", flush=True)
                return
    archives = provenance / "downloads"
    archives.mkdir(parents=True, exist_ok=True)
    metadata_url = f"https://www.kaggle.com/api/v1/datasets/view/{MIRROR}"
    files_url = f"https://www.kaggle.com/api/v1/datasets/list/{MIRROR}"
    fetch(metadata_url, provenance / "kaggle_public_metadata.json")
    fetch(files_url, provenance / "kaggle_public_files.json")
    metadata = json.loads((provenance / "kaggle_public_metadata.json").read_text())
    if metadata["currentVersionNumber"] != VERSION:
        raise RuntimeError("Public dataset version changed; inspect its versioned file listing before proceeding.")
    listing = json.loads((provenance / "kaggle_public_files.json").read_text())
    expected = {entry["name"]: entry["totalBytes"] for entry in listing["datasetFiles"]}
    manifest = {
        "original_source": "https://www.kaggle.com/competitions/home-credit-default-risk/data",
        "download_source": f"https://www.kaggle.com/datasets/{MIRROR}",
        "download_source_kind": "Separately published public Kaggle copy, not an authenticated competition download",
        "dataset_version": VERSION,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_download_status": "HTTP 401 without authentication; no Kaggle credentials were available",
        "identity_limit": "Original official file hashes are unavailable. File identity is checked against a second public copy and original published schema/definitions.",
        "files": {},
    }
    for local_name, remote_name in FILES.items():
        destination = raw / local_name
        archive_details = {}
        url = f"https://www.kaggle.com/api/v1/datasets/download/{MIRROR}/{urllib.parse.quote(remote_name, safe='')}?datasetVersionNumber={VERSION}"
        if destination.exists():
            if destination.stat().st_size != expected[remote_name]:
                raise RuntimeError(f"Existing file {destination} has an unexpected size; do not overwrite it silently.")
            print(f"Already present: {local_name}", flush=True)
        else:
            partial = raw / (local_name + ".part")
            print(f"Downloading {local_name}: expected {expected[remote_name]:,} bytes", flush=True)
            # Kaggle returns large individual files as ZIP archives, small ones as raw CSV.
            if not partial.exists():
                fetch(url, partial)
            if zipfile.is_zipfile(partial):
                with zipfile.ZipFile(partial) as archive:
                    entries = [entry for entry in archive.infolist() if not entry.is_dir()]
                    if len(entries) != 1 or Path(entries[0].filename).name != local_name:
                        raise RuntimeError(f"Unexpected archive contents for {local_name}")
                    if entries[0].file_size != expected[remote_name]:
                        raise RuntimeError(f"Uncompressed size differs from Kaggle metadata: {local_name}")
                    extracted = raw / (local_name + ".extracted.part")
                    with archive.open(entries[0]) as stream, extracted.open("wb") as output:
                        shutil.copyfileobj(stream, output, length=1024 * 1024)
                if extracted.stat().st_size != expected[remote_name]:
                    raise RuntimeError(f"Incomplete extraction: {local_name}")
                archive_details = {"archive_bytes": partial.stat().st_size, "archive_sha256": sha256(partial)}
                extracted.rename(destination)
                partial.rename(archives / (local_name + ".zip"))
            else:
                if partial.stat().st_size != expected[remote_name]:
                    raise RuntimeError(f"Downloaded file size differs from Kaggle metadata: {local_name}")
                partial.rename(destination)
        manifest["files"][local_name] = {
            "remote_name": remote_name, "download_url": url,
            "bytes": destination.stat().st_size, "sha256": sha256(destination),
            **archive_details,
        }
        (root / "data/source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Verified size and recorded SHA-256: {local_name}", flush=True)
    print("Download complete. Run data-quality checks before training.", flush=True)


if __name__ == "__main__":
    main()
