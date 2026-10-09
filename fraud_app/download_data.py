"""Download the public benchmark mirror used by TensorFlow's tutorial.

Does not upload local data. Raw data is kept in the ignored data directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import httpx

URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
EXPECTED_MD5 = "e90efcb83d69faf99fcab8b0255024de"


def download(destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        with destination.open("rb") as source:
            checksum = hashlib.file_digest(source, "md5").hexdigest()
        if checksum != EXPECTED_MD5:
            raise ValueError("Existing file differs from the verified mirror. Use another destination; it was not overwritten.")
        print("Verified existing benchmark:", destination)
        return
    temporary = destination.with_suffix(".csv.part")
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    size = 0
    try:
        with httpx.stream("GET", URL, timeout=120, follow_redirects=False) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_bytes(1024 * 1024):
                    size += len(chunk)
                    if size > 200_000_000:
                        raise ValueError("Unexpected download size")
                    output.write(chunk)
                    md5.update(chunk)
                    sha256.update(chunk)
        if md5.hexdigest() != EXPECTED_MD5:
            raise ValueError("Mirror checksum mismatch; dataset not published")
        os.replace(temporary, destination)
        print(json.dumps({"url": URL, "bytes": size, "sha256": sha256.hexdigest(), "path": str(destination)}, indent=2))
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("data/creditcard.csv"))
    download(parser.parse_args().destination)
