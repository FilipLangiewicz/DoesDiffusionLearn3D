"""Download the output files of a Kaggle job to disk, with resume.

The Kaggle CLI reads each output file into memory before it writes it. A file
of 10 GB thus fails on a computer with little RAM. This script asks the API for
the file addresses and gives them to curl. Curl writes to disk and continues a
broken download from its last byte. A new address comes from the API before
each try, because the addresses are valid only for a short time.

Usage: python tools/kaggle/kfetch.py OWNER/SLUG OUT_DIR [--pattern REGEX]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


def list_outputs(kernel_id: str) -> list[tuple[str, str]]:
    """Return (file name, download address) for all output files of a job."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    owner, slug = kernel_id.split("/")
    api = KaggleApi()
    api.authenticate()
    files, token = [], None
    with api.build_kaggle_client() as kaggle:
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name = owner
            request.kernel_slug = slug
            if token:
                request.page_token = token
            response = kaggle.kernels.kernels_api_client.list_kernel_session_output(request)
            files += [(f.file_name, f.url) for f in response.files or []]
            token = response.next_page_token
            if not token:
                return files


def fetch(kernel_id: str, out_dir: Path, pattern: str | None = None, tries: int = 30) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [n for n, _ in list_outputs(kernel_id) if not pattern or re.search(pattern, n)]
    done = []
    for name in names:
        target = out_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(tries):
            url = dict(list_outputs(kernel_id))[name]
            # -C - continues from the size of the partial file; -f fails on HTTP errors.
            code = subprocess.run(
                ["curl", "-sSfL", "-C", "-", "--retry", "5", "--retry-delay", "10", "-o", str(target), url]
            ).returncode
            if code in (0, 33):  # 33: the server cannot continue, the file is already complete
                break
            print(f"[kfetch] {name}: curl exit {code}, try {attempt + 1}", flush=True)
            time.sleep(min(30 * (attempt + 1), 300))
        else:
            raise SystemExit(f"[kfetch] gave up: {name}")
        print(f"[kfetch] {name} {target.stat().st_size / 1e9:.2f} GB", flush=True)
        done.append(target)
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel_id")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--pattern", default=None)
    args = parser.parse_args()
    fetch(args.kernel_id, args.out_dir, args.pattern)


if __name__ == "__main__":
    sys.exit(main())
