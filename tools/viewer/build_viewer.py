"""Build the light viewer: one HTML file with the sweep and patch data inside.

The inputs are the JSON files of scripts/experiments/sweep_package.py. The
output is a single HTML file that works offline and can also be published.

Usage: python tools/viewer/build_viewer.py --sweep sweep.json --patch patch.json --out viewer.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent / "template.html"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, default=None)
    parser.add_argument("--patch", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = {}
    for key, path in (("sweep", args.sweep), ("patch", args.patch)):
        if path is not None:
            data[key] = json.loads(path.read_text())
    # A "</" inside the JSON would close the script element of the page.
    text = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("__DATA__", text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"{args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
