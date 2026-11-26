"""Download the public fly and mouse datasets that fit on this disk, smallest first, into data/raw/datasets/<name>/ (not in git). Nothing is unpacked or run.
Resumes. Stops before a file that would leave under 60 GB free. Several sets are non-commercial: research use only. See reports/Mouse experiment types and datasets.md.
Run from the repo root:  python scripts/fetch_public_datasets.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/datasets"
KEEP_FREE = 60e9
UA = {"User-Agent": "EthoLive-data-fetch"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={**UA, "Accept": "application/json"}), timeout=60) as r:
        return json.load(r), r.headers


def caltech(record, only=None):
    d, _ = get(f"https://data.caltech.edu/api/records/{record}/files")
    return [(e["key"], e["size"], f"https://data.caltech.edu/api/records/{record}/files/{urllib.parse.quote(e['key'])}/content") for e in d["entries"] if only is None or e["key"] in only]


def zenodo(record):
    d, _ = get(f"https://zenodo.org/api/records/{record}")
    return [(f["key"], f["size"], f"https://zenodo.org/api/records/{record}/files/{urllib.parse.quote(f['key'])}/content") for f in d["files"]]


def huggingface(repo):
    url, out = f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true&limit=1000", []
    while url:
        d, h = get(url)
        out += [(f["path"], f.get("size", 0), f"https://huggingface.co/datasets/{repo}/resolve/main/{urllib.parse.quote(f['path'])}") for f in d if f["type"] == "file"]
        link = h.get("Link", "")
        url = link.split(";")[0].strip("<> ") if 'rel="next"' in link else None
    return out


SETS = [("calms21_task3", lambda: caltech("s0vdx-0k302", {"task3_new_behaviors.zip", "readme.md"})),
        ("superanimal_topviewmouse", lambda: zenodo("10618947")),
        ("mars_pose", lambda: caltech("j1ww1-mdc55")),
        ("jax_tracking", lambda: zenodo("5806397")),
        ("crim13_subset", lambda: huggingface("paninski-lab/crim13")),
        ("della_valle_swim", lambda: zenodo("14638257")),
        ("calms21_task1_video", lambda: caltech("s0vdx-0k302", {"task1_videos_mp4.zip"})),
        ("jax_grooming", lambda: zenodo("4646088")),
        ("mabe22", lambda: caltech("rdsa8-rde65")),
        ("ethoscope_joyce_part4", lambda: zenodo("10966461"))]


def main():
    for name, files in SETS:
        try:
            items = files()
        except Exception as exc:
            print(f"SKIP {name}: {type(exc).__name__} {exc}", flush=True)
            continue
        for key, size, url in items:
            path = OUT / name / key
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_file() and path.stat().st_size == size:
                continue
            if shutil.disk_usage(ROOT).free - size < KEEP_FREE:
                print(f"STOP before {name}/{key}: too little disk left", flush=True)
                return
            r = subprocess.run(["curl", "-sSL", "--fail", "--retry", "5", "--retry-delay", "10", "-C", "-", "-A", UA["User-Agent"], "-o", str(path), url])
            if r.returncode:
                print(f"FAILED {name}/{key}: curl {r.returncode}", flush=True)
        print(f"done {name}: {len(items)} files, {sum(i[1] for i in items) / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    sys.exit(main())
