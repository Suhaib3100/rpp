#!/usr/bin/env python
"""Acquire the NEU-CLS steel surface defect database.

The canonical distribution route for NEU-CLS is a Kaggle dataset (and a set of
Google Drive / Baidu mirrors linked from the authors' lab page, whose original
URL is widely reported dead). Kaggle requires an API token, which this
environment does not have, so we pull a public Hugging Face parquet mirror and
then *verify it against the published description of NEU-CLS* rather than
trusting it blindly:

    1800 images, 6 classes, 300 images per class, 200x200 grayscale.

Song & Yan (2013) is the originating publication for the database. The class
index -> class name mapping is NOT carried by the mirror (labels are bare
int64), so it is resolved separately and recorded in data/raw/label_map.json;
see scripts/resolve_label_names.py.

Usage:
    python scripts/download_dataset.py
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

REPO_ID = "newguyme/neu_cls"
FILES = ["data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"]

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
IMAGES = RAW / "images"

# Published specification of NEU-CLS, used as an acceptance test on the mirror.
EXPECTED_TOTAL = 1800
EXPECTED_CLASSES = 6
EXPECTED_PER_CLASS = 300
EXPECTED_SIZE = (200, 200)


def main() -> int:
    IMAGES.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    label_counter: Counter = Counter()
    sizes: Counter = Counter()
    modes: Counter = Counter()
    digests: dict[str, str] = {}

    for fname in FILES:
        split_hint = "train" if "train" in fname else "test"
        print(f"[download] {REPO_ID}:{fname}")
        local = hf_hub_download(repo_id=REPO_ID, filename=fname, repo_type="dataset")
        table = pq.read_table(local)
        print(f"[read]     {fname}: {table.num_rows} rows, columns={table.column_names}")

        images = table.column("image").to_pylist()
        labels = table.column("label").to_pylist()

        for idx, (img_struct, label) in enumerate(zip(images, labels)):
            # HF image feature serialises as {'bytes': ..., 'path': ...}
            raw = img_struct["bytes"] if isinstance(img_struct, dict) else img_struct
            img = Image.open(io.BytesIO(raw))
            img.load()
            sizes[img.size] += 1
            modes[img.mode] += 1
            label_counter[label] += 1

            # Content hash lets us detect exact-duplicate images and leakage
            # between the mirror's own train/test split.
            sha = hashlib.sha256(img.convert("L").tobytes()).hexdigest()

            name = f"{split_hint}_{idx:05d}_cls{label}.png"
            out = IMAGES / name
            if not out.exists():
                img.convert("L").save(out, format="PNG", optimize=True)

            records.append(
                {
                    "filename": name,
                    "label_index": int(label),
                    "mirror_split": split_hint,
                    "width": img.size[0],
                    "height": img.size[1],
                    "mode": img.mode,
                    "sha256_pixels": sha,
                }
            )
            digests.setdefault(sha, name)

    # ---------------- verification against the published spec ----------------
    print("\n=== NEU-CLS mirror verification ===")
    checks: list[tuple[str, bool, str]] = []

    total = len(records)
    checks.append((f"total images == {EXPECTED_TOTAL}", total == EXPECTED_TOTAL, str(total)))

    n_classes = len(label_counter)
    checks.append(
        (f"distinct classes == {EXPECTED_CLASSES}", n_classes == EXPECTED_CLASSES, str(n_classes))
    )

    per_class_ok = all(c == EXPECTED_PER_CLASS for c in label_counter.values())
    checks.append(
        (
            f"{EXPECTED_PER_CLASS} images per class",
            per_class_ok,
            str(dict(sorted(label_counter.items()))),
        )
    )

    size_ok = set(sizes) == {EXPECTED_SIZE}
    checks.append((f"all images {EXPECTED_SIZE}", size_ok, str(dict(sizes))))

    unique_pixels = len(digests)
    dup_count = total - unique_pixels
    checks.append(("no duplicate images", dup_count == 0, f"{dup_count} duplicates"))

    ok = True
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:<28} -> {detail}")
        ok &= passed

    print(f"\n  image modes: {dict(modes)}")

    manifest = {
        "source": f"https://huggingface.co/datasets/{REPO_ID}",
        "note": (
            "Public parquet mirror of NEU-CLS. Verified against the published "
            "specification of the database (1800 images / 6 classes / 300 per "
            "class / 200x200 grayscale). Originating publication: Song & Yan, "
            "'A noise robust method based on completed local binary patterns for "
            "hot-rolled steel strip surface defects', Applied Surface Science, 2013."
        ),
        "verification_passed": bool(ok),
        "total_images": total,
        "per_class_counts": {str(k): v for k, v in sorted(label_counter.items())},
        "unique_pixel_hashes": unique_pixels,
        "duplicate_images": dup_count,
        "records": records,
    }
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[write]    {RAW/'manifest.json'}")
    print(f"[write]    {IMAGES} ({total} PNGs)")

    if not ok:
        print("\nMirror did NOT match the published NEU-CLS specification.", file=sys.stderr)
        return 1
    print("\nMirror matches the published NEU-CLS specification.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
