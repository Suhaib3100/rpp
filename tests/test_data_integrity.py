"""Dataset integrity and split-hygiene tests.

These encode the guarantees the experimental claims rest on: the dataset is
really NEU-CLS, the splits do not overlap, and no duplicate image spans the
train/test boundary.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import pytest

from src.datasets.neu_cls import CLASS_NAMES, NEUCLSDataset
from src.datasets.splits import load_splits

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/raw/manifest.json"
SPLITS = ROOT / "data/splits.json"
IMAGES = ROOT / "data/raw/images"

pytestmark = pytest.mark.skipif(
    not MANIFEST.exists(), reason="dataset not downloaded; run scripts/download_dataset.py"
)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def test_matches_published_neu_cls_spec(manifest):
    """1800 images, 6 classes, 300 per class, 200x200."""
    recs = manifest["records"]
    assert len(recs) == 1800
    counts = Counter(r["label_index"] for r in recs)
    assert len(counts) == 6
    assert all(c == 300 for c in counts.values()), dict(counts)
    assert {(r["width"], r["height"]) for r in recs} == {(200, 200)}


def test_class_names_present():
    lm = json.loads((ROOT / "data/raw/label_map.json").read_text())
    assert [lm["index_to_name"][str(i)] for i in range(6)] == CLASS_NAMES


def test_splits_are_disjoint():
    sp = load_splits(SPLITS)
    tr = {f for f, _ in sp["train"]}
    va = {f for f, _ in sp["val"]}
    te = {f for f, _ in sp["test"]}
    assert not (tr & va) and not (tr & te) and not (va & te)


def test_no_duplicate_image_crosses_splits():
    """The split must not put two copies of one image on opposite sides."""
    sp = load_splits(SPLITS)
    owner: dict[str, str] = {}
    for split, recs in sp.items():
        for fn, _ in recs:
            img = cv2.imread(str(IMAGES / fn), cv2.IMREAD_GRAYSCALE)
            sha = hashlib.sha256(img.tobytes()).hexdigest()
            assert sha not in owner or owner[sha] == split, (
                f"duplicate image spans {owner.get(sha)} and {split}: {fn}"
            )
            owner[sha] = split


def test_test_set_is_balanced():
    sp = load_splits(SPLITS)
    counts = Counter(y for _, y in sp["test"])
    assert set(counts) == set(range(6))
    assert all(c == 60 for c in counts.values()), dict(counts)


def test_dataset_returns_normalised_tensors():
    sp = load_splits(SPLITS)
    ds = NEUCLSDataset(sp["val"], IMAGES, size=160, train=False)
    x, y = ds[0]
    assert x.shape == (1, 160, 160)
    assert x.dtype.is_floating_point
    assert 0 <= y < 6
    # standardised, so it must not still be in [0, 255]
    assert x.abs().max() < 10


def test_train_pipeline_is_stochastic_and_eval_is_not():
    sp = load_splits(SPLITS)
    tr = NEUCLSDataset(sp["train"], IMAGES, size=160, train=True)
    ev = NEUCLSDataset(sp["train"], IMAGES, size=160, train=False)
    assert (ev[0][0] == ev[0][0]).all(), "eval transform must be deterministic"
    diffs = [not (tr[0][0] == tr[0][0]).all() for _ in range(8)]
    assert any(diffs), "train transform never augmented across 8 draws"
