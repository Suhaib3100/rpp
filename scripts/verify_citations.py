#!/usr/bin/env python
"""Verify every citation against Crossref / arXiv before it enters the report.

The project brief requires that "every citation must correspond to a real,
verified source". This script resolves each DOI (or arXiv id) against the
publisher's metadata API and records exactly what came back, so the reference
list in the report and presentation is machine-checked rather than asserted.

Output: reports/citations_verified.json
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UA = "SAGE-HE-citation-check/1.0 (mailto:research@example.invalid)"

# key -> DOI or arXiv id. Nothing is cited that is not in this list.
SOURCES: dict[str, str] = {
    "jia2023":        "10.1186/s13677-023-00537-0",
    "song2013":       "10.1016/j.apsusc.2013.09.002",
    "cheon2017ckks":  "10.1007/978-3-319-70694-8_15",
    "tan2019effnet":  "arXiv:1905.11946",
    "selvaraju2017":  "10.1007/s11263-019-01228-7",
    "akiba2019optuna": "10.1145/3292500.3330701",
    "zuiderveld1994clahe": "10.1016/B978-0-12-336156-1.50061-6",
    "elkhatib2025prism": "arXiv:2511.07807",
    "he2016resnet":   "10.1109/CVPR.2016.90",
    "hu2018senet":    "10.1109/CVPR.2018.00745",
    "lee2022minimax": "10.1109/TDSC.2021.3105111",
    "paszke2019torch": "arXiv:1912.01703",
    "pedregosa2011sklearn": "arXiv:1201.0490",
    "buslaev2020albumentations": "10.3390/info11020125",
    "dosovitskiy2021vit": "arXiv:2010.11929",
}


def fetch_json(url: str, timeout: int = 45) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
        return None


def fetch_text(url: str, timeout: int = 45) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return None


def verify_doi(doi: str) -> dict:
    data = fetch_json(f"https://api.crossref.org/works/{doi}")
    if not data or "message" not in data:
        return {"verified": False, "reason": "not found in Crossref"}
    m = data["message"]
    authors = [f"{a.get('given','')} {a.get('family','')}".strip()
               for a in m.get("author", [])]
    dp = (m.get("published") or m.get("issued") or {}).get("date-parts", [[None]])
    return {
        "verified": True,
        "source": "crossref",
        "title": (m.get("title") or [""])[0],
        "authors": authors,
        "container": (m.get("container-title") or [""])[0],
        "year": dp[0][0] if dp and dp[0] else None,
        "volume": m.get("volume"),
        "issue": m.get("issue"),
        "pages": m.get("page"),
        "article_number": m.get("article-number"),
        "type": m.get("type"),
        "doi": m.get("DOI"),
        "url": f"https://doi.org/{m.get('DOI')}",
    }


def verify_arxiv(arxiv_id: str) -> dict:
    aid = arxiv_id.split(":", 1)[1]
    xml = fetch_text(f"http://export.arxiv.org/api/query?id_list={aid}")
    if not xml or "<entry>" not in xml:
        return {"verified": False, "reason": "not found on arXiv"}
    entry = re.search(r"<entry>(.*?)</entry>", xml, re.S).group(1)
    title = re.search(r"<title>(.*?)</title>", entry, re.S)
    pub = re.search(r"<published>(.*?)</published>", entry, re.S)
    return {
        "verified": True,
        "source": "arxiv",
        "title": " ".join(title.group(1).split()) if title else "",
        "authors": re.findall(r"<name>(.*?)</name>", entry),
        "container": "arXiv preprint",
        "year": int(pub.group(1)[:4]) if pub else None,
        "doi": None,
        "arxiv_id": aid,
        "url": f"https://arxiv.org/abs/{aid}",
    }


def main() -> int:
    out: dict[str, dict] = {}
    for key, ident in SOURCES.items():
        rec = verify_arxiv(ident) if ident.startswith("arXiv:") else verify_doi(ident)
        rec["key"] = key
        rec["identifier"] = ident
        out[key] = rec
        status = "OK  " if rec["verified"] else "FAIL"
        title = rec.get("title", rec.get("reason", ""))[:78]
        print(f"[{status}] {key:<24} {title}")
        time.sleep(0.6)   # be polite to the APIs

    n_ok = sum(1 for r in out.values() if r["verified"])
    print(f"\n{n_ok}/{len(out)} citations verified")

    path = ROOT / "reports" / "citations_verified.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"written: {path}")
    return 0 if n_ok == len(out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
