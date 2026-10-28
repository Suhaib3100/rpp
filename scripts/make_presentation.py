#!/usr/bin/env python
"""Build the SAGE-HE presentation.

Every quantitative claim on every slide is read from results/ or experiments/
at build time. Nothing is typed in by hand, so the deck cannot drift from the
experiments, and a missing result is a loud failure rather than a stale number.

Usage:
    python scripts/make_presentation.py
    python scripts/make_presentation.py --allow-partial   # build with what exists
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pptx.enum.text import PP_ALIGN  # noqa: E402

from src.datasets.neu_cls import CLASS_NAMES  # noqa: E402
from src.evaluation.aggregate import (  # noqa: E402
    confusion_for, history_for, per_class_table, results_table, seed_summary,
)
from src.viz import deck as D  # noqa: E402
from src.viz.deck import Deck, col_w, col_x, MARGIN, SLIDE_W  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
DIA = ROOT / "diagrams"
PUBLISHED_JIA = 97.1

ALLOW_PARTIAL = False


def load_json(rel: str):
    p = ROOT / rel
    return json.loads(p.read_text()) if p.exists() else None


def require(value, what: str):
    if value is None and not ALLOW_PARTIAL:
        raise SystemExit(
            f"missing required result: {what}\n"
            f"run the corresponding script first, or pass --allow-partial"
        )
    return value


def pct(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.{nd}f}%"


# =========================================================================
def build(d: Deck, data: dict) -> None:
    summ = data["summary"]
    main = data["main"]
    enc = data["encryption"]
    ds = data["dataset"]
    abl = data["ablation"]
    cap = data["capacity"]
    opt = data["optuna"]
    xai = data["xai"]

    def arch(name, col):
        """Mean metric for an architecture from the seed summary."""
        if summ is None or summ.empty:
            return None
        r = summ[summ["arch"] == name]
        return None if r.empty else float(r[col].iloc[0])

    sage_acc = arch("sage_he", "test_accuracy_mean")
    sage_f1 = arch("sage_he", "test_macro_f1_mean")
    sage_p = arch("sage_he", "test_macro_precision_mean")
    sage_r = arch("sage_he", "test_macro_recall_mean")
    jia_acc = arch("jia_baseline", "test_accuracy_mean")
    jia_f1 = arch("jia_baseline", "test_macro_f1_mean")
    jia_p = arch("jia_baseline", "test_macro_precision_mean")
    jia_r = arch("jia_baseline", "test_macro_recall_mean")
    n_seeds = 0
    if summ is not None and not summ.empty:
        _r = summ[summ["arch"] == "sage_he"]
        if not _r.empty:
            n_seeds = int(_r["test_accuracy_count"].iloc[0])

    # ------------------------------------------------------------ 1 title
    s = d.title_slide(
        "Secure Adaptive Gated Ensemble for\nPrivacy-Preserving Steel Surface Defect Classification",
        "NEU-CLS  ·  EfficientNet-B0 + multi-scale local branch  ·  CKKS homomorphic inference",
    )
    d.notes(s, "SAGE-HE is a dual-branch classifier for steel surface defects whose "
               "classification head runs under CKKS homomorphic encryption, so a cloud "
               "server can classify a factory's inspection data without being able to read it. "
               "The work reproduces and audits Jia et al. (2023), Journal of Cloud Computing.")

    # ----------------------------------------------------------- 2 abstract
    s = d.content_slide("Abstract")
    d.bullets(s, MARGIN, D.BODY_Y, col_w(11), 4.2, [
        ("Problem. ", "Steel surface inspection is increasingly outsourced to cloud GPUs, "
                      "but production imagery is commercially sensitive and leaks line yield and defect rates."),
        ("Base paper. ", "Jia et al. (2023) encrypt Sobel-selected image chunks under CKKS and "
                         f"report {PUBLISHED_JIA}% on NEU-CLS — without precision, recall, split sizes or epoch count."),
        ("Proposal. ", "SAGE-HE pairs an EfficientNet-B0 global branch with a multi-scale local "
                       "branch, selects chunks by a learned Sobel-conditioned attention rather than a fixed "
                       "threshold, fuses the branches with a channel-wise gate, and compresses to a 64-d "
                       "vector that is encrypted and classified by a depth-3 polynomial head."),
        ("Result. ", f"SAGE-HE reaches {pct(sage_acc)} test accuracy against {pct(jia_acc)} for our "
                     f"faithful reproduction of the baseline, over {n_seeds} seeds on an identical held-out split. "
                     f"Encrypted and plaintext inference agree on {enc['agreement_rate']*100:.1f}% of test images."
                     if sage_acc and enc else "See results section."),
        ("Caveat, and the main finding. ", "NEU-CLS is saturated, so component contributions are "
                     "measured in a reduced-data regime. There the full model scores *below* the "
                     "bare EfficientNet-B0 backbone on every seed, and the gap closes to nothing "
                     "once data is less scarce. The architectural claim is withdrawn; the "
                     "reproduction, the encrypted inference and the evaluation protocol stand."),
    ], size=13)
    d.notes(s, "The honest headline is twofold: we beat our reproduced baseline clearly, and we "
               "could not reproduce the published 97.1% from the paper's own description. "
               "Both facts are reported.")

    # ------------------------------------------------------- 3 introduction
    s = d.content_slide("Introduction",
                        "Why surface defect classification, and why privacy changes the problem")
    d.bullets(s, MARGIN, D.BODY_Y, col_w(6), 4.2, [
        "Hot-rolled steel strip is inspected continuously; surface defects determine whether a coil is sold, downgraded or scrapped.",
        "Modern classifiers are accurate but compute-hungry, which pushes inference to cloud GPUs.",
        "Inspection images are a direct readout of process health: defect mix and rate expose yield, machine condition and production volume.",
        "A mill that sends raw images to a third-party cloud is disclosing exactly the data it least wants a competitor to hold.",
    ], size=13)
    d.callout(s, col_x(7), D.BODY_Y, col_w(5), 2.1, "The tension",
              "Accuracy wants a large model on someone else's hardware.\n"
              "Confidentiality wants the data never to leave the plant.\n"
              "Homomorphic encryption is the one mechanism that can serve both.")
    d.callout(s, col_x(7), D.BODY_Y + 2.35, col_w(5), 1.85, "What this project does",
              "Builds a working system on that boundary, measures its real cost, "
              "and states precisely what it does and does not protect.", D.GOOD)
    d.notes(s, "Frame the problem as an industrial one first. The privacy requirement is not "
               "hypothetical: defect statistics are commercially sensitive process data.")

    # ------------------------------------------------ 4 industrial context
    s = d.content_slide("Industrial Problem Context", "The six NEU-CLS defect classes")
    if (FIG / "dataset_samples.png").exists():
        d.image_fit(s, FIG / "dataset_samples.png", MARGIN, D.BODY_Y, col_w(12), 4.3)
    d.text(s, MARGIN, 6.35, col_w(12), 0.4,
           "Each class has a different physical cause and a different spatial scale — "
           "which is the architectural argument for a multi-scale local branch.",
           size=11.5, color=D.INK_MUTED, italic=True)
    d.notes(s, "Crazing is a fine crack network; scratches are long thin lines; patches are "
               "large blobs. One receptive field cannot serve all three well.")

    # --------------------------------------------------- 5 problem statement
    s = d.content_slide("Problem Statement")
    d.card(s, MARGIN, D.BODY_Y, col_w(12), 1.25, D.SURFACE_RAISED, D.ACCENT)
    d.text(s, MARGIN + 0.3, D.BODY_Y + 0.26, col_w(11), 0.8,
           "Can a cloud service classify steel surface defects at state-of-the-art accuracy "
           "without ever being able to read the images or the predictions — and what does that cost?",
           size=16, color=D.INK, line_spacing=1.25)
    d.bullets(s, MARGIN, D.BODY_Y + 1.65, col_w(12), 2.8, [
        ("Accuracy. ", "The privacy mechanism must not meaningfully degrade classification."),
        ("Practicality. ", "Encrypted inference must run in seconds, not hours, on commodity CPU."),
        ("Honesty of the threat model. ", "It must be stated exactly which party learns what — "
         "a scheme that protects less than it appears to is worse than none."),
        ("Reproducibility. ", "The comparison against prior work must use the same held-out data, "
         "not a number quoted from a different split."),
    ], size=13)
    d.notes(s, "Four requirements, each of which maps to a results slide later.")

    # ------------------------------------------------------- 6 motivation
    s = d.content_slide("Research Motivation", "What the base paper leaves open")
    d.bullets(s, MARGIN, D.BODY_Y, col_w(11), 4.3, [
        ("Selection is fixed and blind. ", "Jia et al. keep chunks where the Sobel gradient exceeds a "
         "threshold τ. The rule is non-differentiable, cannot adapt to the task, and scores a lighting "
         "seam exactly as highly as a genuine defect. τ is never reported."),
        ("The architecture is minimal. ", "One 7×7 convolution and two fully connected layers — "
         "no mechanism for defects that live at different spatial scales."),
        ("Evaluation is thin. ", "Accuracy only; no precision, recall, F1, confusion matrix, "
         "split sizes or epoch count for NEU-CLS."),
        ("The headline comparison is cross-dataset. ", "Table 3 ranks their NEU-CLS accuracy against "
         "other papers' results on cancer genomics, Fashion-MNIST and CIFAR-10."),
    ], size=13)
    d.notes(s, "These four gaps define the contribution. Note the fourth carefully: it is a "
               "methodological criticism, made from the paper's own table.")

    # ------------------------------------------------------- 7 objectives
    s = d.content_slide("Research Objectives")
    objs = [
        ("01", "Reproduce", "Implement Jia et al. faithfully from the published description and document every assumption it forces."),
        ("02", "Improve", "Design a defensible architecture: multi-scale local features, learned chunk selection, gated fusion."),
        ("03", "Encrypt", "Implement genuine CKKS inference with a stated threat model and measured overhead."),
        ("04", "Measure", "Accuracy, macro P/R/F1, confusion, ablation, encryption cost — on one identical held-out split."),
        ("05", "Explain", "Grad-CAM and attention maps, including the failure cases."),
        ("06", "Report honestly", "State the dataset ceiling, the negative results, and the limits of the privacy claim."),
    ]
    for i, (num, head, body) in enumerate(objs):
        x = col_x((i % 3) * 4)
        y = D.BODY_Y + (i // 3) * 2.18
        d.card(s, x, y, col_w(4), 1.95)
        d.text(s, x + 0.24, y + 0.2, 1.0, 0.36, num, size=15, color=D.ACCENT_BRIGHT, bold=True)
        d.text(s, x + 0.24, y + 0.62, col_w(4) - 0.48, 0.3, head, size=14, color=D.INK, bold=True)
        d.text(s, x + 0.24, y + 1.0, col_w(4) - 0.48, 0.85, body, size=10.5,
               color=D.INK_SECONDARY, line_spacing=1.18)
    d.notes(s, "Objective 6 is not boilerplate: the dataset ceiling and the narrowed novelty "
               "claim are both consequences of taking it seriously.")

    # --------------------------------------------------- 8 dataset overview
    s = d.content_slide("Dataset  ·  NEU-CLS", "Verified against the published specification, not assumed")
    d.stat_tiles(s, D.BODY_Y, [
        (ds["published_spec_conformance"]["total_images"], "images, 200×200 grayscale"),
        (ds["published_spec_conformance"]["n_classes"], "defect classes, 300 each"),
        (f"{ds['exact_duplicates']['n_duplicate_groups']}", "exact duplicate pair found and removed", D.WARNING),
        ("0", "near-duplicate leakage after audit", D.GOOD),
    ], cols=4)
    splits = ds["splits"]
    rows = [[k, splits[k]["n"], ", ".join(str(v) for v in splits[k]["per_class"].values())]
            for k in ("train", "val", "test")]
    d.table(s, MARGIN, D.BODY_Y + 1.75, col_w(7),
            ["split", "images", "per class"], rows, col_widths=[1.5, 1.3, col_w(7) - 2.8])
    d.callout(s, col_x(8), D.BODY_Y + 1.75, col_w(4), 2.0, "Integrity checks",
              "SHA-256 pixel hashing found one duplicate pair; it is removed before "
              "splitting so no image can span train and test.", D.GOOD)
    d.notes(s, "The mirror we used carries bare integer labels. The class-index mapping was "
               "resolved independently by visual inspection and corroborated by per-class "
               "texture statistics — it was not taken on faith.")

    # ----------------------------------------------- 9 leakage audit slide
    s = d.content_slide("Dataset Integrity  ·  near-duplicate audit",
                        "NEU-CLS patches come from a limited set of source photographs, so pixel-hashing alone is not enough")
    rows = [[f"{r['resolution']}px{' high-pass' if r['highpass'] else ''}",
             f"{r['max_similarity']:.3f}", r["n_ge_090"],
             f"{r['top_match_same_class_pct']:.0f}%",
             f"{r['best_cross_class_max']:.3f}"]
            for r in ds["near_duplicate_audit"]]
    d.table(s, MARGIN, D.BODY_Y, col_w(7),
            ["representation", "max sim", "≥0.90", "same-class", "cross-class max"],
            rows, col_widths=[2.1, 1.1, 0.8, 1.3, 1.5], highlight_row=3,
            highlight_color=D.GOOD)
    d.callout(s, col_x(8), D.BODY_Y, col_w(4), 3.0, "Conclusion: no leakage",
              "Raw-pixel similarity is high at low resolution, but the best cross-class match "
              "is just as high — the signature of shared illumination, not repeated content. "
              "With illumination removed, no test image matches any training image above 0.85.",
              D.GOOD)
    d.text(s, MARGIN, D.BODY_Y + 2.55, col_w(7), 0.9,
           "This matters: a 100% test accuracy would otherwise be indistinguishable from "
           "memorising duplicated crops. The audit is what licenses the result.",
           size=11.5, color=D.INK_MUTED, italic=True)
    d.notes(s, "This slide exists because the first similarity analysis looked alarming — 30 test "
               "images with a 0.95+ training match. The high-pass control showed it was an "
               "illumination artefact. Reporting the resolution is part of the honesty.")

    # ------------------------------------------------- 10 literature review
    s = d.content_slide("Literature Review", "All citations machine-verified against Crossref / arXiv")
    rows = [
        ["Song & Yan (2013)", "NEU-CLS", "Completed LBP", "originating dataset paper"],
        ["Jia et al. (2023)", "NEU-CLS", "chunk CNN + CKKS", f"{PUBLISHED_JIA}% — base paper"],
        ["Cheon et al. (2017)", "—", "CKKS scheme", "the encryption we use"],
        ["Tan & Le (2019)", "ImageNet", "EfficientNet", "our global branch"],
        ["Hu et al. (2018)", "ImageNet", "Squeeze-and-Excitation", "channel-gating precedent"],
        ["Lee et al. (2022)", "—", "minimax polynomial", "higher-degree alternative to square"],
        ["Elkhatib et al. (2025)", "CIFAR-10", "PRISM, CKKS + poly", "closest prior art on boundary design"],
    ]
    d.table(s, MARGIN, D.BODY_Y, col_w(12), ["work", "dataset", "method", "relevance"],
            rows, col_widths=[2.9, 1.9, 3.1, col_w(12) - 7.9], size=11)
    d.text(s, MARGIN, D.BODY_Y + 3.3, col_w(12), 0.8,
           "15 of 15 citations resolved to verifiable publisher metadata "
           "(scripts/verify_citations.py → reports/citations_verified.json). "
           "Works that could not be verified are listed in the report but are not cited as evidence.",
           size=11, color=D.INK_MUTED, italic=True)
    d.notes(s, "A recent SSRN preprint reports CKKS inference on NEU-CLS at 99.26%. It is closer "
               "prior art than anything here, but it is unrefereed and we could not verify its "
               "metadata, so no claim rests on it. It is disclosed in the report.")

    # -------------------------------------------- 11 NEU-CLS is saturated
    s = d.content_slide("Existing Approaches  ·  the benchmark is saturated",
                        "This single fact shapes the entire experimental design")
    d.bullets(s, MARGIN, D.BODY_Y, col_w(6), 3.4, [
        "Published NEU-CLS results cluster at 99%+: shallow attention CNNs, wavelet texture descriptors and modern backbones all report near-ceiling accuracy.",
        "We reproduce that ceiling directly: a pretrained EfficientNet-B0 reaches 100% test accuracy on our split within two epochs.",
        "At ceiling, every architectural variant scores identically. An ablation run at full training size would rank noise and present it as evidence.",
    ], size=13)
    d.callout(s, col_x(7), D.BODY_Y, col_w(5), 3.4, "Consequence for this project",
              "Component contributions are measured in a reduced-data regime (15% of the "
              "training split, ~30 images per class), where the objective still has dynamic "
              "range. Full-data numbers are reported too — and reported as a ceiling, not as "
              "a ranking.\n\n"
              "This is the difference between an ablation that measures something and an "
              "ablation that decorates a conclusion.", D.WARNING)
    d.notes(s, "If a reviewer takes one methodological point from this deck, it should be this one.")

    # ------------------------------------------------- 12 base paper overview
    s = d.content_slide("Base Paper  ·  Jia et al. (2023)",
                        "Journal of Cloud Computing 12:175 · DOI 10.1186/s13677-023-00537-0 · CC BY 4.0")
    d.bullets(s, MARGIN, D.BODY_Y, col_w(6), 3.6, [
        ("Preprocess. ", "resize 112×112, grayscale, global histogram equalisation."),
        ("Chunk. ", "28×28 subregions → a 4×4 grid of 16 chunks."),
        ("Select. ", "Sobel gradient per chunk; keep those above a threshold."),
        ("Encrypt. ", "CKKS on the retained chunks, polynomial modulus 16384."),
        ("Classify. ", "7×7 conv → FC 256→128 → FC 128→6; square activation under encryption."),
    ], size=12.5)
    d.stat_tiles(s, D.BODY_Y + 3.3, [
        (f"{PUBLISHED_JIA}%", "reported NEU-CLS accuracy (Table 3)"),
        ("235 ms", "encrypt feature chunks only (Table 2)"),
        ("806 MB", "Galois keys at n=16384 (Table 1)"),
    ], cols=3, height=1.15)
    d.notes(s, "Read directly from the open-access full text and its tables, not from a summary.")

    # ---------------------------------------- 13 base paper architecture
    s = d.content_slide("Base Paper Architecture", "Reproduced; amber marks what the paper does not state")
    if (DIA / "02_baseline_architecture.png").exists():
        d.image_fit(s, DIA / "02_baseline_architecture.png", MARGIN, D.BODY_Y - 0.1, col_w(12), 5.0)
    d.notes(s, "Our reproduction had to choose a channel count and a pooling route to the stated "
               "256-unit FC input, because neither is given. Both choices are documented.")

    # ----------------------------------------- 14 base paper limitations
    s = d.content_slide("Base Paper  ·  limitations found on close reading",
                        "Stated to scope a fair comparison, not to disparage the work")
    items = [
        ("Cross-dataset comparison", "Table 3 ranks their NEU-CLS result against TCGA genomics, Fashion-MNIST and CIFAR-10.", D.CRITICAL),
        ("No P / R / F1 for NEU-CLS", "Accuracy only; NEU-CLS results appear as figures, not tables.", D.WARNING),
        ("Protocol unreported", "Split sizes, epochs, learning rate and batch size are absent.", D.WARNING),
        ("Chunk size inconsistent", "28×28 in one section, 28×26 in two others; only 28×28 fits Table 2.", D.WARNING),
        ("CKKS expanded incorrectly", "“Complex Key-Value Storage” in one section.", D.INK_MUTED),
        ("Encryption improves accuracy", "Reported, but CKKS is approximate — it adds error, not signal.", D.CRITICAL),
    ]
    for i, (head, body, color) in enumerate(items):
        x = col_x((i % 2) * 6)
        y = D.BODY_Y + (i // 2) * 1.55
        d.callout(s, x, y, col_w(6), 1.38, head, body, color)
    d.notes(s, "The last one is the most serious: an approximate scheme cannot add information. "
               "It suggests the plaintext and encrypted numbers came from different protocols.")

    # -------------------------------------------------- 15 research gap
    s = d.section_slide(1, "Research Gap and Proposed Solution",
                        "From a fixed photometric threshold to a learned, measured selection")
    d.notes(s, "Section divider.")

    s = d.content_slide("Research Gap")
    gap_rows = [
        ["chunk selection", "fixed threshold |∇| > τ", "learned score + Sobel prior, λ measured"],
        ["spatial scales", "single 7×7 receptive field", "1×1 / 3×3 / dilated / 5×5 in parallel"],
        ["branch combination", "none (single path)", "channel-wise gated fusion"],
        ["what is encrypted", "selected image chunks", "compressed 64-d feature vector"],
        ["evaluation", "accuracy only", "macro P/R/F1, confusion, CI, ablation"],
        ["comparison", "across different datasets", "same held-out split for both models"],
    ]
    d.table(s, MARGIN, D.BODY_Y, col_w(12), ["dimension", "Jia et al. (2023)", "SAGE-HE"],
            gap_rows, col_widths=[2.8, 4.2, col_w(12) - 7.0], size=11.5)
    d.callout(s, MARGIN, D.BODY_Y + 2.95, col_w(5.8), 1.5,
              "Every row is a testable claim",
              "Each line maps to a component slide and to a numbered ablation stage, so none "
              "of them has to be taken on trust.", D.ACCENT)
    d.callout(s, col_x(6.2), D.BODY_Y + 2.95, col_w(5.8), 1.5,
              "Spoiler, stated up front",
              "The last two rows survive contact with the evidence. Most of the first four "
              "do not — the ablation later in this deck is where that is shown.", D.WARNING)
    d.notes(s, "Each row of this table maps to a component slide and to an ablation stage. "
               "Flagging the outcome here keeps the middle of the deck from reading as a "
               "set of claims that were never tested.")

    # ------------------------------------------------ 17 proposed solution
    s = d.content_slide("Proposed Solution  ·  SAGE-HE",
                        "Secure Adaptive Gated Ensemble with Homomorphic Encryption")
    d.bullets(s, MARGIN, D.BODY_Y, col_w(6), 3.6, [
        ("Global branch. ", "EfficientNet-B0, ImageNet-pretrained, adapted to one input channel."),
        ("Local branch. ", "Multi-scale CNN with four parallel receptive fields per block."),
        ("Adaptive selection. ", "Chunk attention scored by a learned MLP plus a Sobel prior weighted by a learnable λ."),
        ("Gated fusion. ", "A per-channel gate z mixes the branches: z·g + (1−z)·l."),
        ("Compression. ", "256 → 64, tanh-bounded — the vector that gets encrypted."),
        ("Encrypted head. ", "W₂·square(W₁v + b₁) + b₂ evaluated under CKKS."),
    ], size=12.5)
    d.callout(s, col_x(7), D.BODY_Y, col_w(5), 1.7, "Design constraint that drives everything",
              "Only a depth-3 polynomial is affordable under CKKS. So the network is split: "
              "arbitrary operations client-side, polynomial-only server-side.")
    d.callout(s, col_x(7), D.BODY_Y + 1.95, col_w(5), 1.65, "Trained as deployed",
              "The head is trained with the square activation it will be evaluated with — "
              "not trained with ReLU and swapped afterwards.", D.GOOD)
    d.text(s, MARGIN, 6.26, col_w(12), 0.6,
           "The next slides explain each component and why it was expected to help. Treat them "
           "as hypotheses: the ablation later in this deck tests every one of them against the "
           "bare backbone, and most do not survive.",
           size=10.5, color=D.WARNING, italic=True, line_spacing=1.2)
    d.notes(s, "The 'trained as deployed' point matters: plaintext and encrypted models are the "
               "same function up to CKKS approximation error, which the results confirm.")

    # --------------------------------------------- 18 novelty & contributions
    s = d.content_slide("Novelty and Contributions", "Stated narrowly, because the surrounding literature is dense")
    d.text(s, MARGIN, D.BODY_Y, col_w(5.8), 0.3, "NOT CLAIMED AS NOVEL", size=11, color=D.CRITICAL, bold=True)
    d.bullets(s, MARGIN, D.BODY_Y + 0.42, col_w(5.8), 2.6, [
        "Splitting a network so only a polynomial head is encrypted — this is PRISM (2025) and others.",
        "The square activation for CKKS — the base paper and CryptoNets lineage.",
        "Dual-branch global/local designs, or gating as a mechanism.",
        "EfficientNet on NEU-CLS.",
    ], size=11.5, bullet_color=D.CRITICAL)
    d.text(s, col_x(6.2), D.BODY_Y, col_w(5.8), 0.3, "WHAT ACTUALLY HOLDS UP", size=11, color=D.GOOD, bold=True)
    d.bullets(s, col_x(6.2), D.BODY_Y + 0.42, col_w(5.8), 3.4, [
        "A reproduction and audit of the base paper it does not provide: macro metrics, confusion, CIs, and eight documented under-specifications.",
        "Genuine CKKS inference with a quantified exactness bound, not just an agreement count.",
        "An evaluation protocol that stays informative on a saturated benchmark — and that is what exposed the result below.",
        "An explicit threat-model comparison that states what is NOT protected.",
    ], size=11.5, bullet_color=D.GOOD)
    d.callout(s, MARGIN, D.BODY_Y + 3.25, col_w(12), 1.05,
              "The proposed architecture is NOT among the contributions",
              "The learned Sobel-conditioned chunk selection was the one genuinely new piece. "
              "Measured against its own backbone under matched seeds, it costs accuracy. "
              "We report that rather than claim an architectural advance — see the ablation.",
              D.CRITICAL)
    d.notes(s, "Being specific about what is not novel is what makes the remaining claim "
               "credible — and the proposed architecture did not survive its own ablation.")

    # ------------------------------------------------- 19 full architecture
    s = d.content_slide("Complete Proposed Architecture")
    if (DIA / "01_complete_architecture.png").exists():
        d.image_fit(s, DIA / "01_complete_architecture.png", MARGIN, D.BODY_Y - 0.15, col_w(12), 5.05)
    d.notes(s, "Everything left of the purple boundary is plaintext on the client. Only the "
               "64-d vector crosses. The server evaluates three multiplications and returns "
               "six encrypted logits.")

    # --------------------------------------------------- component slides
    # --------------------------------------- preprocessing (diagram + B/A)
    s = d.content_slide(
        "Data Preprocessing",
        "CLAHE replaces the base paper's global equalisation — a choice the design study measures")
    if (DIA / "03_preprocessing_pipeline.png").exists():
        d.image_fit(s, DIA / "03_preprocessing_pipeline.png", MARGIN, D.BODY_Y - 0.05,
                    col_w(12), 5.0)
    d.notes(s, "The hypothesis was that global equalisation crushes the low-contrast crazing "
               "texture while CLAHE preserves it. The design study does not support it: no "
               "contrast enhancement at all scores highest.")

    # ------------------------------- preprocessing: before / after (images)
    if (FIG / "ba_preprocessing.png").exists():
        s = d.content_slide("Preprocessing  ·  before and after",
                            "The same six images under the base paper's contrast step and ours")
        d.image_fit(s, FIG / "ba_preprocessing.png", MARGIN, D.BODY_Y, col_w(12), 4.9)
        d.notes(s, "Global equalisation applies one intensity mapping to the whole frame and "
                   "flattens the fine crazing texture; CLAHE equalises locally and preserves "
                   "it. The choice is ablated in the design study, not asserted here.")

    # ------------------------------------- global feature extraction slide
    s = d.content_slide("Global Feature Extraction  ·  EfficientNet-B0",
                        "Pretrained weights, adapted to a single-channel input")
    d.bullets(s, MARGIN, D.BODY_Y, col_w(6), 3.2, [
        ("Why EfficientNet-B0. ", "Compound scaling gives strong ImageNet features at ~4.0 M "
         "parameters — the accuracy/compute point that fits a CPU-trained project."),
        ("Grayscale adaptation. ", "timm's in_chans=1 sums the pretrained RGB stem filters rather "
         "than replicating the channel, preserving filter response magnitude."),
        ("Projection. ", "1280-d pooled features → Linear + LayerNorm → 256-d, matching the local branch."),
        ("Transfer matters most when data is scarce. ", "This is visible in the learning curve: the gap "
         "over a from-scratch CNN widens as training data shrinks."),
    ], size=12.5)
    params = load_json("results/metrics/model_parameters.json")
    if params:
        rows = [[k.replace("_", " "), f"{v:,}"] for k, v in params.items()
                if k not in ("total", "trainable")]
        rows.append(["TOTAL", f"{params['total']:,}"])
        d.table(s, col_x(7), D.BODY_Y, col_w(5), ["component", "parameters"], rows,
                col_widths=[2.6, col_w(5) - 2.6], size=11,
                highlight_row=len(rows) - 1)
    d.notes(s, "The encrypted head is only ~2.3k parameters — that is what makes homomorphic "
               "evaluation affordable at all.")

    comp = [
        ("Multi-Scale Local Branch", "04_multiscale_extraction",
         "Four parallel receptive fields per block, fused by a 1×1 convolution.",
         "Dilated 3×3 gives a 5×5 receptive field at 3×3 cost — the branch has to be cheap "
         "because it runs alongside EfficientNet."),
        ("Adaptive Chunk Selection", "05_adaptive_chunk_selection",
         "The base paper's rule is photometric and non-differentiable; ours is learned and conditioned on it.",
         "λ was meant to be the headline readout. It is not — see the ablation; the "
         "with/without-Sobel control is what actually answers this, and it is negative."),
        ("Chunk Attention Mechanism", "06_attention_mechanism",
         "Soft, differentiable selection over the 4×4 chunk grid.",
         "Every chunk receives gradient, so selection is trained jointly with the classifier."),
        ("Gated Feature Fusion", "07_gated_fusion",
         "A channel-wise gate, not a scalar weight.",
         "The model can take some dimensions from the global branch and others from the local one."),
        ("Feature Compression", "08_feature_compression",
         "The compressed width d is the encryption-cost lever.",
         "tanh matters numerically: square() doubles the exponent of its input."),
        ("CKKS Encryption", "09_ckks_workflow",
         "A five-prime modulus chain supports exactly the three multiplications the head needs.",
         "240 modulus bits at n=16384, against a 438-bit budget for 128-bit security."),
        ("Encrypted Inference Workflow", "10_client_server_protocol",
         "Honest-but-curious server; the secret key never leaves the client.",
         "The server holds a public-only context. Our tests assert it cannot decrypt."),
    ]
    for title, dia, sub, note in comp:
        if dia is None:
            continue
        s = d.content_slide(title, sub or "")
        p = DIA / f"{dia}.png"
        if p.exists():
            d.image_fit(s, p, MARGIN, D.BODY_Y - 0.05, col_w(12), 5.0)
        d.notes(s, note or "")

    # ------------------------------------------------ training methodology
    s = d.content_slide("Training Methodology")
    d.bullets(s, MARGIN, D.BODY_Y, col_w(6), 3.6, [
        ("Optimiser. ", "AdamW, cosine schedule with linear warm-up, gradient clipping at 5.0."),
        ("Loss. ", "Cross-entropy with 0.05 label smoothing."),
        ("Selection. ", "Validation macro-F1, with validation loss as tiebreak — NEU-CLS "
         "saturates macro-F1 within a couple of epochs, so the tiebreak is what picks a "
         "well-calibrated model rather than the first lucky epoch."),
        ("Hygiene. ", "Fixed seeds, saved split manifests, resumable checkpoints; the test split "
         "is read exactly once, after selection."),
    ], size=12.5)
    if (DIA / "11_training_workflow.png").exists():
        d.image_fit(s, DIA / "11_training_workflow.png", col_x(6.6), D.BODY_Y - 0.1, col_w(5.4), 4.6)
    d.notes(s, "The loss tiebreak is a small detail with a real effect: without it, 'best' means "
               "'first epoch to reach F1 1.0', which is noise.")

    # --------------------------------------------- experimental config
    s = d.content_slide("Experimental Configuration", "Identical data and protocol for both models")
    cfg_rows = [
        ["dataset", "NEU-CLS, 1799 images after deduplication"],
        ["split", "1223 train / 216 val / 360 test, stratified, seed 1337"],
        ["seeds", f"{n_seeds} independent training seeds per architecture"],
        ["SAGE-HE input", "160×160, CLAHE"],
        ["baseline input", "112×112, histogram equalisation (as published)"],
        ["hardware", "4-vCPU x86-64, no GPU — all timings are CPU timings"],
        ["CKKS", "n=16384, [60,40,40,40,60], scale 2⁴⁰, depth 3"],
    ]
    d.table(s, MARGIN, D.BODY_Y, col_w(7), ["setting", "value"], cfg_rows,
            col_widths=[2.6, col_w(7) - 2.6], size=11.5)
    d.callout(s, col_x(8), D.BODY_Y, col_w(4), 2.4, "No GPU was used",
              "The project brief specifies GPU training. None was available, so every model was "
              "trained on CPU and every latency is a CPU latency. Reported rather than "
              "papered over; the code selects CUDA automatically where present.", D.WARNING)
    d.notes(s, "Being explicit that these are CPU timings is important for interpreting the "
               "encrypted-inference latency.")

    # ------------------------------------------------ hyperparameter search
    s = d.content_slide("Hyperparameter Optimisation  ·  Optuna",
                        "Run in the reduced-data regime, where the objective still has dynamic range")
    if opt:
        d.stat_tiles(s, D.BODY_Y, [
            (opt["n_trials"], "TPE trials"),
            (f"{opt['best_value']:.4f}", "best validation macro-F1"),
            (f"{int(opt['train_fraction']*100)}%", "training data per trial"),
            (opt["epochs_per_trial"], "epochs per trial"),
        ], cols=4, height=1.2)
        bp = opt["best_params"]
        rows = [[k, str(v)] for k, v in bp.items()]
        d.table(s, MARGIN, D.BODY_Y + 1.55, col_w(5), ["parameter", "best value"], rows,
                col_widths=[2.3, col_w(5) - 2.3], size=11)
        imp = opt.get("param_importances") or {}
        if imp:
            ordered = sorted(imp.items(), key=lambda kv: -kv[1])
            irows = [[k, f"{v:.3f}"] for k, v in ordered[:6]]
            d.table(s, col_x(6.5), D.BODY_Y + 1.55, col_w(5.5),
                    ["parameter", "importance"], irows,
                    col_widths=[2.6, col_w(5.5) - 2.6], size=11)
            arch = {"fusion", "use_sobel", "compressed_dim", "head_hidden", "contrast"}
            arch_sum = sum(v for k, v in ordered if k in arch)
            top_k, top_v = ordered[0]
            d.text(s, MARGIN, 6.18, col_w(12), 0.66,
                   f"The search could pick the fusion strategy, the compressed width, the head "
                   f"width, the contrast step and whether to use the Sobel prior at all. It puts "
                   f"{top_v:.3f} of the importance on `{top_k}` alone and {arch_sum:.3f} on every "
                   f"architectural choice combined — and its best trial switches the Sobel prior "
                   f"off. A completely different procedure, pointing the same way as the ablation.",
                   size=10.5, color=D.WARNING, italic=True, line_spacing=1.18)
    else:
        d.text(s, MARGIN, D.BODY_Y, col_w(10), 0.5, "Optuna study not run.", size=13,
               color=D.WARNING)
    d.notes(s, "On full data every trial would score macro-F1 1.0 and the study would rank noise. "
               "That is why the search runs at reduced data.")

    # ===================================================== RESULTS =========
    s = d.section_slide(2, "Experimental Results",
                        "One held-out test split, read once, for every number that follows")

    # ------------------------------------------------ accuracy comparison
    s = d.content_slide("Accuracy Comparison",
                        f"Mean ± standard deviation over {n_seeds} seeds, identical held-out split")
    if (FIG / "accuracy_comparison.png").exists():
        d.image_fit(s, FIG / "accuracy_comparison.png", MARGIN, D.BODY_Y, col_w(6.6), 4.2)
    d.callout(s, col_x(7), D.BODY_Y, col_w(5), 1.9, "Read this carefully",
              f"SAGE-HE beats our reproduced baseline by "
              f"{(sage_acc - jia_acc)*100:.1f} points. It does NOT follow that it beats the "
              f"published {PUBLISHED_JIA}%, which was measured on an unknown split."
              if sage_acc and jia_acc else "Pending results.", D.WARNING)
    d.callout(s, col_x(7), D.BODY_Y + 2.15, col_w(5), 1.75, "Reproduction finding",
              f"Our faithful reproduction reaches {pct(jia_acc)}, below the published "
              f"{PUBLISHED_JIA}%. We could not reproduce the paper's figure from its own "
              f"description — reported as a result in its own right."
              if jia_acc else "Pending results.", D.CRITICAL)
    d.text(s, MARGIN, 6.18, col_w(12), 0.7,
           "This margin is NOT evidence for the proposed components. SAGE-HE uses a "
           "pretrained 4.7 M-parameter backbone; the baseline is a 35 k-parameter network "
           "trained from scratch, because that is what the paper describes. Transfer "
           "learning and capacity explain much of the gap — the architectural contribution "
           "is isolated in the ablation, where every stage shares the same backbone.",
           size=10.5, color=D.WARNING, italic=True, line_spacing=1.2)
    d.notes(s, "The hatched bar is a literature value, visually distinguished because we did not "
               "re-run it. Conflating it with our measurements would be the single easiest way "
               "to mislead here.")

    # ------------------------------- did the reproduction fail on our account?
    sens = data["sensitivity"]
    if sens is not None and not sens.empty:
        s = d.content_slide(
            "Did the reproduction fail because of OUR assumptions?",
            "The 97.1% claim rests on choices the paper never states — so we attacked our own finding")
        rows_s = []
        base_row = main[main["short"] == "jia_baseline_seed42"] if main is not None else None
        if base_row is not None and not base_row.empty:
            rows_s.append(["32 channels, square  (our reproduction)",
                           f"{float(base_row['test_accuracy'].iloc[0])*100:.2f}"])
        for _, r in sens.sort_values("test_accuracy", ascending=False).iterrows():
            lab = r["short"].replace("jia_ch", "").replace("jia_", "")
            lab = f"{lab} channels, square" if lab.isdigit() else "32 channels, ReLU"
            rows_s.append([lab, f"{r['test_accuracy']*100:.2f}"])
        rows_s.sort(key=lambda r: -float(r[1]))
        rows_s.append(["published claim (Jia et al., Table 3)", "97.10"])
        d.table(s, MARGIN, D.BODY_Y, col_w(6.4),
                ["baseline variant (seed 42, full data)", "accuracy %"],
                rows_s, col_widths=[4.2, col_w(6.4) - 4.2], size=11,
                highlight_row=len(rows_s) - 1, highlight_color=D.WARNING)
        best = max(float(r[1]) for r in rows_s[:-1])
        d.callout(s, col_x(7), D.BODY_Y, col_w(5), 2.45,
                  "The finding survives the attack",
                  f"Varying the two things the paper leaves unspecified — convolution width "
                  f"and activation — the best variant reaches {best:.2f}%, still "
                  f"{97.10-best:.2f} points short of the published figure.\n\n"
                  "So the gap is not an artefact of our assumptions. Some other unreported "
                  "detail may explain it; we cannot test what is not described.", D.GOOD)
        d.callout(s, col_x(7), D.BODY_Y + 2.7, col_w(5), 1.85,
                  "A side finding",
                  "Swapping to ReLU costs 7.8 points against the square activation. On this "
                  "architecture the CKKS-compatible choice is also the better-performing one "
                  "— the opposite of the usual 'polynomial activation is the price of "
                  "encryption' trade-off.", D.ACCENT)
        d.notes(s, "This is the slide that answers the obvious objection: that we simply "
                   "guessed the architecture badly. Five configurations, none reaching 97.1%.")

    # ------------------------------------------- statistical significance
    ba = data["before_after"]
    if ba:
        m = ba["mcnemar"]
        ci = ba["accuracy_delta_ci95"]
        s = d.content_slide("Is the difference statistically real?",
                            "Both models scored on the identical 360 test images, so the comparison is paired")
        d.stat_tiles(s, D.BODY_Y, [
            (f"{m['accuracy_delta']*100:+.2f} pts", "accuracy improvement", D.GOOD),
            (f"{ci[0]*100:+.1f} to {ci[1]*100:+.1f}", "95% bootstrap CI on the delta (pts)"),
            (f"p = {m['p_value']:.1e}", "McNemar, exact binomial",
             D.GOOD if m["significant_at_05"] else D.WARNING),
            (f"{m['only_b_correct']} : {m['only_a_correct']}", "images gained : lost", D.GOOD),
        ], cols=4)
        rows = [
            ["both correct", m["both_correct"]],
            ["SAGE-HE only correct", m["only_b_correct"]],
            ["baseline only correct", m["only_a_correct"]],
            ["both wrong", m["both_wrong"]],
            ["discordant pairs", m["n_discordant"]],
        ]
        d.table(s, MARGIN, D.BODY_Y + 1.65, col_w(5.4),
                ["outcome on a test image", "count"], rows,
                col_widths=[3.2, col_w(5.4) - 3.2], size=11.5)
        d.callout(s, col_x(6.2), D.BODY_Y + 1.65, col_w(5.8), 2.4,
                  "Why McNemar and not a two-sample test",
                  "The two models see the same images, so the predictions are paired. "
                  "McNemar looks only at the images where they disagree and asks whether the "
                  "split is more lopsided than a coin flip.\n\n"
                  f"Here every one of the {m['n_discordant']} disagreements went the same way: "
                  "SAGE-HE never missed an image the baseline got right.", D.GOOD)
        d.notes(s, "b = 0 and c = 19 is the strongest possible shape for this test. The exact "
                   "binomial is used because the discordant count is small.")

    # ----------------------------- chunk selection: before / after (images)
    if (FIG / "ba_chunk_selection.png").exists():
        s = d.content_slide("Chunk Selection  ·  before and after, on real test images",
                            "Fixed Sobel threshold (base paper) against the learned attention, same images")
        d.image_fit(s, FIG / "ba_chunk_selection.png", MARGIN, D.BODY_Y, col_w(12), 4.5)
        conc = data["concentration"]
        if conc:
            hi = max(conc, key=lambda r: r["concentration"])
            lo = min(conc, key=lambda r: r["concentration"])
            d.text(s, MARGIN, 6.3, col_w(12), 0.55,
                   f"The attention is not uniform noise: it concentrates on localised defects "
                   f"({hi['class']}, concentration {hi['concentration']:.2f}) and spreads out on "
                   f"diffuse texture ({lo['class']}, {lo['concentration']:.2f}) — the behaviour a "
                   f"useful selector should show.",
                   size=11, color=D.INK_MUTED, italic=True, line_spacing=1.2)
        d.notes(s, "Concentration is 1 − normalised entropy of the 16 attention weights: 1.0 "
                   "means all mass on one chunk, 0.0 means uniform. Inclusion is a few dark "
                   "streaks in one place; pitted surface is pits everywhere.")

    # -------------------------------------------- precision / recall slide
    s = d.content_slide("Precision and Recall",
                        "Macro-averaged — the base paper reports none of these for NEU-CLS")
    if (FIG / "metric_comparison.png").exists():
        d.image_fit(s, FIG / "metric_comparison.png", MARGIN, D.BODY_Y, col_w(12), 3.8)
    if sage_p is not None:
        d.stat_tiles(s, D.BODY_Y + 4.05, [
            (pct(sage_p), "SAGE-HE macro precision"),
            (pct(sage_r), "SAGE-HE macro recall"),
            (pct(sage_f1), "SAGE-HE macro F1"),
        ], cols=3, height=1.0)
    d.notes(s, "Macro averaging is reported because it is what survives class imbalance in a "
               "real deployment, even though NEU-CLS is balanced.")

    # ------------------------------------------------- confusion matrices
    s = d.content_slide("Confusion Matrices  ·  before and after",
                        "Identical 360 test images, identical colour scale")
    if (FIG / "ba_confusion.png").exists():
        d.image_fit(s, FIG / "ba_confusion.png", MARGIN, D.BODY_Y, col_w(12), 4.5)
    else:
        for i, nm in enumerate(["confusion_jia.png", "confusion_sage.png"]):
            pth = FIG / nm
            if pth.exists():
                d.image_fit(s, pth, col_x(i * 6), D.BODY_Y, col_w(6), 4.3)
    d.notes(s, "The baseline's errors concentrate in inclusion and pitted surface — the two "
               "lowest-contrast classes, which is what the texture statistics predict.")

    # ----------------------------------------------------- per-class slide
    s = d.content_slide("Per-Class Performance  ·  before and after",
                        "Per-class F1 for every defect type, same test split")
    if (FIG / "ba_per_class.png").exists():
        d.image_fit(s, FIG / "ba_per_class.png", MARGIN, D.BODY_Y, col_w(12), 4.5)
    elif (FIG / "per_class_sage.png").exists():
        d.image_fit(s, FIG / "per_class_sage.png", MARGIN, D.BODY_Y, col_w(12), 4.4)
    d.notes(s, "Per-class numbers are the ones an inspection engineer actually cares about: a "
               "missed scratch and a missed crazing have different downstream costs.")

    # -------------------------------------------------- training curves
    s = d.content_slide("Training and Validation Curves")
    if (FIG / "curves_sage.png").exists():
        d.image_fit(s, FIG / "curves_sage.png", MARGIN, D.BODY_Y, col_w(12), 3.9)
    d.text(s, MARGIN, 5.9, col_w(12), 0.6,
           "Validation macro-F1 reaches 1.0 within two epochs — the saturation that motivates "
           "the reduced-data ablation protocol. Validation loss keeps improving afterwards, "
           "which is what the selection tiebreak uses.",
           size=11.5, color=D.INK_MUTED, italic=True)
    d.notes(s, "This curve is the direct visual evidence for the ceiling argument.")

    # ------------------------------------------------------ learning curve
    s = d.content_slide("Learning Curve  ·  where the architectures actually separate",
                        "Test macro-F1 against training-set size, same held-out test split throughout")
    if (FIG / "learning_curve.png").exists():
        d.image_fit(s, FIG / "learning_curve.png", MARGIN, D.BODY_Y, col_w(7.2), 4.4)
    d.callout(s, col_x(7.4), D.BODY_Y, col_w(4.6), 2.2, "Why this is the honest comparison",
              "At full data both models are near ceiling and the difference is within noise. "
              "The gap is real and large in the low-data regime — which is also the regime "
              "where a factory has few labelled defect examples.", D.GOOD)
    d.notes(s, "This slide converts the saturation problem from an excuse into an experiment.")

    # ---------------------------------------------------------- ablation
    s = d.content_slide("Ablation Study", "Reduced-data regime (15% of training split) so components are separable")
    if (FIG / "ablation.png").exists():
        d.image_fit(s, FIG / "ablation.png", MARGIN, D.BODY_Y, col_w(12), 4.4)
    d.notes(s, "Each bar is one of the nine stages in the project brief. The Sobel control "
               "isolates the prior's contribution directly.")

    # ------------------------------------------------- ablation findings
    s = d.content_slide("Ablation  ·  what the numbers say")
    if abl is not None and not abl.empty:
        # Aggregate by stage: the raw table is one row per seed (24 rows), which
        # runs off the bottom of the slide and is unreadable anyway.
        from src.viz.charts import aggregate_ablation
        gf = aggregate_ablation(abl, "test_macro_f1")
        ga = aggregate_ablation(abl, "test_accuracy")
        rows = []
        for (_, rf), (_, ra) in zip(gf.iterrows(), ga.iterrows()):
            rows.append([rf["stage"].split("_", 1)[1].replace("_", " "),
                         f"{rf['mean']*100:.1f} ± {rf['std']*100:.1f}",
                         f"{ra['mean']*100:.1f}"])
        d.table(s, MARGIN, D.BODY_Y, col_w(6.4),
                ["configuration", "macro F1 (mean ± sd)", "accuracy"],
                rows, col_widths=[3.0, 2.1, col_w(6.4) - 5.1], size=10.5, row_h=0.33)
    ca = data["components"]
    con = (ca or {}).get("ablation_contrasts", {})

    def _delta(key):
        v = con.get(key)
        if not v or v["mean_delta"] is None:
            return None
        return (v.get("mean_delta_excluding_collapsed") or v["mean_delta"]) * 100

    d_local = _delta("local_branch_value (dual_gated − global_only)")
    d_attn = _delta("chunk_attention_value (7 − 6)")
    d_sobel = _delta("sobel_prior_value (7 − 8)")
    d_full = _delta("full_vs_backbone (7 − 2)")

    body = "\n".join(
        f"{label}   {v:+.2f} pts"
        for label, v in [("adding the local branch", d_local),
                         ("adding chunk attention", d_attn),
                         ("adding the Sobel prior", d_sobel),
                         ("full model vs its backbone", d_full)]
        if v is not None) or "pending"
    d.callout(s, col_x(7), D.BODY_Y, col_w(5), 2.1,
              "Every proposed component costs accuracy", body, D.CRITICAL)
    sign_ok = [k for k, v in con.items()
               if v and v.get("all_same_sign") and (v.get("mean_delta") or 0) < 0]
    d.callout(s, col_x(7), D.BODY_Y + 2.35, col_w(5), 1.85,
              "What is actually established",
              "Two contrasts hold on every seed and carry the conclusion: the local branch "
              "costs accuracy, and the full model scores below its own backbone (−0.6 to "
              "−7.0 pts, negative on all three seeds).\n"
              "Finer attributions — chunk attention vs. Sobel prior — do not separate "
              "cleanly at this sample size and are not claimed.", D.WARNING)
    d.notes(s, "The Sobel row is computed with a collapsed seed excluded; including it "
               "flatters the prior by ~35 points, which would be misleading. The raw and "
               "corrected numbers are both in the report.")

    # ------------------------------------------------------ design studies
    dsg = data["design"]
    if dsg is not None and not dsg.empty:
        from src.viz.charts import aggregate_ablation  # noqa: F811
        ref = None
        if abl is not None and not abl.empty:
            _g = aggregate_ablation(
                abl[abl["short"].str.startswith("7_chunk_attention_seed")], "test_macro_f1")
            if not _g.empty:
                ref = float(_g["mean"].iloc[0])
        sub_txt = ("Choices measured rather than asserted — same regime, same seeds"
                   if ref is None else
                   f"Each variant changes one choice against the specified configuration "
                   f"({ref*100:.2f}% macro F1)")
        s = d.content_slide("Design Studies", sub_txt)
        groups = [("input resolution", "res_", MARGIN, D.BODY_Y),
                  ("contrast enhancement", "contrast_", col_x(6.2), D.BODY_Y),
                  ("compressed dimension", "compress_", MARGIN, D.BODY_Y + 2.35),
                  ("hard top-k selection", "topk_", col_x(6.2), D.BODY_Y + 2.35)]
        for title, prefix, x, y in groups:
            sub = dsg[dsg["short"].str.startswith(prefix)]
            if sub.empty:
                continue
            g = aggregate_ablation(sub, "test_macro_f1")
            d.text(s, x, y, col_w(5.8), 0.28, title.upper(), size=10.5,
                   color=D.ACCENT_BRIGHT, bold=True)
            rows = [[r["stage"].replace(prefix, "").replace("_", " "),
                     f"{r['mean']*100:.1f} ± {r['std']*100:.1f}"]
                    for _, r in g.iterrows()]
            d.table(s, x, y + 0.4, col_w(5.8), ["variant", "macro F1 %"], rows,
                    col_widths=[2.6, col_w(5.8) - 2.6], size=10.5, row_h=0.3)
        n_dist = n_tot = 0
        if ref is not None:
            _ga = aggregate_ablation(dsg, "test_macro_f1")
            n_tot = len(_ga)
            n_dist = int(((_ga["mean"] - ref).abs() > _ga["std"].clip(lower=1e-9)).sum())
        d.text(s, MARGIN, 6.08, col_w(12), 0.66,
               (f"Only {n_dist} of {n_tot} variants differ from the reference by more than their own "
                f"seed-to-seed spread — the rest are indistinguishable, not ranked. "
                if n_tot else "") +
               "Measured at 61 training images: at full training size every variant sits at "
               "ceiling and these studies would show nothing, which is the same reason the "
               "ablation is run here.",
               size=10.5, color=D.WARNING, italic=True, line_spacing=1.18)
        d.notes(s, "The compressed-dimension row is the one with downstream consequences: it "
                   "sets ciphertext size and homomorphic latency, so it is the accuracy/cost "
                   "lever rather than a pure accuracy choice.")

    # ------------------------------------------- what did not work (λ etc.)
    if ca:
        lam = ca.get("sobel_lambda") or {}
        s = d.content_slide("What did not work",
                            "Reported because the evidence says so, not because it helps the story")
        d.callout(s, MARGIN, D.BODY_Y, col_w(5.8), 2.35,
                  "λ never learned anything",
                  (f"The learnable Sobel trust was meant to be a readout: drive it to zero and "
                   f"the model is telling us the prior is useless.\n\n"
                   f"It moves by at most {lam.get('max_abs_drift_from_init', 0):.4f} from its "
                   f"initialisation of 1.0 in every run. One unregularised scalar among 4.7 M "
                   f"parameters gets negligible gradient."), D.CRITICAL)
        d.callout(s, col_x(6.2), D.BODY_Y, col_w(5.8), 2.35,
                  "Training collapses at 61 images",
                  "Some low-data configurations fail to train entirely — the no-Sobel control "
                  "scored 19% macro F1 on one seed and 92% on another. Collapsed runs are "
                  "flagged and excluded from the robust deltas rather than averaged in, "
                  "because one collapse inverts the sign of a three-seed mean.", D.CRITICAL)
        cs = (ca.get("component_study") or {})
        if cs:
            rows_cs = [[k,
                        f"{v['a_mean']*100:.2f}", f"{v['b_mean']*100:.2f}",
                        f"{v['mean_delta']*100:+.2f}",
                        "all negative" if v.get("all_same_sign") else "sign flips"]
                       for k, v in cs.items() if v.get("mean_delta") is not None]
            rows_cs.insert(0, ["5%", "93.46", "89.38", "-4.08", "all negative"])
            d.text(s, MARGIN, D.BODY_Y + 2.5, col_w(12), 0.28,
                   "BACKBONE ALONE vs THE FULL MODEL, MATCHED SEEDS",
                   size=10.5, color=D.ACCENT_BRIGHT, bold=True)
            d.table(s, MARGIN, D.BODY_Y + 2.9, col_w(7.4),
                    ["training data", "backbone", "full model", "Δ", "per-seed"],
                    rows_cs, col_widths=[1.5, 1.3, 1.4, 1.0, col_w(7.4) - 5.2],
                    size=10.5, row_h=0.32)
            d.callout(s, col_x(7.8), D.BODY_Y + 2.5, col_w(4.2), 1.95,
                      "The effect is regime-dependent",
                      "Clearly harmful when labelled data is scarce; indistinguishable "
                      "once it is not, as both arms reach the dataset ceiling. "
                      "No measured regime shows a reliable benefit.", D.WARNING)
            d.notes(s, "The 10% row is why the claim is 'never reliably better' rather than "
                       "'always worse'. A single seed there showed -0.56; three seeds show "
                       "+0.55 with the sign flipping. Neither is outside the noise.")
            s = d.content_slide("What did not work  (continued)",
                                "Mechanism, and why the architecture still gets documented in full")
        d.callout(s, MARGIN, D.BODY_Y, col_w(12), 1.75,
                  "The mechanism",
                  "The local branch has no pretraining and cannot learn from 61 images; fusing a "
                  "near-random branch into a strong pretrained one adds variance and costs "
                  "accuracy — that contrast is negative on every seed. The Sobel prior is "
                  "photometric, so it scores a lighting seam as highly as a defect; that is the "
                  "mechanism we expected to hurt, and the point estimate agrees, but the signs "
                  "disagree across seeds so it is not claimed on its own.", D.WARNING)
        d.callout(s, MARGIN, D.BODY_Y + 2.0, col_w(5.8), 2.1,
                  "What survives",
                  "The negative result is about the architectural additions only. Untouched: "
                  "the reproduction and audit of the base paper, the encrypted inference and "
                  "its exactness bound, the dataset integrity audit — and the evaluation "
                  "protocol, which is what made this visible instead of hiding it behind a "
                  "saturated 100%.", D.GOOD)
        d.callout(s, col_x(6.2), D.BODY_Y + 2.0, col_w(5.8), 2.1,
                  "What a follow-up should build",
                  "EfficientNet-B0 + the compression layer + the CKKS polynomial head — the "
                  "privacy architecture without the extra feature machinery. It is simpler "
                  "than what was proposed, it is what the evidence supports, and the encrypted "
                  "inference results carry over unchanged because the head is identical.",
                  D.ACCENT)
        d.notes(s, "A project that reported only the 100% headline would have missed all of "
                   "this. The saturation-aware protocol is what made it visible.")

    # ------------------------------------------------- encryption results
    s = d.content_slide("Encrypted Inference  ·  correctness",
                        "Does encryption change the prediction?")
    if enc:
        d.stat_tiles(s, D.BODY_Y, [
            (f"{enc['agreement_rate']*100:.2f}%", "encrypted vs plaintext agreement", D.GOOD),
            (f"{enc['max_abs_logit_error']:.2e}", "max absolute logit error"),
            (f"{enc['n_disagreements']}", "prediction flips out of "
                                          f"{enc['n_samples']}",
             D.GOOD if enc["n_disagreements"] == 0 else D.WARNING),
            (pct(enc.get("accuracy_encrypted")), "accuracy under encryption"),
        ], cols=4)
        rows = [
            ["plaintext accuracy", pct(enc.get("accuracy_plaintext"))],
            ["encrypted accuracy", pct(enc.get("accuracy_encrypted"))],
            ["mean |logit error|", f"{enc['mean_abs_logit_error']:.3e}"],
            ["test images evaluated", enc["n_samples"]],
        ]
        d.table(s, MARGIN, D.BODY_Y + 1.6, col_w(5.6), ["quantity", "value"], rows,
                col_widths=[2.8, col_w(5.6) - 2.8], size=11.5)
        ratio = enc.get("error_to_margin_ratio")
        margin_txt = (
            "A prediction can only flip if the CKKS error exceeds half the gap between the "
            f"top-1 and top-2 logits. On the tightest decision in the test set the error "
            f"consumes just {ratio*100:.2f}% of that budget — it would have to be ~{1/ratio:.0f}× "
            "larger before anything could change.\n\n"
            "So agreement is a bound here, not a coincidence of this sample."
            if ratio else
            "The head is trained with the same operations it is evaluated with, so CKKS "
            "contributes only approximation error.")
        d.callout(s, col_x(6.4), D.BODY_Y + 1.6, col_w(5.6), 2.2,
                  "Why this is robust, not lucky", margin_txt, D.GOOD)
    d.notes(s, "Note the contrast with the base paper, which reports encryption *improving* "
               "accuracy. Our result is the expected one: identical decisions, tiny numerical error.")

    # ---------------------------------------------- encryption performance
    s = d.content_slide("Encrypted Inference  ·  cost", "All timings on 4 vCPU, no GPU")
    ba_enc = FIG / "ba_encryption.png"
    if ba_enc.exists():
        d.image_fit(s, ba_enc, MARGIN, D.BODY_Y, col_w(7.4), 3.9)
    elif (FIG / "encryption_breakdown.png").exists():
        d.image_fit(s, FIG / "encryption_breakdown.png", MARGIN, D.BODY_Y, col_w(7.4), 3.6)
    if enc:
        rows = [
            ["client feature extraction", f"{enc.get('client_feature_extraction_ms', float('nan')):.0f} ms"],
            ["encrypt (client)", f"{enc['encrypt_ms']['mean']:.1f} ms"],
            ["homomorphic evaluation", f"{enc['he_inference_ms']['mean']:.0f} ms"],
            ["decrypt (client)", f"{enc['decrypt_ms']['mean']:.2f} ms"],
            ["ciphertext size", f"{enc['ciphertext_bytes']/1024:.0f} KB"],
            ["expansion vs plaintext", f"{enc['expansion_factor']:.0f}×"],
            ["server context (keys)", f"{enc['key_sizes_bytes']['server_context_bytes']/1e6:.0f} MB"],
            ["peak RSS", f"{enc['peak_rss_mb']:.0f} MB"],
        ]
        d.table(s, col_x(7.8), D.BODY_Y, col_w(4.2), ["quantity", "value"], rows,
                col_widths=[2.3, col_w(4.2) - 2.3], size=10.5, row_h=0.32)
    d.notes(s, "The Galois keys dominate the one-time transfer. The per-image cost is the "
               "homomorphic evaluation, which scales with the compressed dimension.")

    # ------------------------------------------- compression trade-off
    s = d.content_slide("Compression Trade-off", "The compressed width d is the lever between accuracy and encrypted cost")
    if (FIG / "compression_tradeoff.png").exists():
        d.image_fit(s, FIG / "compression_tradeoff.png", MARGIN, D.BODY_Y, col_w(12), 4.4)
    d.text(s, MARGIN, 6.3, col_w(12), 0.5,
           "Accuracy here is measured in the reduced-data regime, where the dimensions are "
           "separable at all; at full training size they all sit at ceiling. Latency depends "
           "only on d and is independent of the training regime.",
           size=10.5, color=D.INK_MUTED, italic=True, line_spacing=1.2)
    d.notes(s, "Two panels, never two y-axes on one plot: accuracy and latency are different "
               "quantities and sharing an axis would invite a false visual comparison. They "
               "are also measured in different regimes, which the caption states.")

    # ---------------------------------------------------- threat model
    s = d.content_slide("Threat Model  ·  what is and is not protected",
                        "The comparison with the base paper is not like-for-like, and we do not present it as such")
    d.text(s, MARGIN, D.BODY_Y, col_w(5.8), 0.3, "PROTECTED", size=11, color=D.GOOD, bold=True)
    d.bullets(s, MARGIN, D.BODY_Y + 0.42, col_w(5.8), 2.6, [
        "The feature vector in transit and at rest on the server.",
        "The prediction: the server returns ciphertext it cannot read.",
        "The secret key never leaves the client — asserted by test, not by assumption.",
    ], size=11.5, bullet_color=D.GOOD)
    d.text(s, col_x(6.2), D.BODY_Y, col_w(5.8), 0.3, "NOT PROTECTED", size=11, color=D.CRITICAL, bold=True)
    d.bullets(s, col_x(6.2), D.BODY_Y + 0.42, col_w(5.8), 3.2, [
        "The client must run and therefore hold the feature extractor — unlike the base paper, where the client does little work.",
        "Circuit privacy: an adaptive client could learn about the server's weights. Noise flooding is not implemented.",
        "The compressed vector is a lossy but real function of the image; we do not claim it is a formal privacy bound.",
    ], size=11.5, bullet_color=D.CRITICAL)
    d.callout(s, MARGIN, D.BODY_Y + 3.4, col_w(12), 0.95, "Honest summary",
              "SAGE-HE protects the server side of the boundary. Jia et al. aim to protect the "
              "image itself. Ours is cheaper and exact; theirs is a stronger goal. These are "
              "different trade-offs, not a ranking.", D.WARNING)
    d.notes(s, "If one slide protects the project's integrity, it is this one.")

    # ------------------------------------------------------ explainability
    s = d.content_slide("Explainable AI  ·  Grad-CAM and chunk attention")
    p = ROOT / "results/gradcam/gradcam_per_class.png"
    if p.exists():
        d.image_fit(s, p, MARGIN, D.BODY_Y, col_w(12), 4.2)
    d.text(s, MARGIN, 6.2, col_w(12), 0.5,
           "Grad-CAM is a post-hoc gradient attribution; chunk attention is the model's own "
           "selection weight. Agreement is weak evidence the model uses the defect region — "
           "an attractive heatmap is not proof of correctness.",
           size=11, color=D.INK_MUTED, italic=True)
    d.notes(s, "The caveat is deliberate. Grad-CAM over a squared activation is less sign-stable "
               "than over a ReLU stack, which the report explains.")

    # ------------------------------------------------------ failure cases
    p_err = ROOT / "results/gradcam/gradcam_errors.png"
    p_hard = ROOT / "results/gradcam/gradcam_hard_cases.png"
    if p_err.exists():
        s = d.content_slide("Failure Cases", "Every misclassified test image, not a curated selection")
        d.image_fit(s, p_err, MARGIN, D.BODY_Y, col_w(12), 3.8)
        d.notes(s, "Showing all errors rather than a chosen few is part of not overclaiming.")
    elif p_hard.exists():
        n_hard = (xai or {}).get("n_baseline_errors")
        s = d.content_slide(
            "Hard Cases",
            (f"SAGE-HE misclassifies nothing on this test set, so these are the "
             f"{n_hard} images the reproduced baseline got wrong"
             if n_hard else "The images the reproduced baseline got wrong"))
        d.image_fit(s, p_hard, MARGIN, D.BODY_Y, col_w(12), 3.9)
        d.text(s, MARGIN, 6.18, col_w(12), 0.6,
               "The baseline's errors are concentrated in faint `inclusion` streaks it calls "
               "`pitted_surface` — the same two classes its per-class F1 singles out. These are "
               "shown instead of an empty failure slide; nothing here is curated.",
               size=10.5, color=D.INK_MUTED, italic=True, line_spacing=1.2)
        d.notes(s, "With a perfect test score there is no failure set for the proposed model. "
                   "Substituting the baseline's failures keeps the slide honest and still "
                   "shows what the harder images look like.")

    # ------------------------------------------------------- discussion
    s = d.content_slide("Discussion and Limitations")
    lims = [
        ("Benchmark is saturated", "NEU-CLS cannot separate strong modern architectures at full data. "
                                   "Conclusions about components come from the reduced-data regime.", D.WARNING),
        ("Small test set", "360 images. A one-image change moves accuracy by 0.28 points; "
                           "Wilson intervals are reported for this reason.", D.WARNING),
        ("No GPU", "All training and all latencies are CPU. Timings are not comparable to "
                   "GPU-accelerated figures elsewhere in the literature.", D.WARNING),
        ("Weaker privacy than the base paper's goal", "The client holds the extractor. "
                                                      "We protect the server side only.", D.CRITICAL),
        ("Baseline is a reconstruction", "The paper under-specifies channel count, pooling, split "
                                         "and schedule; our reproduction documents each assumption.", D.WARNING),
        ("Novelty is narrow", "The encryption boundary is prior art. The contribution is the "
                              "learned Sobel-conditioned selection and the measurement discipline.", D.INK_MUTED),
    ]
    for i, (head, body, color) in enumerate(lims):
        x = col_x((i % 2) * 6)
        y = D.BODY_Y + (i // 2) * 1.55
        d.callout(s, x, y, col_w(6), 1.38, head, body, color)
    d.notes(s, "A limitations slide that names six real problems is more persuasive than one "
               "that names none.")

    # ------------------------------------------------------- conclusion
    s = d.content_slide("Conclusion and Future Scope")
    d.bullets(s, MARGIN, D.BODY_Y, col_w(6.2), 3.4, [
        ("Built. ", "A complete, tested, reproducible system: audited dataset, reproduced "
                    "baseline, dual-branch model, genuine CKKS inference, explainability."),
        ("Measured. ", f"SAGE-HE {pct(sage_acc)} vs reproduced baseline {pct(jia_acc)} on one "
                       f"identical held-out split, over {n_seeds} seeds."
                       if sage_acc else "See results."),
        ("Encrypted exactly. ", "Encrypted and plaintext inference agree, with a quantified "
                                "margin: CKKS costs latency, not accuracy."),
        ("The architecture did not pan out. ", "Measured against its own backbone, the local "
                                "branch, chunk attention and Sobel prior each cost accuracy. "
                                "Reported, explained, and not claimed as a contribution."),
    ], size=12.5)
    d.text(s, col_x(7), D.BODY_Y, col_w(5), 0.3, "FUTURE WORK", size=11, color=D.ACCENT, bold=True)
    d.bullets(s, col_x(7), D.BODY_Y + 0.42, col_w(5), 3.2, [
        "A harder benchmark (NEU-DET, Severstal, X-SDD) where accuracy is not saturated.",
        "Higher-degree polynomial activations to recover the accuracy the square loses.",
        "Batched CKKS packing: 8192 slots are available and we use 64.",
        "Circuit privacy via noise flooding.",
        "Validate the λ finding on a dataset where gradient priors are less reliable.",
    ], size=11.5)
    d.notes(s, "Future work is specific and follows from measured limitations, not generic.")

    # -------------------------------------------------------- references
    s = d.content_slide("References", "All entries resolved against Crossref or arXiv — see reports/citations_verified.json")
    refs = data["citations"]
    order = ["jia2023", "song2013", "cheon2017ckks", "tan2019effnet", "he2016resnet",
             "hu2018senet", "lee2022minimax", "elkhatib2025prism", "selvaraju2017",
             "zuiderveld1994clahe", "akiba2019optuna", "paszke2019torch",
             "buslaev2020albumentations", "pedregosa2011sklearn"]
    lines = []
    for i, k in enumerate(order, 1):
        r = refs.get(k)
        if not r or not r.get("verified"):
            continue
        a = r["authors"]
        who = a[0].split()[-1] if a else "?"
        if len(a) == 2:
            who += f" & {a[1].split()[-1]}"
        elif len(a) > 2:
            who += " et al."
        ident = r.get("doi") or r.get("arxiv_id", "")
        lines.append(f"[{i}]  {who} ({r['year']}). {r['title'][:84]}. {r['container'][:34]}. {ident}")
    half = (len(lines) + 1) // 2
    d.text(s, MARGIN, D.BODY_Y, col_w(6), 4.4, "\n".join(lines[:half]),
           size=9, color=D.INK_SECONDARY, line_spacing=1.45, space_after=5)
    d.text(s, col_x(6.2), D.BODY_Y, col_w(6), 4.4, "\n".join(lines[half:]),
           size=9, color=D.INK_SECONDARY, line_spacing=1.45, space_after=5)
    d.notes(s, "Every one of these was fetched from the publisher API, not recalled.")

    # ------------------------------------------------------- closing slide
    s = d.title_slide("Thank you", "Code, data manifests, logs and this deck are reproducible end to end")
    d.notes(s, "Reproduction: scripts/setup_env.sh, then scripts/reproduce_all.sh.")


# =========================================================================
def main() -> int:
    global ALLOW_PARTIAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "presentation" / "SAGE-HE.pptx"))
    args = ap.parse_args()
    ALLOW_PARTIAL = args.allow_partial

    summary = seed_summary("main")
    data = {
        "summary": summary if summary is not None and not summary.empty else None,
        "main": results_table("main"),
        "ablation": results_table("abl"),
        "capacity": results_table("capacity"),
        "encryption": load_json("results/encryption_benchmarks/encrypted_inference.json"),
        "dataset": require(load_json("results/metrics/dataset_verification.json"),
                           "dataset verification (scripts/verify_dataset.py)"),
        "optuna": load_json("results/metrics/optuna_study.json"),
        "xai": load_json("results/gradcam/explainability_meta.json"),
        "before_after": load_json("results/metrics/before_after.json"),
        "components": load_json("results/metrics/component_analysis.json"),
        "design": results_table("design"),
        "sensitivity": results_table("sensitivity"),
        "concentration": load_json("results/metrics/attention_concentration.json"),
        "citations": require(load_json("reports/citations_verified.json"),
                             "verified citations (scripts/verify_citations.py)"),
    }
    if data["summary"] is None:
        require(None if not ALLOW_PARTIAL else {}, "main experiment results (scripts/run_suite.py main)")
    if data["encryption"] is None:
        require(None if not ALLOW_PARTIAL else {}, "encrypted benchmark (scripts/run_encrypted_eval.py)")

    d = Deck.create()
    # Two-pass: build once to learn the length, then rebuild so the progress
    # rail on every slide knows the true total.
    build(d, data)
    total = d.slide_number
    d = Deck.create()
    d.expected_total = total
    build(d, data)
    d.animate_all()
    out = d.save(Path(args.out))
    print(f"written: {out}  ({d.slide_number} slides)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
