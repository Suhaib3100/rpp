"""Publication-quality architecture diagrams for SAGE-HE.

Every diagram here corresponds to code that actually exists; tensor shapes are
the real shapes produced by the implementation at 160x160 input (verified by
scripts/verify_diagram_shapes.py).
"""
from __future__ import annotations

from pathlib import Path

from src.viz import theme as T
from src.viz.diagram import arrow, band, canvas, caption, node, region, save, tensor


# ---------------------------------------------------------------- diagram 1
def complete_architecture(out: Path) -> list:
    fig, ax = canvas(
        17, 9.6,
        "SAGE-HE  ·  Complete Proposed Architecture",
        "Dual-branch feature extraction with Sobel-gated chunk selection, channel-wise gated fusion, "
        "and a CKKS-evaluable polynomial head",
    )

    # trust boundary bands. CLIENT spans x 1..81 and must enclose every
    # plaintext stage including compression; SERVER is a narrow vertical stack.
    band(ax, 41, 49, 80, 78, T.ACCENT, 0.05)
    band(ax, 90.5, 49, 17, 78, "#8B5CF6", 0.07)
    region(ax, 41, 49, 80, 78, "CLIENT  ·  plaintext  ·  holds secret key", T.ACCENT)
    region(ax, 90.5, 49, 17, 78, "SERVER  ·  no key", "#8B5CF6")

    img = node(ax, 7, 51, 10, 9, "Input\nimage", "NEU-CLS", "data")
    tensor(ax, 7, 44.4, "(B,1,160,160)")

    # ---- global branch (top row)
    g1 = node(ax, 24, 72, 17, 9.5, "EfficientNet-B0", "ImageNet-pretrained, in_chans=1", "compute")
    tensor(ax, 24, 65.4, "(B,1280)")
    g2 = node(ax, 42, 72, 14, 8, "Linear + LN", "1280 → 256", "compute")

    # ---- local branch (bottom row)
    l1 = node(ax, 24, 30, 17, 9.5, "Multi-scale CNN", "1×1 / 3×3 / 3×3 d=2 / 5×5", "compute")
    tensor(ax, 24, 23.4, "(B,128,10,10)")
    l2 = node(ax, 42, 30, 17, 10, "Sobel-gated\nchunk attention",
              "a = MLP(f) + λ·ê  ·  Linear 128 → 256", "compute")

    sob = node(ax, 24, 16, 17, 7, "Sobel energy ê", "per-chunk gradient, standardised", "frozen")

    # ---- merge column
    fus = node(ax, 60, 51, 14, 10, "Gated fusion", "z·g + (1−z)·l", "compute")
    tensor(ax, 51.5, 44.2, "(B,256)")
    cmp = node(ax, 74, 51, 12, 10, "Compression", "256 → 64\ntanh-bounded", "compute")

    # ---- server stack
    enc = node(ax, 90.5, 76, 14, 8, "CKKS encrypt", "n=16384, scale 2⁴⁰", "crypto")
    h1 = node(ax, 90.5, 62, 14, 7, "W₁·v + b₁", "64 → 32   (depth 1)", "crypto")
    h2 = node(ax, 90.5, 50, 14, 7, "square( · )", "x²        (depth 2)", "crypto")
    h3 = node(ax, 90.5, 38, 14, 7, "W₂·h + b₂", "32 → 6    (depth 3)", "crypto")
    dec = node(ax, 90.5, 24, 14, 8, "Decrypt\n+ softmax", "client-side", "out")

    # ---- wiring
    arrow(ax, img.top, (16, 68), rad=0.12)
    arrow(ax, img.bottom, (16, 34), rad=-0.12)
    arrow(ax, (16, 16), sob.left)
    arrow(ax, g1.right, g2.left)
    arrow(ax, l1.right, l2.left)
    arrow(ax, sob.right, l2.bottom, rad=0.2, dashed=True, color=T.INK_MUTED)
    arrow(ax, g2.right, (60, 56), rad=-0.18, label="g")          # global → fusion
    arrow(ax, l2.right, (60, 46), rad=0.18, label="l", labeloff=-3.4)  # local → fusion
    arrow(ax, fus.right, cmp.left)
    arrow(ax, cmp.right, enc.left, color="#8B5CF6", rad=-0.22,
          label="Enc(v)", labeloff=3.0)
    for a, b in [(enc, h1), (h1, h2), (h2, h3), (h3, dec)]:
        arrow(ax, a.bottom, b.top, color="#8B5CF6")
    caption(ax, "The server evaluates a depth-3 polynomial on ciphertext and never holds the secret key; "
                "only the 64-d compressed vector crosses the boundary.")
    return save(fig, out)


# ---------------------------------------------------------------- diagram 2
def baseline_architecture(out: Path) -> list:
    fig, ax = canvas(
        16, 8.4,
        "Baseline  ·  Jia et al. (2023) chunk-based CNN",
        "Reproduced from the published description; assumptions marked in amber",
    )
    n1 = node(ax, 9, 62, 13, 9, "Input image", "resize 112×112, grayscale", "data")
    n2 = node(ax, 27, 62, 13, 9, "Histogram\nequalisation", "global", "compute")
    n3 = node(ax, 45, 62, 14, 9, "Chunking", "28×28 → 4×4 = 16 chunks", "compute")
    n4 = node(ax, 64, 62, 14, 9, "Sobel per chunk", "threshold τ → keep subset", "compute")
    n5 = node(ax, 85, 62, 15, 9, "CKKS encrypt\nselected chunks", "n=16384 (their Table 1)", "crypto")

    c1 = node(ax, 16, 28, 15, 9, "Conv 7×7", "32 ch, stride 1, pad 3", "compute")
    c2 = node(ax, 36, 28, 11, 8, "activation", "ReLU / square", "compute")
    c3 = node(ax, 54, 28, 12, 8, "MaxPool 2", "→ 56×56", "compute")
    c4 = node(ax, 72, 28, 13, 8, "Adaptive pool", "→ 256 units", "warn")
    c5 = node(ax, 90, 28, 12, 9, "FC 256→128\nFC 128→6", "", "out")

    for a, b in [(n1, n2), (n2, n3), (n3, n4), (n4, n5)]:
        arrow(ax, a.right, b.left)
    arrow(ax, n5.bottom, (16, 33), rad=0.25, color="#8B5CF6")
    for a, b in [(c1, c2), (c2, c3), (c3, c4), (c4, c5)]:
        arrow(ax, a.right, b.left)

    ax.text(72, 19.5, "channel count and the route to 256 units\nare not stated in the paper",
            ha="center", va="top", color=T.WARNING, fontsize=8.5, style="italic")
    caption(ax, "Reported: 97.1% on NEU-CLS with a square activation (their Table 3). "
                "Precision, recall, F1, split sizes and epoch count are not reported.", y=4)
    return save(fig, out)


# --------------------------------------------------------------- diagram 10
def client_server_protocol(out: Path) -> list:
    fig, ax = canvas(
        16, 9,
        "Client–Server Encrypted Inference Protocol",
        "Honest-but-curious server; the secret key never leaves the client",
    )
    region(ax, 23, 50, 40, 74, "CLIENT  (factory)", T.ACCENT)
    region(ax, 77, 50, 40, 74, "SERVER  (cloud)", "#8B5CF6")

    steps_c = [
        (80, "1 · Generate CKKS keys", "secret + public/Galois/relin"),
        (66, "2 · Extract features", "EfficientNet + local + fusion"),
        (52, "3 · Compress to 64-d", "tanh-bounded vector v"),
        (38, "4 · Encrypt v", "Enc(v), 856 KB"),
        (16, "7 · Decrypt + softmax", "predicted defect class"),
    ]
    for y, t, s in steps_c:
        node(ax, 23, y, 30, 9, t, s, "out" if y == 16 else "compute")

    node(ax, 77, 73, 30, 9, "5 · Receive public context", "no secret key present", "crypto")
    node(ax, 77, 52, 30, 13, "6 · Evaluate polynomial head",
         "Enc(W₂·square(W₁v+b₁)+b₂)   depth 3", "crypto")
    node(ax, 77, 32, 30, 9, "Return Enc(logits)", "6 encrypted scores", "crypto")

    arrow(ax, (38, 80), (62, 73), label="public context (122 MB)", color="#8B5CF6", rad=-0.12)
    arrow(ax, (38, 38), (62, 52), label="Enc(v)", color="#8B5CF6", rad=-0.1)
    arrow(ax, (62, 32), (38, 20), label="Enc(logits)", color="#8B5CF6", rad=-0.1)

    ax.text(50, 6.5, "Server learns: nothing about the image or the prediction.\n"
                     "Client learns: the prediction, and must be trusted with the extractor weights.",
            ha="center", va="center", color=T.INK_MUTED, fontsize=9.5, style="italic")
    return save(fig, out)


# --------------------------------------------------------------- diagram 5
def adaptive_chunk_selection(out: Path) -> list:
    fig, ax = canvas(
        16, 8.6,
        "Adaptive Chunk Selection  ·  fixed threshold vs. learned gating",
        "The base paper's rule is photometric and non-differentiable; ours is learned and conditioned on it",
    )
    # --- baseline path
    ax.text(4, 84, "Jia et al. (2023)", color=T.SERIES[1], fontsize=12, fontweight="bold")
    b1 = node(ax, 16, 70, 18, 8, "Sobel gradient per chunk", "", "frozen")
    b2 = node(ax, 40, 70, 18, 8, "keep if gradient > τ", "hard, non-differentiable", "warn")
    b3 = node(ax, 64, 70, 18, 8, "subset of chunks", "τ never reported", "data")
    arrow(ax, b1.right, b2.left, color=T.SERIES[1])
    arrow(ax, b2.right, b3.left, color=T.SERIES[1])

    # --- ours
    ax.text(4, 48, "SAGE-HE", color=T.ACCENT, fontsize=12, fontweight="bold")
    s1 = node(ax, 16, 34, 18, 8, "chunk features fᵢ", "pooled 4×4 grid", "data")
    s2 = node(ax, 16, 20, 18, 7, "Sobel energy êᵢ", "standardised prior", "frozen")
    s3 = node(ax, 43, 27, 22, 10, "aᵢ = MLP(fᵢ) + λ·êᵢ",
              "λ learnable: how far to trust the prior", "compute")
    s4 = node(ax, 72, 27, 18, 10, "w = softmax(a)", "differentiable soft selection", "compute")
    s5 = node(ax, 92, 27, 11, 10, "descriptor", "Σ wᵢfᵢ", "out")
    arrow(ax, s1.right, (32, 29.5))
    arrow(ax, s2.right, (32, 24.5))
    arrow(ax, s3.right, s4.left)
    arrow(ax, s4.right, s5.left)

    caption(ax, "Measured outcome: λ barely moves from its initialisation, so it does not by itself "
                "test the prior. The with/without-Sobel ablation control does — and it is negative.",
            y=5, color=T.WARNING)
    return save(fig, out)


# --------------------------------------------------------------- diagram 7
def gated_fusion_diagram(out: Path) -> list:
    fig, ax = canvas(
        15, 8,
        "Gated Feature Fusion",
        "A channel-wise gate, not a scalar weight: different feature dimensions may come from different branches",
    )
    g = node(ax, 16, 70, 20, 10, "global  g", "EfficientNet-B0 → 256", "compute")
    l = node(ax, 16, 30, 20, 10, "local  l", "multi-scale + attention → 256", "compute")
    cat = node(ax, 42, 50, 14, 9, "concat", "[g ; l] → 512", "data")
    gate = node(ax, 64, 50, 18, 11, "gate network",
                "Linear→ReLU→Linear→σ\n512 → 256", "compute")
    out_n = node(ax, 89, 50, 16, 11, "fused", "z·g + (1−z)·l", "out")

    arrow(ax, g.right, (35, 54), rad=-0.12)
    arrow(ax, l.right, (35, 46), rad=0.12)
    arrow(ax, cat.right, gate.left)
    arrow(ax, gate.right, out_n.left, label="z in (0,1), 256-d")
    arrow(ax, g.right, (81, 56), rad=-0.3, dashed=True, color=T.INK_MUTED)
    arrow(ax, l.right, (81, 44), rad=0.3, dashed=True, color=T.INK_MUTED)

    ax.text(50, 14,
            "Measured against: global only · local only · concatenation · scalar attention.\n"
            "Outcome: the gate does not win. Scalar attention edges it, and the global branch\n"
            "alone beats every fusion variant — see the ablation.",
            ha="center", va="center", color=T.WARNING, fontsize=9.5, style="italic")
    return save(fig, out)


# ---------------------------------------------------------------- diagram 3
def preprocessing_pipeline(out: Path) -> list:
    fig, ax = canvas(16, 8, "Dataset Preprocessing Pipeline",
                     "Deterministic for validation and test; stochastic augmentation on train only")
    stages = [
        (10, "Raw NEU-CLS", "200×200 grayscale", "data"),
        (27, "CLAHE", "clip 2.0, 8×8 tiles", "compute"),
        (44, "Resize", "→ 160×160 (INTER_AREA)", "compute"),
        (61, "Augment (train only)", "rot ±15°, hflip,\ntranslate ±10%, scale ±8%", "compute"),
        (80, "Standardise", "(x/255 − 0.449)/0.226", "out"),
    ]
    prev = None
    for x, t, s, role in stages:
        n = node(ax, x, 62, 15, 12, t, s, role)
        if prev:
            arrow(ax, prev.right, n.left)
        prev = n

    node(ax, 27, 30, 22, 9, "CLAHE vs global equalisation",
         "measured — neither wins", "warn")
    node(ax, 63, 30, 22, 9, "Why no vertical flip",
         "scratches / rolled-in scale carry a\nhorizontal orientation prior", "warn")
    caption(ax, "Contrast enhancement is applied to the raw uint8 image before any geometry, so that "
                "reflect-padding from rotation cannot bias the equaliser's histogram.\n"
                "The design study measures CLAHE, global equalisation and no enhancement; no "
                "enhancement scores highest.", y=9)
    return save(fig, out)


# ---------------------------------------------------------------- diagram 4
def multiscale_detail(out: Path) -> list:
    fig, ax = canvas(15.5, 8.2, "Multi-Scale Local Feature Extraction",
                     "Four parallel receptive fields per block, fused by a 1×1 convolution")
    inp = node(ax, 10, 50, 13, 10, "input", "(B,C,H,W)", "data")
    paths = [
        (76, "1×1", "pointwise mixing", "RF 1"),
        (60, "3×3", "fine cracks, thin scratches", "RF 3"),
        (44, "3×3  d=2", "wider context, 3×3 cost", "RF 5"),
        (28, "5×5", "blob-scale: patches, pits", "RF 5"),
    ]
    nodes = []
    for y, t, s, rf in paths:
        n = node(ax, 36, y, 20, 11, t, s, "compute")
        ax.text(26.5, y + 7.0, rf, color=T.ACCENT_BRIGHT, fontsize=8,
                family="monospace", va="center", ha="center")
        arrow(ax, inp.right, n.left, rad=0.0)
        nodes.append(n)
    cat = node(ax, 64, 52, 11, 42, "concat", "", "data")
    for n in nodes:
        arrow(ax, n.right, (58.5, n.y), rad=0.0)
    fuse = node(ax, 80, 52, 13, 11, "1×1 fuse", "BN + ReLU", "compute")
    res = node(ax, 80, 26, 13, 9, "+ shortcut", "projection if C changes", "frozen")
    arrow(ax, cat.right, fuse.left)
    arrow(ax, res.top, fuse.bottom)
    arrow(ax, inp.bottom, (74, 26), rad=-0.25, dashed=True, color=T.INK_MUTED)
    ax.text(36, 88, "each path emits C/4 channels", color=T.INK_MUTED, fontsize=9, ha="center")
    caption(ax, "Dilated 3×3 reaches a 5×5 receptive field at 3×3 parameter cost — "
                "the branch runs alongside EfficientNet-B0, so its budget matters.", y=9)
    return save(fig, out)


# ---------------------------------------------------------------- diagram 6
def attention_mechanism(out: Path) -> list:
    fig, ax = canvas(15.5, 8.2, "Chunk Attention Mechanism",
                     "Soft, differentiable selection over a 4×4 chunk grid")
    node(ax, 12, 66, 18, 12, "feature map", "(B,128,10,10)", "data")
    node(ax, 12, 36, 18, 11, "adaptive pool", "→ (B,128,4,4)", "compute")
    node(ax, 36, 36, 16, 11, "chunk tokens", "(B,16,128)", "data")
    node(ax, 36, 66, 16, 12, "scoring MLP", "128 → 64 → 1\ntanh hidden", "compute")
    node(ax, 36, 12, 16, 9, "Sobel ê", "standardised", "frozen")

    node(ax, 60, 40, 20, 14, "aᵢ = MLP(fᵢ) + λ·êᵢ", "λ learnable scalar", "compute")
    node(ax, 84, 62, 16, 11, "softmax(a/T)", "weights wᵢ, Σwᵢ = 1", "compute")
    node(ax, 84, 30, 16, 11, "descriptor", "Σᵢ wᵢ fᵢ → (B,128)", "out")

    arrow(ax, (12, 60), (12, 42))
    arrow(ax, (21, 36), (28, 36))
    arrow(ax, (44, 36), (50, 40), rad=0.1)
    arrow(ax, (44, 66), (50, 45), rad=-0.15)
    arrow(ax, (44, 12), (50, 35), rad=0.2, dashed=True, color=T.INK_MUTED)
    arrow(ax, (70, 44), (76, 58), rad=-0.12)
    arrow(ax, (84, 56.5), (84, 35.5))

    node(ax, 84, 86, 16, 8, "optional top-k", "mask all but k chunks", "warn")
    arrow(ax, (84, 82), (84, 67.5), dashed=True, color=T.WARNING)
    caption(ax, "Unlike a hard gradient threshold, every chunk receives gradient signal, "
                "so the selection is trained jointly with the classifier.", y=4)
    return save(fig, out)


# ---------------------------------------------------------------- diagram 8
def compression_diagram(out: Path) -> list:
    fig, ax = canvas(15.5, 7.6, "Feature Compression  ·  the encryption-cost lever",
                     "The compressed width d sets ciphertext size, CKKS latency and rotation count")
    f = node(ax, 13, 62, 16, 11, "fused features", "(B,256)", "data")
    lin = node(ax, 35, 62, 15, 11, "Linear 256 → d", "learned projection", "compute")
    tanh = node(ax, 56, 62, 14, 11, "tanh", "bounds v to [−1,1]", "compute")
    enc = node(ax, 79, 62, 16, 11, "CKKS encrypt", "d slots used of 8192", "crypto")
    for a, b in [(f, lin), (lin, tanh), (tanh, enc)]:
        arrow(ax, a.right, b.left)

    node(ax, 24, 26, 26, 14, "Why tanh?",
         "square() doubles the exponent of its input.\n"
         "Bounding v keeps intermediate magnitudes inside\n"
         "the range where the modulus chain holds precision.", "warn")
    node(ax, 62, 26, 30, 14, "Swept in experiments",
         "d = 32, 64, 128, 256 — accuracy vs.\nciphertext bytes, encrypt time and HE latency.\n"
         "d = 64 selected on the measured trade-off.", "out")
    return save(fig, out)


# ---------------------------------------------------------------- diagram 9
def ckks_workflow(out: Path) -> list:
    fig, ax = canvas(16, 8.4, "CKKS Encryption Workflow and Depth Budget",
                     "A five-prime modulus chain supports exactly the three multiplications the head needs")
    chain = [
        (12, "60-bit", "special prime", "frozen"),
        (30, "40-bit", "level 3", "crypto"),
        (48, "40-bit", "level 2", "crypto"),
        (66, "40-bit", "level 1", "crypto"),
        (84, "60-bit", "special prime", "frozen"),
    ]
    for x, t, s, role in chain:
        node(ax, x, 74, 15, 10, t, s, role)
    for x0, x1 in [(19.5, 22.5), (37.5, 40.5), (55.5, 58.5), (73.5, 76.5)]:
        arrow(ax, (x0, 74), (x1, 74), color=T.INK_MUTED)
    ax.text(48, 86, "coeff_mod_bit_sizes = [60, 40, 40, 40, 60]   ·   total 240 bits   ·   n = 16384",
            ha="center", color=T.ACCENT_BRIGHT, fontsize=10, family="monospace")

    ops = [
        (22, "mm(W₁)", "consumes level 3"),
        (48, "square()", "consumes level 2"),
        (74, "mm(W₂)", "consumes level 1"),
    ]
    for x, t, s in ops:
        node(ax, x, 44, 20, 11, t, s, "compute")
    arrow(ax, (32, 44), (38, 44))
    arrow(ax, (58, 44), (64, 44))
    arrow(ax, (30, 69), (22, 49.5), rad=0.1, dashed=True, color=T.INK_MUTED)
    arrow(ax, (48, 69), (48, 49.5), dashed=True, color=T.INK_MUTED)
    arrow(ax, (66, 69), (74, 49.5), rad=-0.1, dashed=True, color=T.INK_MUTED)

    node(ax, 27, 17, 36, 13, "Security",
         "240 modulus bits at n = 16384;\nthe 128-bit bound for this degree is 438 bits.", "out")
    node(ax, 70, 17, 36, 13, "Verified empirically",
         "a 4th multiplication exhausts the chain and raises —\nasserted in tests/test_encryption.py.", "out")
    return save(fig, out)


# --------------------------------------------------------------- diagram 11
def training_workflow(out: Path) -> list:
    fig, ax = canvas(16, 8, "Model Training Workflow",
                     "The test split is never read during training or model selection")
    region(ax, 30, 56, 56, 56, "TRAIN / VAL ONLY", T.ACCENT)
    region(ax, 82, 56, 30, 56, "TOUCHED ONCE", T.SERIES[2])

    node(ax, 13, 74, 18, 10, "stratified split", "1223 / 216 / 360\nseed 1337", "data")
    node(ax, 13, 48, 18, 10, "dedup by\npixel hash", "no leakage across splits", "frozen")
    node(ax, 36, 74, 18, 10, "train epoch", "AdamW, cosine + warmup\nlabel smoothing 0.05", "compute")
    node(ax, 36, 48, 18, 10, "validate", "macro-F1, loss tiebreak", "compute")
    node(ax, 36, 26, 18, 9, "checkpoint", "best.pt / last.pt (resumable)", "frozen")
    node(ax, 58, 61, 15, 12, "early stop", "patience 8", "warn")
    node(ax, 82, 61, 24, 12, "final test evaluation", "accuracy, macro P/R/F1,\nconfusion, Wilson CI", "out")

    arrow(ax, (22, 74), (27, 74))
    arrow(ax, (22, 48), (27, 48))
    arrow(ax, (36, 69), (36, 53))
    arrow(ax, (36, 43), (36, 30.5))
    arrow(ax, (45, 50), (50.5, 59), rad=-0.1)
    arrow(ax, (58, 67), (36, 79), rad=0.3, dashed=True, color=T.INK_MUTED, label="loop")
    arrow(ax, (65.5, 61), (70, 61), color=T.SERIES[2])
    caption(ax, "Model selection uses validation macro-F1 with validation loss as tiebreak — "
                "NEU-CLS saturates macro-F1 within a few epochs.", y=6)
    return save(fig, out)


# --------------------------------------------------------------- diagram 12
def evaluation_pipeline(out: Path) -> list:
    fig, ax = canvas(16, 8, "Experimental Evaluation Pipeline",
                     "Every reported number traces back to one of these paths")
    src = node(ax, 10, 55, 15, 12, "best.pt", "selected on val", "frozen")
    a = node(ax, 33, 80, 22, 12, "Plaintext metrics", "accuracy, macro P/R/F1,\nper-class, confusion", "compute")
    b = node(ax, 33, 55, 22, 12, "Encrypted inference", "agreement, logit error,\nlatency, ciphertext size", "crypto")
    c = node(ax, 33, 30, 22, 12, "Explainability", "Grad-CAM + attention maps", "compute")
    d = node(ax, 33, 14, 22, 11, "Ablation / design", "component contributions", "compute")

    out_n = node(ax, 68, 55, 20, 14, "results/*.json", "machine-readable,\nregenerated end to end", "out")
    fig_n = node(ax, 90, 76, 16, 11, "figures", "charts + diagrams", "data")
    rep_n = node(ax, 90, 52, 16, 11, "report", "technical_report.md", "data")
    ppt_n = node(ax, 90, 28, 16, 11, "presentation", "SAGE-HE.pptx", "data")

    for n in (a, b, c, d):
        arrow(ax, src.right, n.left, rad=0.0)
        arrow(ax, n.right, out_n.left, rad=0.0)
    for n in (fig_n, rep_n, ppt_n):
        arrow(ax, out_n.right, n.left, rad=0.0)
    caption(ax, "No figure or slide contains a hand-entered number: all are read from results/.", y=3)
    return save(fig, out)


ALL_DIAGRAMS = [
    ("01_complete_architecture", complete_architecture),
    ("02_baseline_architecture", baseline_architecture),
    ("03_preprocessing_pipeline", preprocessing_pipeline),
    ("04_multiscale_extraction", multiscale_detail),
    ("05_adaptive_chunk_selection", adaptive_chunk_selection),
    ("06_attention_mechanism", attention_mechanism),
    ("07_gated_fusion", gated_fusion_diagram),
    ("08_feature_compression", compression_diagram),
    ("09_ckks_workflow", ckks_workflow),
    ("10_client_server_protocol", client_server_protocol),
    ("11_training_workflow", training_workflow),
    ("12_evaluation_pipeline", evaluation_pipeline),
]
