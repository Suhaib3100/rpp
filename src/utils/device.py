"""Hardware detection.

The project was developed on a CPU-only host; everything below is written so
that moving to a CUDA box requires no code change. ``autocast_context`` returns
a real mixed-precision context on CUDA and a no-op elsewhere, because CPU
autocast (bfloat16) measurably *slows* small convolutional models down.
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    name: str
    kind: str          # "cuda" | "mps" | "cpu"
    amp: bool          # mixed precision worth using?
    threads: int

    def __str__(self) -> str:
        return f"{self.kind}:{self.name} (amp={self.amp}, threads={self.threads})"


def get_device(prefer: str | None = None) -> DeviceInfo:
    """Pick the best available accelerator, honouring an explicit override."""
    if prefer and prefer != "auto":
        dev = torch.device(prefer)
        kind = dev.type
    elif torch.cuda.is_available():
        dev, kind = torch.device("cuda"), "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev, kind = torch.device("mps"), "mps"
    else:
        dev, kind = torch.device("cpu"), "cpu"

    if kind == "cuda":
        name = torch.cuda.get_device_name(dev)
        threads = torch.get_num_threads()
    else:
        name = _cpu_name()
        # Oversubscribing threads on a small box thrashes; cap at physical cores.
        threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))
        torch.set_num_threads(threads)

    return DeviceInfo(device=dev, name=name, kind=kind, amp=(kind == "cuda"), threads=threads)


def _cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "cpu"


def autocast_context(info: DeviceInfo):
    """Mixed precision where it pays off, a no-op where it does not."""
    if info.amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()
