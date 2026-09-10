"""GradeForge Phase 1 - environment setup and verification.

    . .\\env.ps1
    python setup_env.py                  # full setup
    python setup_env.py --skip-install   # verify + download models only

Steps: Python/driver checks -> pip install -> Ollama (pull + measure real VRAM) ->
PyTorch CUDA check -> download TrOCR + embedder -> smoke-test both on the GPU.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

# This script is the one place allowed to download from HuggingFace.
os.environ["PAPERMIND_ALLOW_DOWNLOADS"] = "1"

from src import config  # noqa: E402  (redirects caches/temp before any heavy import)

MIN_DRIVER_CUDA = (13, 2)
MIN_OLLAMA = (0, 34, 0)
NUM_CTX = 8192
HANDWRITING_FONT = r"C:\Windows\Fonts\segoepr.ttf"  # Segoe Print - read-only use


class SetupError(RuntimeError):
    pass


def step(title: str) -> None:
    print(f"\n== {title}", flush=True)


def ok(msg: str) -> None:
    print(f"  [ok] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"  [!!] {msg}", flush=True)


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", text)[:3])


def check_python() -> None:
    step("Python")
    v = sys.version_info
    if v[:2] < (3, 14):
        raise SetupError(f"Python 3.14+ required, found {sys.version.split()[0]}")
    if v[:3] == (3, 14, 1):
        raise SetupError("Python 3.14.1 is excluded by torchvision 0.29 - use 3.14.0 or a later patch")
    if sys.prefix == sys.base_prefix:
        warn("not running inside a virtual environment (.venv)")
    ok(f"Python {sys.version.split()[0]} at {sys.executable}")


def check_driver() -> None:
    step("NVIDIA driver")
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise SetupError("nvidia-smi failed - is the NVIDIA driver installed?") from e
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    if not match:
        raise SetupError("could not read the driver's CUDA version from nvidia-smi")
    cuda = (int(match[1]), int(match[2]))
    if cuda < MIN_DRIVER_CUDA:
        raise SetupError(f"driver supports CUDA {cuda[0]}.{cuda[1]}; torch cu132 needs 13.2+ - update the driver")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    ok(f"{gpu} - driver CUDA {cuda[0]}.{cuda[1]} (no toolkit needed)")


def install_requirements() -> None:
    step("pip install -r requirements.txt")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(config.PROJECT_ROOT / "requirements.txt")],
        check=True,
    )
    ok("requirements installed")


def setup_ollama(pull: bool) -> None:
    step("Ollama")
    from src.models.ollama_probe import OllamaModelProbe, OllamaUnavailable, save_measurement

    probe = OllamaModelProbe()
    try:
        version = probe.version()
    except OllamaUnavailable as e:
        raise SetupError(f"{e} - start the Ollama app and re-run") from e
    if _version_tuple(version) < MIN_OLLAMA:
        warn(f"Ollama {version} is older than {'.'.join(map(str, MIN_OLLAMA))} - update recommended")
    else:
        ok(f"Ollama {version} at {config.OLLAMA_URL}")

    installed = probe.list_models()
    for model in (config.DEFAULT_LLM, config.SECONDARY_LLM):
        if model not in installed:
            if not pull:
                warn(f"{model} not installed (run without --no-pull to fetch it)")
                continue
            print(f"  pulling {model} ...", flush=True)
            _pull_with_progress(probe, model)
            ok(f"pulled {model}")

        identity = probe.probe(model)
        measurement = probe.measure_vram(model, NUM_CTX)
        save_measurement(measurement)
        probe.unload(model)
        placement = "100% GPU" if measurement.fully_on_gpu else f"{measurement.gpu_fraction:.0%} GPU (rest on CPU)"
        ok(
            f"{model}: {identity.architecture}, {identity.parameter_size}, {identity.quantization}, "
            f"ctx {identity.context_length}, caps={list(identity.capabilities)} - "
            f"loaded {measurement.size_bytes / 2**30:.2f} GiB @ num_ctx={NUM_CTX}, {placement}"
        )


def _pull_with_progress(probe, model: str) -> None:
    last_pct: dict[str, int] = {}

    def on_progress(status: str, completed: int, total: int) -> None:
        if total:
            pct = completed * 100 // total
            if pct >= last_pct.get(status, -10) + 10:
                last_pct[status] = pct
                print(f"    {status[:40]:40} {pct:3d}% of {total / 2**30:.2f} GiB", flush=True)
        elif status and status not in last_pct:
            last_pct[status] = 0
            print(f"    {status}", flush=True)

    probe.pull(model, on_progress)


def check_torch() -> None:
    step("PyTorch + CUDA")
    import torch

    if not torch.cuda.is_available():
        raise SetupError(f"torch {torch.__version__} cannot see the GPU - was the cu132 wheel installed?")
    free, total = torch.cuda.mem_get_info()
    ok(
        f"torch {torch.__version__} (CUDA {torch.version.cuda}) on {torch.cuda.get_device_name(0)} - "
        f"{free / 2**30:.1f} / {total / 2**30:.1f} GiB free"
    )


def download_models() -> None:
    step("HuggingFace models")
    from huggingface_hub import snapshot_download

    start = time.time()
    snapshot_download(
        config.TROCR_REPO,
        local_dir=config.TROCR_BASE_DIR,
        allow_patterns=["*.json", "*.txt", "model.safetensors"],
    )
    ok(f"{config.TROCR_REPO} -> {config.TROCR_BASE_DIR}")
    for name, repo in config.EMBEDDER_REPOS.items():
        snapshot_download(
            repo,
            local_dir=config.EMBEDDERS_DIR / name,
            allow_patterns=["*.json", "*.txt", "model.safetensors", "1_Pooling/*"],
        )
        ok(f"{repo} -> {config.EMBEDDERS_DIR / name}")
    ok(f"downloads done in {time.time() - start:.0f}s")


def _render_line(text: str):
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype(HANDWRITING_FONT, 44)
    except OSError:
        font = ImageFont.load_default(size=44)
    left, top, right, bottom = font.getbbox(text)
    image = Image.new("RGB", (right - left + 40, bottom - top + 40), "white")
    ImageDraw.Draw(image).text((20 - left, 20 - top), text, font=font, fill="black")
    return image


def smoke_test_trocr() -> None:
    step("TrOCR smoke test (GPU, offline load)")
    import jiwer
    import torch

    from src.ocr.text_extractor import TrOCRExtractor

    torch.cuda.reset_peak_memory_stats()
    extractor = TrOCRExtractor()
    expected = "The mitochondria is the powerhouse of the cell"
    start = time.time()
    [reading] = extractor.read_lines([_render_line(expected)])
    elapsed = time.time() - start
    peak = torch.cuda.max_memory_allocated() / 2**30
    extractor.unload()

    ok(f"read: '{reading.text}' (confidence {reading.confidence:.2f})")
    ok(f"expected: '{expected}' - CER {jiwer.cer(expected, reading.text):.1%}, "
       f"{elapsed:.2f}s, peak VRAM {peak:.2f} GiB")


def smoke_test_embedder() -> None:
    step("Embedder smoke test")
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(config.EMBEDDERS_DIR / config.DEFAULT_EMBEDDER), device="cuda")
    reference, close, far = model.encode(
        [
            "Photosynthesis converts light energy into chemical energy in chloroplasts.",
            "Plants use sunlight in their chloroplasts to make chemical energy.",
            "The French Revolution began in 1789.",
        ],
        normalize_embeddings=True,
    )
    ok(f"similar answer: {float(reference @ close):.2f}   unrelated answer: {float(reference @ far):.2f}")
    del model
    torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-install", action="store_true", help="don't run pip install")
    parser.add_argument("--skip-ollama", action="store_true", help="skip Ollama checks and VRAM measurement")
    parser.add_argument("--no-pull", action="store_true", help="don't pull missing Ollama models")
    parser.add_argument("--skip-models", action="store_true", help="skip HuggingFace downloads and smoke tests")
    args = parser.parse_args()

    config.ensure_dirs()
    try:
        check_python()
        check_driver()
        if not args.skip_install:
            install_requirements()
        # Measure Ollama VRAM before this process creates its own CUDA context.
        if not args.skip_ollama:
            setup_ollama(pull=not args.no_pull)
        check_torch()
        if not args.skip_models:
            download_models()
            smoke_test_trocr()
            smoke_test_embedder()
    except SetupError as e:
        print(f"\nSETUP FAILED: {e}", file=sys.stderr)
        return 1

    print("\nGradeForge environment ready. HuggingFace runs offline from now on "
          "(set PAPERMIND_ALLOW_DOWNLOADS=1 to download again).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
