"""Model manager: the three-layer resolution chain, capability probe and training router put
together for the Settings and Learning pages.

Listing models is cache-only (Ollama metadata + cached probe reports and HF resolutions), so it
never waits on the network or a model. Probing and resolving are explicit teacher actions.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from src import config
from src.models.arch_gate import ArchitectureGate, TrainabilityResult
from src.models.cache import ModelCache
from src.models.capability_prober import CapabilityProber, ProbeReport, assign_tier, cached_report
from src.models.hf_resolver import HFResolution, HFSourceResolver
from src.models.ollama_probe import ModelIdentity, OllamaModelProbe, load_measurement
from src.models.override_registry import load_overrides
from src.models.secret_store import SecretStore
from src.models.training_router import Hardware, TrainingPlan, TrainingRouter, detect_hardware

HF_TOKEN_NAME = "huggingface"
_OLLAMA_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")
_MODELFILE_KEEP = ("TEMPLATE", "RENDERER", "PARSER", "PARAMETER", "SYSTEM")


def gpu_usage() -> dict | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10,
                             check=True).stdout.strip().splitlines()[0]
        name, total, used = [x.strip() for x in out.rsplit(",", 2)]
        return {"gpu": name, "total_gb": round(float(total) / 1024, 2), "used_gb": round(float(used) / 1024, 2)}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class ModelManager:
    def __init__(self, cache: ModelCache | None = None, probe: OllamaModelProbe | None = None,
                 secrets: SecretStore | None = None, hardware: Hardware | None = None,
                 gate: ArchitectureGate | None = None, resolver: HFSourceResolver | None = None):
        self.cache = cache or ModelCache()
        self.probe = probe or OllamaModelProbe(timeout=15)
        self.secrets = secrets or SecretStore()
        self._hardware = hardware
        self.gate = gate or ArchitectureGate()
        self._resolver = resolver
        self._own_resolver = resolver is None  # an injected resolver (tests) is kept

    @property
    def hardware(self) -> Hardware:
        if self._hardware is None:
            self._hardware = detect_hardware()
        return self._hardware

    @property
    def resolver(self) -> HFSourceResolver:
        if self._resolver is None:
            self._resolver = HFSourceResolver(cache=self.cache, token=self.secrets.get(HF_TOKEN_NAME))
        return self._resolver

    def token_changed(self) -> None:
        """The HF token was saved or removed: the next resolution uses the new one."""
        if self._own_resolver:
            self._resolver = None

    # --- the chain, from caches ----------------------------------------------------------
    def report(self, name: str, digest: str | None = None) -> ProbeReport | None:
        return cached_report(name, digest, self.cache)

    def resolution(self, name: str) -> HFResolution | None:
        return self.resolver.cached(name)

    def trainability(self, identity: ModelIdentity) -> TrainabilityResult | None:
        resolution = self.resolution(identity.name)
        return self.gate.check(resolution, identity) if resolution else None

    def training_plan(self, identity: ModelIdentity, corrections: int) -> TrainingPlan | None:
        trainability = self.trainability(identity)
        if trainability is None:
            return None
        resolution = self.resolution(identity.name)
        params = (resolution.parameter_count if resolution else None) or identity.parameter_count
        router = TrainingRouter(self.hardware, load_overrides(self.cache))
        return router.plan(identity.name, identity.family, trainability, params, corrections)

    def summary(self, identity: ModelIdentity, digest: str | None, corrections: int = 0) -> dict:
        report = self.report(identity.name, digest)
        plan = self.training_plan(identity, corrections)
        vram = load_measurement(identity.name, config.LLM_NUM_CTX, self.cache.path)
        tier = assign_tier(report, plan)
        return {
            "name": identity.name, "family": identity.family, "architecture": identity.architecture,
            "parameter_size": identity.parameter_size, "quantization": identity.quantization,
            "context_length": identity.context_length, "capabilities": list(identity.capabilities),
            "vram_gib": round(vram.size_bytes / 2**30, 2) if vram else None,
            "fully_on_gpu": vram.fully_on_gpu if vram else None,
            "default": identity.name == config.DEFAULT_LLM,
            "tier": tier.__dict__, "checked": report is not None,
            "vision_ok": report.vision_ok if report else None,
        }

    def list_models(self, corrections: int = 0) -> list[dict]:
        digests = self.probe.digests()
        return [self.summary(self.probe.probe(name), digest, corrections) for name, digest in digests.items()]

    def card(self, name: str, corrections: int = 0) -> dict:
        identity = self.probe.probe(name)
        digest = self.probe.digests().get(name)
        report = self.report(name, digest)
        resolution = self.resolution(name)
        trainability = self.gate.check(resolution, identity) if resolution else None
        plan = self.training_plan(identity, corrections)
        return self.summary(identity, digest, corrections) | {
            "probe": report.to_dict() if report else None,
            "resolution": resolution.to_dict() if resolution else None,
            "trainability": trainability.to_dict() if trainability else None,
            "training_plan": plan.to_dict() if plan else None,
            "hardware": self.hardware.__dict__,
        }

    # --- explicit actions -----------------------------------------------------------------
    def run_probe(self, name: str, llm, progress: Callable[[float, str], None] | None = None) -> ProbeReport:
        identity = self.probe.probe(name)
        return CapabilityProber(llm, self.cache).probe(name, self.probe.digests().get(name),
                                                       has_vision=identity.has_vision, progress=progress)

    def resolve(self, name: str, refresh: bool = True) -> HFResolution:
        return self.resolver.resolve(self.probe.probe(name), refresh=refresh)

    def set_repo(self, name: str, repo: str | None) -> HFResolution:
        identity = self.probe.probe(name)
        return self.resolver.set_manual(identity, repo) if repo else self.resolver.clear_manual(identity)


# --- importing a fine-tuned GGUF ----------------------------------------------------------

def modelfile_directives(modelfile: str) -> list[str]:
    """Chat template, renderer/parser and parameters from `ollama show --modelfile`, so the
    fine-tune is prompted exactly like its base model (a mismatched template is the most common
    reason an exported model behaves worse). FROM and LICENSE are dropped."""
    kept, lines, i = [], modelfile.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        keyword = line.split(" ", 1)[0].upper()
        block = [line]
        if line.count('"""') == 1:  # multi-line value: runs to the closing """
            i += 1
            while i < len(lines):
                block.append(lines[i])
                if '"""' in lines[i]:
                    break
                i += 1
        if keyword in _MODELFILE_KEEP:
            kept.append("\n".join(block))
        i += 1
    return kept


def next_version(folder: Path) -> str:
    existing = [int(p.name[1:]) for p in folder.glob("v*") if p.name[1:].isdigit()] if folder.exists() else []
    return f"v{max(existing, default=0) + 1}"


def tuned_name(base_model: str) -> str:
    """The Ollama name of GradeForge's fine-tunes of a base model: qwen3.5:9b -> gradeforge-qwen3-5-9b."""
    return "gradeforge-" + re.sub(r"[^a-z0-9]+", "-", base_model.lower()).strip("-")


def _ollama_cli() -> str:
    found = shutil.which("ollama")
    if found:
        return found
    default = Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe"
    if default.exists():
        return str(default)
    raise FileNotFoundError("the ollama command was not found")


def import_gguf(gguf_path: Path, name: str, base_model: str, *, root: Path = config.LLM_CHECKPOINTS_DIR,
                probe: OllamaModelProbe | None = None, run=subprocess.run, hf_repo: str | None = None,
                progress: Callable[[float, str], None] | None = None) -> dict:
    """Copy a GGUF from Colab/Kaggle into models/llm_checkpoints/<name>/v<N>/ and register it
    with Ollama as <name>:v<N>, using the base model's chat template."""
    report = progress or (lambda p, m: None)
    gguf_path = Path(gguf_path)
    if not _OLLAMA_NAME.match(name):
        raise ValueError("model name: lowercase letters, digits, '.', '_' or '-'")
    if gguf_path.suffix.lower() != ".gguf" or not gguf_path.is_file():
        raise ValueError(f"{gguf_path} is not a .gguf file")
    with gguf_path.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{gguf_path.name} is not a valid GGUF file")

    probe = probe or OllamaModelProbe()
    directives = modelfile_directives(probe.show(base_model).get("modelfile", ""))
    version = next_version(root / name)
    folder = root / name / version
    folder.mkdir(parents=True)
    report(0.05, f"Copying {gguf_path.name}")
    shutil.copy2(gguf_path, folder / "model.gguf")
    (folder / "Modelfile").write_text("\n".join(["FROM ./model.gguf", *directives]) + "\n", encoding="utf-8")

    tag = f"{name}:{version}"
    report(0.5, f"Registering {tag} with Ollama")
    done = run([_ollama_cli(), "create", tag, "-f", str(folder / "Modelfile")], cwd=folder,
               capture_output=True, text=True, timeout=3600)
    if done.returncode != 0:
        shutil.rmtree(folder, ignore_errors=True)
        raise RuntimeError(f"ollama create failed: {(done.stderr or done.stdout).strip()[-400:]}")
    meta = {"ollama_name": tag, "base_model": base_model, "hf_repo": hf_repo, "source_file": gguf_path.name,
            "created_at": time.time()}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta | {"path": str(folder)}


def import_safetensors(model_dir: Path, name: str, base_model: str, version: str, *,
                       root: Path = config.LLM_CHECKPOINTS_DIR, probe: OllamaModelProbe | None = None,
                       run=subprocess.run, quantize: str | None = "q4_K_M", meta: dict | None = None) -> dict:
    """Register a merged fine-tune (safetensors) with Ollama as <name>:<version>. Ollama converts and
    quantises it itself (its converter handles Qwen3.5 and Gemma 4, unlike llama.cpp GGUFs with a
    separate vision file), and the base model's renderer/parser/parameters are reused so the
    fine-tune is prompted exactly as it was trained. Ollama keeps its temporary files in its own
    model folder."""
    model_dir = Path(model_dir).resolve()
    if not _OLLAMA_NAME.match(name):
        raise ValueError("model name: lowercase letters, digits, '.', '_' or '-'")
    if not (model_dir / "config.json").exists() or not any(model_dir.glob("*.safetensors")):
        raise ValueError(f"{model_dir} is not a safetensors model folder")
    probe = probe or OllamaModelProbe()
    directives = modelfile_directives(probe.show(base_model).get("modelfile", ""))
    folder = root / name / version
    folder.mkdir(parents=True, exist_ok=True)
    modelfile = folder / "Modelfile"
    modelfile.write_text("\n".join([f"FROM {model_dir}", *directives]) + "\n", encoding="utf-8")
    tag = f"{name}:{version}"
    command = [_ollama_cli(), "create", tag, "-f", str(modelfile)] + (["--quantize", quantize] if quantize else [])
    done = run(command, cwd=folder, capture_output=True, text=True, timeout=4 * 3600)
    if done.returncode != 0:
        raise RuntimeError(f"ollama create failed: {(done.stderr or done.stdout).strip()[-400:]}")
    record = {"ollama_name": tag, "base_model": base_model, "created_at": time.time(), **(meta or {})}
    (folder / "meta.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record | {"path": str(folder)}
