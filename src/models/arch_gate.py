"""Layer 3 of model resolution: can the fine-tuning pipeline actually handle this architecture?

Gated on real library support, not on a list:
- transformers: the HF config's `model_type` must be in the installed CONFIG_MAPPING_NAMES
  (checked offline from the type the resolver read from the HF API).
- GGUF conversion + Ollama: the model already runs in Ollama as a GGUF of this architecture,
  so llama.cpp's converter and Ollama's loader both support it. A fine-tune of the same
  architecture exports and loads the same way.
- Unsloth: training runs on Colab/Kaggle with the latest Unsloth, which can't be checked from
  here; the exported notebook verifies it in its first cell. Locally, the pinned .venv-train
  (if installed) is checked for the architecture.
"""
from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass

from src import config
from src.models.hf_resolver import HFResolution
from src.models.ollama_probe import ModelIdentity

TRAIN_ENV_PYTHON = config.PROJECT_ROOT / ".venv-train" / "Scripts" / "python.exe"


@dataclass(frozen=True)
class ComponentCheck:
    component: str
    status: str   # pass | fail | unknown
    detail: str


@dataclass(frozen=True)
class TrainabilityResult:
    repo: str | None
    model_type: str | None
    checks: tuple[ComponentCheck, ...]

    @property
    def supported(self) -> bool:
        return self.repo is not None and not any(c.status == "fail" for c in self.checks)

    @property
    def blocker(self) -> ComponentCheck | None:
        return next((c for c in self.checks if c.status == "fail"), None)

    def to_dict(self) -> dict:
        return asdict(self) | {"supported": self.supported}


def transformers_check(model_type: str | None) -> ComponentCheck:
    import transformers
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    version = transformers.__version__
    if not model_type:
        return ComponentCheck("transformers", "fail", "the HuggingFace repo has no config with a model type")
    if model_type in CONFIG_MAPPING_NAMES:
        return ComponentCheck("transformers", "pass", f"transformers {version} supports {model_type}")
    return ComponentCheck("transformers", "fail",
                          f'transformers {version} doesn\'t know "{model_type}" yet; try: pip install -U transformers')


def ollama_check(identity: ModelIdentity | None) -> ComponentCheck:
    if identity is None or not identity.architecture:
        return ComponentCheck("GGUF export + Ollama", "fail", "not an Ollama model, so there is no GGUF path")
    return ComponentCheck("GGUF export + Ollama", "pass",
                          f"Ollama already runs {identity.architecture} GGUF files, so a fine-tune will convert and load")


def unsloth_check(model_type: str | None, train_python=TRAIN_ENV_PYTHON, run=subprocess.run) -> ComponentCheck:
    if not train_python.exists():
        return ComponentCheck("Unsloth", "unknown",
                              "checked by the exported notebook's first cell (Colab installs the latest Unsloth)")
    code = ("import sys; from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as m; "
            "sys.exit(0 if sys.argv[1] in m else 3)")
    try:
        done = run([str(train_python), "-c", code, model_type or ""], capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        return ComponentCheck("Unsloth", "unknown", f"could not run the training environment ({type(e).__name__})")
    if done.returncode == 0:
        return ComponentCheck("Unsloth", "pass", f"the local training environment supports {model_type}")
    if done.returncode == 3:
        return ComponentCheck("Unsloth", "fail",
                              f"the local training environment (.venv-train) doesn't support {model_type}; "
                              "update requirements-train.txt")
    return ComponentCheck("Unsloth", "unknown", "the local training environment failed to start")


class ArchitectureGate:
    def __init__(self, train_python=TRAIN_ENV_PYTHON, run=subprocess.run):
        self.train_python = train_python
        self.run = run

    def check(self, resolution: HFResolution | None, identity: ModelIdentity | None) -> TrainabilityResult:
        if resolution is None or not resolution.resolved:
            reason = resolution.reasons[0] if resolution and resolution.reasons else "no trainable HF source found"
            return TrainabilityResult(None, None, (ComponentCheck("HuggingFace source", "fail", reason),))
        checks = [
            ComponentCheck("HuggingFace source", "pass", f"{resolution.repo} ({resolution.confidence} confidence)"),
            transformers_check(resolution.model_type),
            unsloth_check(resolution.model_type, self.train_python, self.run),
            ollama_check(identity),
        ]
        if resolution.gated:
            checks.insert(1, ComponentCheck("HuggingFace access", "unknown",
                                            "gated repo: accept its licence on huggingface.co and log in inside the notebook"))
        return TrainabilityResult(resolution.repo, resolution.model_type, tuple(checks))
