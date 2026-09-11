"""Layer 2 of model resolution: find the HuggingFace repo holding an Ollama model's trainable
weights. GGUF is inference-only, so fine-tuning needs the original safetensors.

No model list: candidates come from the HF search API and are ranked on evidence.
- Parameter count is the strongest signal (plan gap #4: config.json has no count, so this uses
  the API's `safetensors.total`). Measured 2026-09-11: Qwen/Qwen3.5-9B has 9,653,104,368
  parameters, exactly Ollama's `general.parameter_count` for qwen3.5:9b; google/gemma-4-E4B-it
  differs from gemma4:e4b by 1,184 parameters.
- Quantised re-uploads (GGUF/AWQ/GPTQ/FP8/MLX...) keep the same count, so they are penalised.
- A repo other candidates name as their `base_model` is likely the upstream original.
- Third-party derivatives (base_model in another org: merges, "uncensored" finetunes) are penalised.

Only model names leave the machine (search terms); results are cached for 30 days.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import asdict, dataclass, field

import httpx

from src.models.cache import ModelCache
from src.models.ollama_probe import ModelIdentity
from src.models.override_registry import Overrides, load_overrides, refresh_overrides

HF_URL = "https://huggingface.co"
TTL = 30 * 24 * 3600
SEARCH_LIMIT = 50
_EXPAND = ("safetensors", "config", "tags", "downloads", "gated")

OFFICIAL_ORGS = {"google", "meta-llama", "qwen", "microsoft", "mistralai", "deepseek-ai", "ibm-granite",
                 "allenai", "huggingfacetb", "openai", "coherelabs", "tiiuae", "nvidia", "zai-org"}
_QUANT = {"gguf", "awq", "gptq", "bnb", "bitsandbytes", "exl2", "exl3", "mlx", "fp8", "nvfp4", "mxfp4",
          "int4", "int8", "4bit", "8bit", "4-bit", "8-bit", "onnx", "litert", "openvino", "qat", "w4a16",
          "w8a8", "quantized", "gptqmodel", "hqq", "eetq"}
_SIZE = re.compile(r"^(?:e|a)?\d+(?:\.\d+)?[bm]$")


class ResolverUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class HFCandidate:
    repo: str
    score: float
    parameter_count: int | None
    model_type: str | None
    gated: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class HFResolution:
    model: str
    repo: str | None
    confidence: str                      # high | medium | low | none
    source: str                          # auto | override | manual
    parameter_count: int | None          # from HF safetensors
    ollama_parameter_count: int | None
    model_type: str | None               # config.model_type, for the architecture gate
    gated: bool
    reasons: tuple[str, ...]
    candidates: tuple[HFCandidate, ...] = ()
    resolved_at: float = field(default_factory=time.time)

    @property
    def resolved(self) -> bool:
        return self.repo is not None and self.confidence != "none"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> HFResolution:
        data = dict(data)
        data["reasons"] = tuple(data.get("reasons", ()))
        data["candidates"] = tuple(HFCandidate(**{**c, "reasons": tuple(c["reasons"])})
                                   for c in data.get("candidates", ()))
        return cls(**data)


def size_label(model_name: str) -> str | None:
    """'qwen3.5:9b' -> '9b', 'gemma4:e4b' -> 'e4b', 'llama3.1:8b-instruct-q4_K_M' -> '8b'."""
    if ":" not in model_name:
        return None
    for token in re.split(r"[-_]", model_name.rsplit(":", 1)[1].lower()):
        if _SIZE.match(token):
            return token
    return None


def search_terms(identity: ModelIdentity, overrides: Overrides) -> list[str]:
    base = identity.name.rsplit("/", 1)[-1].split(":")[0].lower()
    terms = [*overrides.family(identity.family).get("hf_search", []), base, identity.family.lower()]
    return list(dict.fromkeys(t for t in terms if t))[:3]


def _tokens(repo: str) -> list[str]:
    return re.split(r"[-_]", repo.split("/")[-1].lower())


def _base_models(entry: dict) -> list[str]:
    """Upstream repos from 'base_model:<repo>' and 'base_model:<relation>:<repo>' tags."""
    bases = []
    for tag in entry.get("tags", []):
        if tag.startswith("base_model:"):
            bases.append(tag.split(":")[-1])
    return list(dict.fromkeys(bases))


def _is_quantised(entry: dict) -> bool:
    tags = {t.lower() for t in entry.get("tags", [])}
    return bool(_QUANT & (set(_tokens(entry["id"])) | tags))


def score_candidate(entry: dict, target_params: int | None, size: str | None,
                    upstream: set[str]) -> HFCandidate:
    repo = entry["id"]
    org = repo.split("/")[0].lower()
    tokens = _tokens(repo)
    total = (entry.get("safetensors") or {}).get("total")
    score, reasons = 0.0, []

    if _is_quantised(entry):
        score -= 10
        reasons.append("quantised re-upload, not trainable weights")

    if target_params and total:
        ratio = total / target_params
        if abs(ratio - 1) <= 0.02:
            score += 6
            reasons.append(f"parameter count matches Ollama's ({total / 1e9:.2f}B)")
        elif abs(ratio - 1) <= 0.15:
            score += 3
            reasons.append(f"parameter count within 15% ({total / 1e9:.2f}B vs {target_params / 1e9:.2f}B)")
        elif 1 < ratio <= 1.35:
            score += 1
            reasons.append("larger than Ollama's count: may include a vision tower")
        else:
            score -= 8
            reasons.append(f"different size ({total / 1e9:.2f}B vs {target_params / 1e9:.2f}B)")

    sizes = {t for t in tokens if _SIZE.match(t)}
    if size and size in tokens:
        score += 2
        reasons.append(f"name has the size label '{size}'")
    elif size and sizes:
        score -= 3
        reasons.append(f"name says {'/'.join(sorted(sizes))}, not {size}")

    if org in OFFICIAL_ORGS:
        score += 2
        reasons.append("published by the model's original developer")
    if repo in upstream:
        score += 2
        reasons.append("other repos name it as their base model")
    bases = _base_models(entry)
    foreign = [b for b in bases if b.split("/")[0].lower() != org]
    if foreign:
        score -= 4
        reasons.append(f"a third-party derivative of {foreign[0]}")
    elif bases:
        # Same developer, trained from their pretrained release: the chat/instruct model,
        # which is what Ollama's library tags serve.
        score += 1.5
        reasons.append(f"the developer's instruction-tuned version of {bases[0]}")
    if "base" in tokens:
        score -= 2
        reasons.append("pretrained base, not the chat model Ollama serves")

    score += min(1.0, math.log10((entry.get("downloads") or 0) + 1) / 7)
    return HFCandidate(repo, round(score, 2), total, (entry.get("config") or {}).get("model_type"),
                       bool(entry.get("gated")), tuple(reasons))


def _confidence(best: HFCandidate, target_params: int | None) -> str:
    if best.score <= 0:
        return "none"
    if target_params and best.parameter_count:
        close = abs(best.parameter_count / target_params - 1)
        if close <= 0.02 and not any("derivative" in r or "quantised" in r for r in best.reasons):
            return "high"
        if close <= 0.15:
            return "medium"
    return "low"


class HFSourceResolver:
    def __init__(self, cache: ModelCache | None = None, http: httpx.Client | None = None,
                 token: str | None = None, overrides: Overrides | None = None):
        self.cache = cache or ModelCache()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.http = http or httpx.Client(base_url=HF_URL, timeout=20, follow_redirects=True, headers=headers)
        self._overrides = overrides

    @property
    def overrides(self) -> Overrides:
        if self._overrides is None:
            refresh_overrides(self.cache)
            self._overrides = load_overrides(self.cache)
        return self._overrides

    # --- cache -------------------------------------------------------------------------
    def cached(self, model: str) -> HFResolution | None:
        data = self.cache.get("hf_resolution", model, max_age=TTL)
        return HFResolution.from_dict(data) if data else None

    def set_manual(self, identity: ModelIdentity, repo: str) -> HFResolution:
        """The teacher's "Not right?" choice: verified, then pinned until they clear it."""
        entry = self._model_info(repo)
        if entry is None:
            raise ValueError(f"{repo} was not found on HuggingFace (or it is private)")
        self.cache.put("hf_manual", identity.name, {"repo": entry["id"]})
        return self.resolve(identity, refresh=True)

    def clear_manual(self, identity: ModelIdentity) -> HFResolution:
        self.cache.delete("hf_manual", identity.name)
        return self.resolve(identity, refresh=True)

    # --- HF API ------------------------------------------------------------------------
    def _get(self, path: str, params: list | None = None) -> httpx.Response:
        try:
            return self.http.get(path, params=params)
        except httpx.HTTPError as e:
            raise ResolverUnavailable(f"can't reach huggingface.co ({type(e).__name__})") from e

    def _search(self, term: str) -> list[dict]:
        params = [("search", term), ("sort", "downloads"), ("direction", "-1"), ("limit", str(SEARCH_LIMIT))]
        params += [("expand[]", e) for e in _EXPAND]
        response = self._get("/api/models", params)
        if response.status_code == 429:
            raise ResolverUnavailable("HuggingFace rate limit reached: add an HF token or try later")
        response.raise_for_status()
        return response.json()

    def _model_info(self, repo: str) -> dict | None:
        response = self._get(f"/api/models/{repo}", [("expand[]", e) for e in _EXPAND])
        if response.status_code in (401, 403, 404):  # HF answers 401 for repos that don't exist
            return None
        response.raise_for_status()
        return response.json()

    # --- resolution --------------------------------------------------------------------
    def resolve(self, identity: ModelIdentity, refresh: bool = False) -> HFResolution:
        if not refresh and (hit := self.cached(identity.name)):
            return hit
        target = identity.parameter_count
        size = size_label(identity.name)

        pinned = self.cache.get("hf_manual", identity.name)
        registry_repo = self.overrides.model(identity.name).get("hf_repo")
        if pinned or registry_repo:
            repo = pinned["repo"] if pinned else registry_repo
            entry = self._model_info(repo)
            source = "manual" if pinned else "override"
            if entry is None:
                result = HFResolution(identity.name, None, "none", source, None, target, None, False,
                                      (f"{repo} is not reachable on HuggingFace",))
            elif not (entry.get("safetensors") or {}).get("total"):
                result = HFResolution(identity.name, None, "none", source, None, target, None, False,
                                      (f"{repo} has no safetensors weights to train from",))
            else:
                c = score_candidate(entry, target, size, set())
                result = HFResolution(identity.name, c.repo, _confidence(c, target) if c.score > 0 else "low",
                                      source, c.parameter_count, target, c.model_type, c.gated, c.reasons, (c,))
            self.cache.put("hf_resolution", identity.name, result.to_dict())
            return result

        entries: dict[str, dict] = {}
        for term in search_terms(identity, self.overrides):
            for entry in self._search(term):
                entries.setdefault(entry["id"], entry)
        upstream = {b for e in entries.values() for b in _base_models(e)}
        trainable = [e for e in entries.values() if (e.get("safetensors") or {}).get("total")]
        ranked = sorted((score_candidate(e, target, size, upstream) for e in trainable),
                        key=lambda c: c.score, reverse=True)[:5]

        if not ranked:
            result = HFResolution(identity.name, None, "none", "auto", None, target, None, False,
                                  ("no HuggingFace repo with trainable safetensors weights was found",))
        else:
            best = ranked[0]
            confidence = _confidence(best, target)
            reasons = best.reasons if confidence != "none" else (
                "no candidate matched this model's size and origin closely enough",)
            result = HFResolution(identity.name, best.repo if confidence != "none" else None, confidence,
                                  "auto", best.parameter_count, target, best.model_type, best.gated,
                                  reasons, tuple(ranked))
        self.cache.put("hf_resolution", identity.name, result.to_dict())
        return result
