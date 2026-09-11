"""Opt-in cloud models: API keys, key checks and cost estimates.

Keys are stored per provider in the OS credential store (see secret_store). Model names and
prices come from LiteLLM's bundled price map (offline copy: LITELLM_LOCAL_MODEL_COST_MAP), so
suggestions stay current with the installed LiteLLM and nothing is hard-coded here. Any other
model name the provider accepts can still be typed in.

Cloud models can never be fine-tuned locally, but few-shot examples and calibration (learning
mechanisms 1 and 2) apply to them exactly as to local models.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from src import config  # noqa: F401  (must precede litellm: disables its price-map download)
from src.grading.answer_key import Question
from src.grading.subjective_grader import build_messages
from src.models.secret_store import SecretStore

PROVIDERS = {"openai": "OpenAI", "anthropic": "Anthropic", "gemini": "Google Gemini"}

# Per-call token assumptions for estimates (measured prompts are ~250-450 tokens + examples).
FEW_SHOT_TOKENS = 3 * 90
OUTPUT_TOKENS = 150
IMAGE_TOKENS = 800
_NOT_CHAT = re.compile(r"realtime|audio|tts|transcribe|search|robotics|image|computer-use|embedding|live|"
                       r"container|lyria|customtools", re.I)
_NEEDS_PREFIX = {"gemini"}  # LiteLLM routes bare "gemini-..." names to Vertex AI, not the Gemini API key
_SNAPSHOT = re.compile(r"-\d{4}-?\d{2}-?\d{2}$|-\d{8}$|-\d{3,4}$")


def _cost_map() -> dict:
    import litellm

    return litellm.model_cost


def chat_models(provider: str, today: date | None = None) -> list[str]:
    """Current chat models for a provider: no fine-tunes, dated snapshots or deprecated names."""
    today = (today or date.today()).isoformat()
    names = []
    for name, info in _cost_map().items():
        if info.get("litellm_provider") != provider or info.get("mode") != "chat":
            continue
        if name.startswith("ft:") or _NOT_CHAT.search(name) or _SNAPSHOT.search(name):
            continue
        prefix = name.split("/", 1)[0] if "/" in name else None
        if (prefix and prefix != provider) or (provider in _NEEDS_PREFIX and prefix != provider):
            continue
        if (deprecated := info.get("deprecation_date")) and deprecated <= today:
            continue
        names.append(name)
    return sorted(names)


def litellm_model(provider: str, model: str) -> str:
    """The name LiteLLM routes: price-map names as they are, others with a provider prefix."""
    model = model.strip()
    if "/" in model or (model in _cost_map() and provider not in _NEEDS_PREFIX):
        return model
    return f"{provider}/{model}"


def model_info(model: str) -> dict:
    info = _cost_map().get(model, {})
    return {"known": bool(info), "vision": bool(info.get("supports_vision")),
            "input_per_million": info.get("input_cost_per_token", 0) * 1e6 if info else None,
            "output_per_million": info.get("output_cost_per_token", 0) * 1e6 if info else None,
            "deprecation_date": info.get("deprecation_date")}


@dataclass(frozen=True)
class CostEstimate:
    model: str
    papers: int
    calls_per_paper: int
    input_tokens: int
    output_tokens: int
    usd: float | None
    per_paper_usd: float | None
    note: str


def estimate_cost(model: str, questions: list[Question], papers: int) -> CostEstimate:
    """Only questions the LLM grades cost anything: written answers, justifications of
    'choose and justify' questions, and diagrams (one vision call each). MCQs are free."""
    calls = input_tokens = output_tokens = 0
    for q in questions:
        if q.qtype in ("short", "descriptive") or (q.qtype == "mixed" and q.has_written_part):
            prompt = sum(len(m["content"]) for m in build_messages(q, q.model_answer or "")) // 4
            calls += 1
            input_tokens += prompt + FEW_SHOT_TOKENS
            output_tokens += OUTPUT_TOKENS
        if q.diagram is not None:
            calls += 1
            input_tokens += IMAGE_TOKENS + 200
            output_tokens += OUTPUT_TOKENS
    info = _cost_map().get(model)
    if not info or "input_cost_per_token" not in info:
        return CostEstimate(model, papers, calls, input_tokens, output_tokens, None, None,
                            "no price listed for this model; check the provider's pricing page")
    per_paper = input_tokens * info["input_cost_per_token"] + output_tokens * info.get("output_cost_per_token", 0)
    return CostEstimate(model, papers, calls, input_tokens, output_tokens, round(per_paper * papers, 4),
                        round(per_paper, 5), "estimate from LiteLLM's price list; repairs and retries add a little")


@dataclass(frozen=True)
class KeyCheck:
    ok: bool
    detail: str


class CloudProvider:
    def __init__(self, secrets: SecretStore | None = None, completion=None):
        self.secrets = secrets or SecretStore()
        self._completion = completion

    def completion(self, **kwargs):
        if self._completion is not None:
            return self._completion(**kwargs)
        import litellm

        return litellm.completion(**kwargs)

    def status(self) -> list[dict]:
        return [{"id": pid, "label": label, "has_key": self.secrets.get(pid) is not None,
                 "masked_key": self.secrets.masked(pid)} for pid, label in PROVIDERS.items()]

    def save_key(self, provider: str, api_key: str) -> None:
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}")
        self.secrets.set(provider, api_key)

    def delete_key(self, provider: str) -> None:
        self.secrets.delete(provider)

    def api_key(self, provider: str) -> str | None:
        return self.secrets.get(provider)

    def validate_api_key(self, provider: str, model: str, api_key: str | None = None) -> KeyCheck:
        """One tiny request ("Reply with OK", 5 tokens). Sends no student data."""
        key = api_key or self.secrets.get(provider)
        if not key:
            return KeyCheck(False, "no API key saved for this provider")
        import litellm

        try:
            self.completion(model=litellm_model(provider, model), api_key=key, max_tokens=5, timeout=30,
                            messages=[{"role": "user", "content": "Reply with OK."}])
        except litellm.AuthenticationError:
            return KeyCheck(False, "the provider rejected this API key")
        except litellm.NotFoundError:
            return KeyCheck(False, f"the key works but the model {model!r} was not found")
        except litellm.RateLimitError:
            return KeyCheck(True, "the key works, but the account is rate-limited or out of credit")
        except litellm.APIConnectionError:
            return KeyCheck(False, "couldn't reach the provider: check the internet connection")
        except Exception as e:
            return KeyCheck(False, f"{type(e).__name__}: {str(e)[:200]}")
        return KeyCheck(True, f"the key works with {model}")
