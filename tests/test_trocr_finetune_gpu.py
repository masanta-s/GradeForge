"""Integration: real TrOCR LoRA fine-tuning on the simulated difficult writer (GPU, ~30 s).

Measured when built (48 lines, 12 epochs): held-out CER 23.1 % -> 1.5 %, and on sentences made
of words never seen in training 11.7 % -> 0.8 %, so it learns the handwriting, not the words.
"""
import pytest
import torch

from src import config
from tests.learning_data import messy_line, sentences

pytestmark = [
    pytest.mark.skipif(not (config.TROCR_BASE_DIR / "model.safetensors").exists(), reason="run setup_env.py"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]


def test_finetune_improves_and_is_promoted(tmp_path):
    from src.learning.ocr_fine_tuner import active_adapter, train_trocr_lora

    texts = sentences(32, seed=11)
    samples = [(messy_line(t, seed=100 + i, strength=1.2), t) for i, t in enumerate(texts)]
    result = train_trocr_lora(samples, root=tmp_path, epochs=8)
    print(f"\nCER {result.baseline_cer:.1%} -> {result.tuned_cer:.1%} in {result.seconds}s")
    assert result.baseline_cer > 0.10          # base model really struggles with this writer
    assert result.tuned_cer < result.baseline_cer / 2
    assert result.promoted and active_adapter(tmp_path).name == result.version


def test_worse_checkpoint_is_not_promoted_and_current_stays(tmp_path, monkeypatch):
    """The gate, end to end: a checkpoint that measures worse must not replace the one in use.
    (Shuffled labels can't be relied on to make CER worse: the held-out labels are shuffled
    too, and training on the shared vocabulary can still lower that CER.)"""
    import json

    import src.learning.ocr_fine_tuner as tuner

    (tmp_path / "v1").mkdir()
    (tmp_path / tuner.ACTIVE_FILE).write_text(json.dumps({"version": "v1", "cer": 0.10}))
    measured = iter([0.10, 0.25])  # current model, then the new checkpoint
    monkeypatch.setattr(tuner, "_cer", lambda extractor, samples: next(measured))
    monkeypatch.setattr(tuner, "active_adapter", lambda root=tmp_path: None)  # baseline loads base weights

    texts = sentences(20, seed=3)
    result = tuner.train_trocr_lora([(messy_line(t, seed=i), t) for i, t in enumerate(texts)],
                                    root=tmp_path, epochs=1)
    assert (result.baseline_cer, result.tuned_cer, result.promoted) == (0.10, 0.25, False)
    assert json.loads((tmp_path / tuner.ACTIVE_FILE).read_text())["version"] == "v1"
