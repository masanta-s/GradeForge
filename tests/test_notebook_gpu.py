"""The exported Colab/Kaggle notebook, run for real on a tiny Qwen3.5-architecture model.

Everything except the pip cell and the configuration values runs unchanged: loading, LoRA,
tokenisation with the chat template, the training loop with early stopping, and saving the
adapter, which GradeForge's merge then accepts."""
import json

import pytest
import torch

from src.learning.correction_store import GradeCorrection
from src.learning.llm_dataset import build_examples
from src.learning.llm_finetune_export import CONFIG_MARKER, build_notebook, config_source
from tests.tiny_qwen import make_tiny_qwen

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def corrections(n=40):
    answers = ["Plants use sunlight to make glucose", "Plants breathe at night", "Chlorophyll absorbs light energy",
               "It happens in the chloroplast and releases oxygen"]
    return [GradeCorrection(i, f"2026-09-0{1 + i % 9}T10:00:00+00:00", "e", f"sheet-{i}", "2", "Biology", "m",
                            "What is photosynthesis?", "Plants use sunlight water and carbon dioxide to make glucose.",
                            answers[i % 4], 4, 3, [3, 0.5, 2, 3.5][i % 4], 0.6, [0.8, 0.1, 0.5, 0.9][i % 4], 50,
                            "", "T") for i in range(n)]


def test_exported_notebook_trains_an_adapter_gradeforge_can_merge(tmp_path):
    from src.learning.lora_merge import merge_adapter

    data = corrections()
    train, validation = build_examples(data), build_examples(data, splits=("validation",))
    texts = [m["content"] for e in train + validation for m in e.messages] + [e.target for e in train]
    tiny = make_tiny_qwen(tmp_path / "tiny", texts)
    notebook = build_notebook(train, validation, base_model="Qwen/Qwen3.5-9B", ollama_base="qwen3.5:9b",
                              load_in_4bit=False, split=True, trained_until="2026-09-11T12:00:00+00:00")

    code = [("".join(c["source"])) for c in notebook["cells"] if c["cell_type"] == "code"]
    assert code[0].startswith("!pip install") and "bitsandbytes" not in code[0]
    assert code[1].startswith(CONFIG_MARKER) and "'Qwen/Qwen3.5-9B'" in code[1] and "SPLIT_ACROSS_GPUS = True" in code[1]
    for source in code:
        if not source.startswith("!"):
            compile(source, "<cell>", "exec")   # every cell is valid Python

    rows = lambda examples: [{"messages": e.messages, "target": e.target} for e in examples]  # noqa: E731
    local_config = config_source(base_model=str(tiny), load_in_4bit=False, split=True, trained_until="t",
                                 examples=rows(train), validation=rows(validation),
                                 settings={"learning_rate": 2e-2, "epochs": 3, "accumulate": 4, "patience": 3,
                                           "rank": 4, "alpha": 8, "max_length": 2048},
                                 output_dir=str(tmp_path / "gradeforge_adapter"))
    namespace: dict = {}
    for source in [local_config, *code[2:]]:
        exec(compile(source, "<notebook>", "exec"), namespace)

    assert namespace["history"][-1] < namespace["history"][0]          # it learned
    out = tmp_path / "gradeforge_adapter"
    meta = json.loads((out / "gradeforge.json").read_text())
    assert meta["examples"] == len(train) and meta["base_model"] == str(tiny)
    assert (tmp_path / "gradeforge_adapter.zip").exists()
    merged = merge_adapter(tiny, out, tmp_path / "merged")                 # GradeForge accepts the adapter
    assert (merged / "config.json").exists()
