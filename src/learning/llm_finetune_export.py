"""Mechanism 4: export a Colab/Kaggle notebook that fine-tunes the grading LLM on corrections.

On an 8 GB GPU no current model can be fine-tuned locally (Gemma 4 E4B QLoRA needs ~10 GB;
Qwen 3.5 advises against 4-bit training and 16-bit LoRA needs ~22 GB), so training runs on a
free Colab/Kaggle T4 (15 GB) and only the finished GGUF comes back to Ollama.

Privacy: the notebook carries student answers off this machine, so the API requires explicit
consent. Only (prompt, teacher score) pairs are included: no student names, sheet ids or exams.
Each prompt is built exactly as the grader builds it, so the model is trained on what it sees.

This produces a starting template: it can't be executed from here, so the base-model repo id
and Unsloth calls must be checked on first run (noted at the top of the notebook).
"""
from __future__ import annotations

import json

from src.grading.answer_key import Question
from src.grading.subjective_grader import build_messages
from src.learning.correction_store import GradeCorrection

RECOMMENDED_CORRECTIONS = 200
BASE_MODEL = "google/gemma-4-E4B-it"  # verify on huggingface.co before running


def training_examples(corrections: list[GradeCorrection]) -> list[dict]:
    """Chat-format examples: the grader's own prompt -> the teacher's judgement as JSON."""
    examples = []
    for c in corrections:
        if c.teacher_quality is None:
            continue  # only corrections with a comparable quality score (written answers)
        question = Question(id="q", text=c.question_text, qtype="short", max_marks=c.max_marks,
                            model_answer=c.model_answer)
        messages = build_messages(question, c.student_answer)
        reply = {"quality": round(c.teacher_quality, 3),
                 "feedback": c.note or "Graded by the teacher.", "missing_points": []}
        examples.append({"messages": messages + [{"role": "assistant", "content": json.dumps(reply)}]})
    return examples


def _cell(kind: str, source: str) -> dict:
    cell = {"cell_type": kind, "metadata": {}, "source": source.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        cell.update(outputs=[], execution_count=None)
    return cell


def build_notebook(corrections: list[GradeCorrection], *, base_model: str = BASE_MODEL,
                   output_name: str = "gradeforge-grader") -> dict:
    examples = training_examples(corrections)
    if not examples:
        raise ValueError("no written-answer corrections to train on yet")
    warning = ("" if len(examples) >= RECOMMENDED_CORRECTIONS else
               f"\n\n> **Only {len(examples)} examples.** Fine-tuning is recommended from "
               f"{RECOMMENDED_CORRECTIONS}; until then, few-shot examples and calibration do the learning.")
    data = json.dumps(examples, ensure_ascii=False)
    cells = [
        _cell("markdown", f"""
# GradeForge: fine-tune the grading model on your corrections

Runs on a free **Colab or Kaggle T4 GPU**. It trains LoRA adapters on `{base_model}` with
Unsloth, exports a Q4 GGUF, and you import that into Ollama on the school computer.

* **{len(examples)} training examples**, each: the grader's prompt → the teacher's score.
  No student names, sheet ids or exam names are included.
* **Check before running:** the base-model repo id and the Unsloth API (this notebook was
  generated, not run). Use the same chat template at inference as in training.{warning}
"""),
        _cell("code", "!pip install -q unsloth"),
        _cell("code", f"""
import json
from unsloth import FastLanguageModel

BASE_MODEL = "{base_model}"
model, tokenizer = FastLanguageModel.from_pretrained(BASE_MODEL, max_seq_length=2048, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
"""),
        _cell("code", f"""
from datasets import Dataset

EXAMPLES = json.loads({data!r})
dataset = Dataset.from_list([
    {{"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}} for ex in EXAMPLES
])
print(len(dataset), "examples")
"""),
        _cell("code", """
from trl import SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text", per_device_train_batch_size=2, gradient_accumulation_steps=4,
        num_train_epochs=3, learning_rate=2e-4, warmup_ratio=0.05, logging_steps=5,
        fp16=True,  # T4 GPUs have no native bf16
        output_dir="outputs", report_to="none",
    ),
)
trainer.train()
"""),
        _cell("code", f"""
model.save_pretrained_gguf("{output_name}", tokenizer, quantization_method="q4_k_m")
import glob
gguf = glob.glob("{output_name}/*.gguf")[0]
print("Download:", gguf)
try:
    from google.colab import files
    files.download(gguf)
except ImportError:
    pass  # Kaggle: download it from the output panel
"""),
        _cell("markdown", f"""
## Import into Ollama (on the school computer)

Put the `.gguf` next to a file called `Modelfile` containing `FROM ./<file>.gguf`, then run
`ollama create {output_name} -f Modelfile`. In GradeForge, choose **{output_name}** under
*Models & settings*, and re-grade a few already-corrected sheets to compare it with the base model.
"""),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                         "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
