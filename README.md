# PaperMind — AI Exam Paper Evaluator

Local-first exam evaluator: reads handwritten answer sheets (TrOCR), grades them with a
configurable strictness curve and a local LLM (Ollama), collaborates with the teacher on the
answer key, and learns from corrections through four mechanisms. **Student data never leaves
the machine.**

Full design: [`implementation_plan_updated3.md`](implementation_plan_updated3.md).

## Status

| Phase | State |
|---|---|
| 1. Setup & environment | ✅ |
| 2. OCR pipeline | 🟡 built + tested on synthetic pages; needs real answer sheets |
| 3. Grading + strictness | ✅ MCQ, subjective (hybrid LLM + embeddings), mixed choose-and-justify |
| 4. Diagrams | ✅ detection, label reading, vision-model judgement, teacher weightage |
| 5. Knowledge engine | ✅ question-paper parsing, AI answer keys, validation, disputes + audit log |
| 6. React + FastAPI UI | ✅ exams, answer-key review + disputes, grading, sheet review, analytics, settings |
| 7. Self-learning | ⏳ |
| 8. Polish | ⏳ |

## Requirements

| | Version | Notes |
|---|---|---|
| Windows | 11 | |
| Python | **3.14.0** | not 3.14.1 (excluded by torchvision 0.29) |
| NVIDIA driver | CUDA 13.2+ | no CUDA toolkit needed — PyTorch bundles the runtime |
| GPU | 8 GB VRAM | tested on RTX 4060 Laptop |
| Ollama | 0.34+ | |
| Node | 24 LTS | for the frontend (Phase 6) |

## Setup (PowerShell)

Everything — venv, pip cache, temp files, HuggingFace models — stays inside this folder.
`env.ps1` redirects all caches away from `C:`.

```bash
py -3.14 -m venv .venv
```

```bash
. .\env.ps1
```

```bash
python setup_env.py
```

`setup_env.py` checks Python and the driver, installs `requirements.txt` (PyTorch 2.14 +
CUDA 13.2 from the PyTorch index), pulls `qwen3.5:9b` if missing, **measures** each LLM's real
VRAM through Ollama's `/api/ps`, downloads TrOCR and MiniLM into `models/`, and smoke-tests
both on the GPU. After that, HuggingFace runs offline (`HF_HUB_OFFLINE=1` is set by
`src/config.py`; use `PAPERMIND_ALLOW_DOWNLOADS=1` to download again).

Flags: `--skip-install`, `--skip-ollama`, `--no-pull`, `--skip-models`.

## Models

| Role | Model | Where |
|---|---|---|
| Default LLM (grading, answer keys, validation) | `qwen3.5:9b` | Ollama |
| Fast / co-resident LLM, Colab fine-tune target | `gemma4:e4b` | Ollama |
| Handwriting OCR | `microsoft/trocr-base-handwritten` | `models/trocr_base/` |
| Answer similarity | `all-MiniLM-L6-v2` | `models/embedders/` |

Defaults only — any model Ollama serves can be selected.

## Environments

- **`.venv`** — the app, on the latest libraries (`requirements.txt`).
- **`.venv-train`** — optional, only for local LLM fine-tuning on a ≥10 GB GPU
  (`requirements-train.txt`, pinned by Unsloth). On an 8 GB GPU, LLM fine-tuning is exported
  to Colab/Kaggle instead.

## Layout

```
src/config.py              paths, model defaults, cache/offline environment
src/models/ollama_probe.py Layer 1: identify models + measure real VRAM via Ollama
src/ocr/preprocessing.py   illumination, deskew, ruled-line/page-edge removal, line segmentation
src/ocr/text_extractor.py  TrOCR line reading with per-line confidence
src/ocr/pipeline.py        image/PDF -> pages -> lines (digital PDFs use their text layer)
src/ocr/answer_segmenter.py  map lines to question numbers (Q3 / Ans 3 / 3(b) ...)
src/diagram/detector.py    find drawings (tall + sparse strokes), keep them out of text lines
src/diagram/label_extractor.py  read label words inside a drawing with TrOCR
src/diagram/evaluator.py   labels + vision-model structure/completeness (SSIM only as fallback)
src/diagram/weightage.py   teacher's diagram marks, required labels, weights
src/grading/answer_key.py  questions: mcq | short | descriptive | mixed (choose + justify), + optional diagram
src/grading/strictness_curve.py  0-100 slider -> marks curve, similarity rescaling
src/grading/mcq_grader.py  option detection ("(b)", "Option B", option text, 8->B ...)
src/grading/subjective_grader.py  LLM judges, embeddings + keywords cross-check
src/grading/structured_output.py  4-layer JSON defence (schema, tolerant parse, repair, regex)
src/grading/grading_engine.py  answer key + segmented sheet -> graded paper + review reasons
src/knowledge/llm_client.py  LiteLLM -> Ollama (think off, 8K ctx) / opt-in cloud
src/knowledge/question_parser.py  question paper -> questions, marks, options, sub-parts
src/knowledge/answer_generator.py  AI answer key per question type (flags low confidence)
src/knowledge/answer_validator.py  check teacher's key (MCQs solved blind, no anchoring)
src/knowledge/dispute_manager.py  accept / discuss / insist flow for flagged entries
src/knowledge/dispute_logger.py  SQLite audit trail, search, CSV export
demo/run_demo.py           end-to-end: sheet image -> OCR -> grading report
server/                    FastAPI: routes, background jobs, storage, what-if rescoring
frontend/                  React 19 + Vite 8 + Tailwind 4 app (8 screens)
setup_env.py               Phase 1 setup + verification
env.ps1                    shell environment (caches on this drive)
tests/                     pytest suite (synthetic answer sheets, GPU tests auto-skip)
models/                    downloaded weights (git-ignored)
data/                      SQLite DBs + student data (git-ignored)
```

## Run the app

Build the frontend once (uses the portable Node in `tools\node`, see below), then start the API,
which also serves the app:

```bash
cd frontend; npm install; npm run build; cd ..
```

```bash
python -m uvicorn server.main:app --port 8000
```

Open http://localhost:8000. For frontend development, run `npm run dev` in `frontend/`
(Vite on :5173, proxying `/api` to :8000).

**Node:** the frontend needs Node ≥ 22.22 (react-router 8). To keep everything off `C:`, use
the official portable zip unpacked to `tools\node`; `env.ps1` puts it first on `PATH`.

## Demo

```bash
python -m demo.run_demo
```

Renders a sample answer sheet (handwriting-style font, ruled, tilted, uneven light), runs OCR,
maps lines and drawings to questions, and grades MCQ, written, choose-and-justify and
labelled-diagram answers with `qwen3.5:9b`. Use `--sheet photo.jpg` for your own sheet, `--strictness 0-100`, or `--no-llm`.

## Tests

```bash
python -m pytest -q
```

`tests/test_privacy_boundary.py` fails the build if anything under `src/ocr/` or
`src/grading/` imports a network library.
