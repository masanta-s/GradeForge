"""The four learning mechanisms: status, TrOCR fine-tuning, and the LLM fine-tuning notebook."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from server.deps import get_services, job_response
from server.serialize import to_jsonable
from server.services import Services
from src import config

router = APIRouter(prefix="/api/learning", tags=["learning"])


class TrainIn(BaseModel):
    writer: str | None = None    # one student's handwriting, or everyone's corrections
    dataset: str | None = None   # also learn from an imported handwriting dataset
    dataset_lines: int = 2000    # how many of its lines to mix in


class DatasetIn(BaseModel):
    path: str
    name: str = "handwriting"
    limit: int = 5000


class ExportIn(BaseModel):
    consent: bool = False
    model: str | None = None   # the Ollama model to fine-tune; default: the one grading now


@router.get("/status")
def status(services: Services = Depends(get_services)) -> dict:
    from src.learning.llm_finetune_export import RECOMMENDED_CORRECTIONS
    from src.learning.ocr_fine_tuner import ACTIVE_FILE, MIN_SAMPLES

    grades = services.corrections.grade_corrections()
    pairs = sorted({(c.model, c.subject) for c in grades if c.ai_quality is not None})
    ocr = services.corrections.ocr_corrections()
    active_file = config.TROCR_FINETUNED_DIR / ACTIVE_FILE
    active = json.loads(active_file.read_text(encoding="utf-8")) if active_file.exists() else None
    trainable = sum(c.teacher_quality is not None for c in grades)
    reports = [services.calibrator.report(model, subject) for model, subject in pairs]
    return {
        "few_shot": {"active": bool(grades), "corrections": len(grades),
                     "changed": sum(c.kind == "changed" for c in grades),
                     "confirmed": sum(c.kind == "confirmed" for c in grades)},
        "calibration": [to_jsonable(r) | {"needed": r.needed} for r in reports],
        "ocr": {"corrections": len(ocr), "by_writer": dict(Counter(c.writer_id for c in ocr)),
                "min_samples": MIN_SAMPLES, "can_train": len(ocr) >= MIN_SAMPLES,
                "active_version": active["version"] if active else None,
                "active_cer": active["cer"] if active else None},
        "llm": {"examples": trainable, "recommended": RECOMMENDED_CORRECTIONS, "can_export": trainable > 0},
        "history": services.corrections.training_history(),
    }


class BenchmarkIn(BaseModel):
    setups: list[str] = ["alone", "learned"]


@router.get("/progress")
def progress(services: Services = Depends(get_services)) -> dict:
    """How closely the AI's marks match the teacher's, and the trend: day by day (approved sheets)
    and per model version (held-out benchmark)."""
    from src.learning.evaluation import DEFAULT_SAMPLE, holdout_items, weekly_agreement

    grades = services.corrections.grade_corrections()
    reviews = services.corrections.sheet_reviews()
    weekly = weekly_agreement(reviews)
    holdout_total = len(holdout_items(grades, limit=None))
    runs = services.corrections.benchmark_runs()
    return {
        "checked": {"total": len(grades), "changed": sum(c.kind == "changed" for c in grades),
                    "confirmed": sum(c.kind == "confirmed" for c in grades)},
        "approved_sheets": len(reviews),
        "weekly": weekly,
        "first_week": weekly[0] if weekly else None,
        "latest_week": weekly[-1] if weekly else None,
        "holdout": {"total": holdout_total, "measured_per_run": min(holdout_total, DEFAULT_SAMPLE)},
        "benchmarks": runs,
        "training": services.corrections.training_history("llm_lora"),
    }


@router.post("/benchmark")
def benchmark(body: BenchmarkIn, services: Services = Depends(get_services)) -> dict:
    """Re-mark the held-out answers with the current model: alone, and with what it learned."""
    from server.measure import SETUPS, run_benchmark
    from src.learning.evaluation import holdout_items

    if not set(body.setups) <= SETUPS.keys() or not body.setups:
        raise HTTPException(status_code=422, detail=f"setups must be among {sorted(SETUPS)}")
    if not holdout_items(services.corrections.grade_corrections()):
        raise HTTPException(status_code=409, detail="no held-out answers yet: correct or approve a few graded "
                                                    "sheets first (1 in 5 is kept aside for measuring)")

    def work(job):
        return run_benchmark(services, model_name=services.model_name, llm=services.llm, setups=body.setups,
                             progress=job.report)

    return job_response(services.jobs.submit("benchmark", work))


@router.get("/progress.csv")
def progress_csv(services: Services = Depends(get_services)) -> Response:
    """Everything behind the progress chart, for the teacher's own records or a report."""
    import csv
    import io

    from src.learning.evaluation import weekly_agreement

    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["kind", "when", "what", "answers", "exact %", "within half a mark %", "avg difference (marks)",
                     "avg difference (% of marks)", "bias (marks)", "marks kept as-is %"])
    for w in weekly_agreement(services.corrections.sheet_reviews()):
        writer.writerow(["week", w["week"], ", ".join(w["models"]), w["judged"], "", "", w["avg_change"], "", "",
                         w["accepted_pct"]])
    for r in services.corrections.benchmark_runs():
        writer.writerow(["held-out test", r["created_at"], r["label"], r["n"], round(100 * r["exact"], 1),
                         round(100 * r["close"], 1), round(r["mae"], 2), round(r["mae_pct"], 1), round(r["bias"], 2), ""])
    return Response(out.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="gradeforge_progress.csv"'})


@router.post("/trocr/train")
def train_trocr(body: TrainIn, services: Services = Depends(get_services)) -> dict:
    from src.learning.ocr_fine_tuner import MIN_SAMPLES

    corrections = services.corrections.ocr_corrections(body.writer)
    if not body.dataset and len(corrections) < MIN_SAMPLES:
        raise HTTPException(status_code=409, detail=f"need {MIN_SAMPLES} corrected lines, have {len(corrections)}")
    if body.dataset and not any(d.name == body.dataset for d in ocr_datasets_list()):
        raise HTTPException(status_code=404, detail=f"no imported dataset called {body.dataset!r}")

    def work(job):
        from src.learning.ocr_fine_tuner import train_trocr_lora
        from src.models.ollama_probe import OllamaModelProbe

        job.report(0.01, "Freeing the GPU")
        services.reload_ocr()  # drop the in-memory TrOCR
        try:
            if not services.cloud_active:
                OllamaModelProbe(timeout=10).unload(services.settings["model"])
        except Exception:
            pass  # Ollama not running: nothing to unload
        from src.learning.ocr_dataset import load_samples

        samples = []
        for c in corrections:
            image = cv2.imdecode(np.fromfile(c.image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                samples.append((image, c.corrected_text))

        heldout = None
        if body.dataset:
            # The teacher's own lines decide whether the result is better; the dataset only teaches.
            job.report(0.02, f"Loading {body.dataset}")
            extra = load_samples(body.dataset, body.dataset_lines)
            if len(samples) >= 2 * MIN_SAMPLES:
                keep = max(5, round(len(samples) * 0.25))
                heldout, samples = samples[:keep], samples[keep:]
            samples = samples + extra
        result = train_trocr_lora(samples, heldout=heldout, progress=job.report)
        note = f"writer: {body.writer or 'all'}; baseline {result.baseline}"
        if body.dataset:
            note += f"; + {body.dataset} dataset"
        services.corrections.log_training(
            mechanism="trocr_lora", model="trocr-base-handwritten", version=result.version, metric_name="cer",
            metric_before=result.baseline_cer, metric_after=result.tuned_cer, num_samples=len(samples),
            promoted=result.promoted, notes=note)
        if result.promoted:
            services.reload_ocr()  # next OCR run loads the new adapter
        return to_jsonable(result)

    return job_response(services.jobs.submit("train_trocr", work))


def ocr_datasets_list():
    from src.learning.ocr_dataset import datasets

    return datasets()


@router.get("/ocr/datasets")
def ocr_datasets() -> dict:
    """Handwriting datasets the teacher has imported (deletable once training is done)."""
    from src.learning.ocr_dataset import DATASETS_DIR

    found = ocr_datasets_list()
    return {"datasets": [d.to_dict() for d in found], "folder": str(DATASETS_DIR),
            "total_lines": sum(d.lines for d in found)}


@router.post("/ocr/datasets")
def import_ocr_dataset(body: DatasetIn, services: Services = Depends(get_services)) -> dict:
    """Convert a downloaded dataset (folder or .zip) into line images the reader can learn from."""
    _gpu_busy(services)

    def work(job):
        from pathlib import Path as FsPath

        from src.learning.ocr_dataset import prepare

        return prepare(FsPath(body.path), body.name, limit=body.limit, progress=job.report).to_dict()

    return job_response(services.jobs.submit("import_dataset", work))


@router.delete("/ocr/datasets/{name}")
def delete_ocr_dataset(name: str) -> dict:
    from src.learning.ocr_dataset import delete

    if not delete(name):
        raise HTTPException(status_code=404, detail=f"no imported dataset called {name!r}")
    return {"deleted": name}


class ModelIn(BaseModel):
    model: str | None = None      # default: the model grading now (a fine-tune means its base)


class RetrainIn(ModelIn):
    from_scratch: bool = False    # ignore the version in use and train from the original weights


class AdapterIn(ModelIn):
    path: str


def _base(services: Services, model: str | None) -> str:
    from server.retrain import base_of

    return base_of(model or services.settings["model"])


def _gpu_busy(services: Services) -> None:
    if services.jobs.busy("retrain_llm", "import_adapter", "train_trocr", "download_weights"):
        raise HTTPException(status_code=409, detail="another training or download job is running")


@router.get("/llm/weights")
def weights(model: str | None = None, check_size: bool = False, services: Services = Depends(get_services)) -> dict:
    """Are the original weights (needed to train or merge locally) on this computer?"""
    from src.models.weights_download import status

    base = _base(services, model)
    resolution = services.models.resolution(base)
    if resolution is None or not resolution.resolved:
        return {"model": base, "repo": None, "complete": False}
    http = None
    if check_size:
        import httpx

        http = httpx.Client(base_url="https://huggingface.co", timeout=20, follow_redirects=True)
    try:
        return {"model": base, **status(resolution.repo, http)}
    except Exception as e:  # the size lookup is optional (offline, rate limit)
        return {"model": base, **status(resolution.repo), "size_error": str(e)[:200]}


@router.post("/llm/weights/download")
def download_weights(body: ModelIn, services: Services = Depends(get_services)) -> dict:
    """The one large download after setup (~19 GB for Qwen3.5-9B), only when the teacher asks."""
    _gpu_busy(services)
    base = _base(services, body.model)

    def work(job):
        from src.models.model_manager import HF_TOKEN_NAME
        from src.models.weights_download import download

        resolution = services.models.resolution(base) or services.models.resolve(base)
        if not resolution.resolved:
            raise ValueError(f"no trainable HuggingFace source found for {base}")
        return {"path": str(download(resolution.repo, token=services.secrets.get(HF_TOKEN_NAME),
                                     progress=job.report))}

    return job_response(services.jobs.submit("download_weights", work))


@router.post("/llm/retrain")
def retrain_llm(body: RetrainIn, services: Services = Depends(get_services)) -> dict:
    """Fine-tune on this computer (layers streamed through the GPU), then keep the new version
    only if it agrees with the teacher better on held-out answers."""
    _gpu_busy(services)
    base = _base(services, body.model)

    def work(job):
        from server.retrain import retrain

        return retrain(services, job, base_model=base, from_scratch=body.from_scratch)

    return job_response(services.jobs.submit("retrain_llm", work))


@router.post("/llm/pause")
def pause_training(services: Services = Depends(get_services)) -> dict:
    """Stop a running fine-tune at its next step; starting it again resumes from the checkpoint."""
    services.pause_requested = True
    return {"pausing": services.jobs.busy("retrain_llm")}


@router.post("/llm/import-adapter")
def import_adapter(body: AdapterIn, services: Services = Depends(get_services)) -> dict:
    """An adapter trained by the exported notebook: merge, import, check, compare, maybe promote."""
    _gpu_busy(services)
    base = _base(services, body.model)

    def work(job):
        from server.retrain import import_adapter as run

        return run(services, job, base_model=base, adapter_dir=Path(body.path.strip().strip('"')))

    return job_response(services.jobs.submit("import_adapter", work))


class KaggleStartIn(ModelIn):
    consent: bool = False


@router.get("/llm/kaggle")
def kaggle_run(services: Services = Depends(get_services)) -> dict:
    """The Kaggle account status and the current run, if any (polled by the Learning page)."""
    from src.models.kaggle_auth import configured
    from src.models.kaggle_runner import finished, load_run, poll, tail_log

    run = load_run()
    result = {"configured": configured(), "run": run.to_dict() if run else None, "log": []}
    if run and not run.imported and result["configured"]:
        try:
            run = poll(run)
            from src.models.kaggle_runner import save_run

            save_run(run)
            result["run"] = run.to_dict()
            result["log"] = tail_log(run, config.CACHE_DIR / "kaggle" / "log")
            result["ready"] = finished(run) and "COMPLETE" in run.status
        except Exception as e:
            result["error"] = str(e)[:200]
    return result


@router.post("/llm/kaggle/start")
def kaggle_start(body: KaggleStartIn, services: Services = Depends(get_services)) -> dict:
    """Send the notebook to Kaggle and start it on their GPUs (two T4s)."""
    if not body.consent:
        raise HTTPException(status_code=403, detail="training on Kaggle sends students' answers there: "
                                                    "consent required")
    from src.models.kaggle_auth import configured

    if not configured():
        raise HTTPException(status_code=409,
                            detail="no Kaggle token: add KAGGLE_API_TOKEN to the .env file and restart")
    _gpu_busy(services)
    base = _base(services, body.model)

    def work(job):
        from datetime import datetime, timezone

        from server.retrain import training_data
        from src.learning.llm_finetune_export import build_notebook
        from src.models.kaggle_runner import push, save_run

        job.report(0.1, "Collecting your checked answers")
        identity = services.models.probe.probe(base)
        if services.models.resolution(base) is None:
            services.models.resolve(base)
        plan = services.models.training_plan(identity, services.grade_correction_count())
        option = next((o for o in plan.options if o.target == "kaggle" and o.available), None) if plan else None
        if option is None:
            raise ValueError(plan.blocked_reason if plan else "no training plan for this model")
        train, validation = training_data(services, since=None)
        notebook = build_notebook(train, validation, base_model=plan.repo, ollama_base=base,
                                  load_in_4bit=plan.estimate.method == "qlora",
                                  split=plan.estimate.vram_gb > 15,
                                  trained_until=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        job.report(0.6, "Sending the notebook to Kaggle")
        run = push(notebook, model=base, examples=len(train), folder=config.CACHE_DIR / "kaggle" / "push")
        save_run(run)
        job.report(1.0, f"Running on Kaggle: {run.url}")
        return run.to_dict()

    return job_response(services.jobs.submit("kaggle_start", work))


@router.post("/llm/kaggle/import")
def kaggle_import(services: Services = Depends(get_services)) -> dict:
    """Bring the finished run's adapter back: merge, import into Ollama, check, compare, promote."""
    from src.models.kaggle_runner import finished, load_run

    run = load_run()
    if run is None:
        raise HTTPException(status_code=409, detail="no Kaggle run to import")
    if not finished(run) or "COMPLETE" not in run.status:
        raise HTTPException(status_code=409, detail=f"the Kaggle run is {run.status.lower()}")
    _gpu_busy(services)

    def work(job):
        from server.retrain import import_adapter
        from src.models.kaggle_runner import fetch_adapter, save_run

        adapter = fetch_adapter(run, config.CACHE_DIR / "kaggle" / "output", progress=job.report)
        meta = import_adapter(services, job, base_model=run.model, adapter_dir=adapter)
        run.imported = True
        save_run(run)
        return meta

    return job_response(services.jobs.submit("import_adapter", work))


@router.get("/llm/versions")
def llm_versions(model: str | None = None, services: Services = Depends(get_services)) -> list[dict]:
    from server.retrain import versions
    from src.models.model_manager import tuned_name

    return versions(tuned_name(_base(services, model)))


@router.get("/llm/plan")
def llm_plan(model: str | None = None, services: Services = Depends(get_services)) -> dict:
    """Where the chosen model could be fine-tuned (cache-only). `needs_source`: its HF source
    hasn't been looked up yet; the page offers to look it up."""
    from src.models.capability_prober import assign_tier
    from src.models.ollama_probe import OllamaUnavailable

    from src.models.weights_download import is_complete

    name = _base(services, model)
    try:
        identity = services.models.probe.probe(name)
    except OllamaUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from None
    plan = services.models.training_plan(identity, services.grade_correction_count())
    report = services.models.report(name)
    return {"model": name, "needs_source": plan is None, "plan": plan.to_dict() if plan else None,
            "tier": assign_tier(report, plan).__dict__, "hardware": services.models.hardware.__dict__,
            "weights_ready": bool(plan and plan.repo and is_complete(plan.repo))}


@router.post("/llm/export")
def export_notebook(body: ExportIn, services: Services = Depends(get_services)) -> Response:
    if not body.consent:
        raise HTTPException(status_code=403, detail="exporting sends student answers to Colab/Kaggle: consent required")
    from datetime import datetime, timezone

    from server.retrain import training_data
    from src.learning.llm_finetune_export import build_notebook
    from src.models.hf_resolver import ResolverUnavailable
    from src.models.ollama_probe import OllamaUnavailable

    name = _base(services, body.model)
    try:
        identity = services.models.probe.probe(name)
        if services.models.resolution(name) is None:
            services.models.resolve(name)  # sends only the model's name to huggingface.co
        plan = services.models.training_plan(identity, services.grade_correction_count())
    except (OllamaUnavailable, ResolverUnavailable) as e:
        raise HTTPException(status_code=503, detail=str(e)) from None
    cloud = next((o for o in (plan.options if plan else ()) if o.target in ("colab", "kaggle") and o.available), None)
    if cloud is None:
        detail = plan.blocked_reason if plan and plan.blocked_reason else "no free cloud GPU can train this model"
        raise HTTPException(status_code=409, detail=detail)
    train, validation = training_data(services, since=None)   # the notebook always starts from the original
    try:
        notebook = build_notebook(train, validation, base_model=plan.repo, ollama_base=name,
                                  load_in_4bit=plan.estimate.method == "qlora",
                                  split=cloud.target == "kaggle" and plan.estimate.vram_gb > 15,
                                  trained_until=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from None
    return Response(json.dumps(notebook, indent=1, ensure_ascii=False), media_type="application/x-ipynb+json",
                    headers={"Content-Disposition": 'attachment; filename="gradeforge_finetune.ipynb"'})
