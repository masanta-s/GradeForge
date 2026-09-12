"""Import an outside handwriting dataset so the handwriting reader can learn from other people's
writing as well as the teacher's own corrections.

The teacher downloads a dataset themselves (GradeForge never redistributes one) and points at the
folder or zip. Whatever the layout, it is converted to the only thing training needs: one image
per text line plus that line's text, in data/handwriting/<name>/. The originals can then be
deleted, and so can the imported copy once training is done: what is learned lives in the adapter.

Layouts understood:
  GNHK              page images with a matching .json of word boxes carrying `line_idx`
                    (kaggle.com/datasets/thejashwinima/gnhk-handwriting-dataset); words are
                    regrouped into lines, and GNHK's %math%/%SC%/%NA% markers are skipped
  parquet           HuggingFace-style tables with an image column and a text column
                    (e.g. the IAM mirror kaggle.com/datasets/dattrinh12/iam-handwriting-dataset)
  images + labels   a labels file (gt.txt, labels.txt, labels.csv, words.txt, *.csv) with
                    "<file><tab or comma><text>" per line, images anywhere beneath the folder
"""
from __future__ import annotations

import csv
import io
import json
import random
import shutil
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from src import config

DATASETS_DIR = config.DATA_DIR / "handwriting"
LABEL_FILES = ("gt.txt", "labels.txt", "labels.csv", "words.txt", "lines.txt", "train.csv", "labels.tsv")
IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
GNHK_MARKERS = ("%math%", "%SC%", "%NA%", "%%")
MIN_TEXT = 2
MAX_LINES = 20_000
Progress = Callable[[float, str], None]


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    lines: int
    layout: str
    source: str
    prepared_at: str
    bytes: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class UnknownLayout(ValueError):
    pass


# --- readers: each yields (grayscale line image, text) ---------------------------------------

def _read_gray(path: Path) -> np.ndarray | None:
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def _crop(image: np.ndarray, points: list[tuple[float, float]], pad: int = 6) -> np.ndarray | None:
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x0, x1 = int(max(0, min(xs) - pad)), int(min(image.shape[1], max(xs) + pad))
    y0, y1 = int(max(0, min(ys) - pad)), int(min(image.shape[0], max(ys) + pad))
    if x1 - x0 < 16 or y1 - y0 < 8:
        return None
    return image[y0:y1, x0:x1]


def read_gnhk(folder: Path) -> Iterator[tuple[np.ndarray, str]]:
    """Words carry a line number, so a page's words regroup into its lines."""
    for annotation in sorted(folder.rglob("*.json")):
        image_path = next((annotation.with_suffix(suffix) for suffix in IMAGE_TYPES
                           if annotation.with_suffix(suffix).exists()), None)
        if image_path is None:
            continue
        try:
            words = json.loads(annotation.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if not isinstance(words, list) or not words or "line_idx" not in words[0]:
            continue
        image = _read_gray(image_path)
        if image is None:
            continue
        lines: dict[int, list[dict]] = {}
        for word in words:
            if word.get("type", "H") == "H" and not any(m in str(word.get("text", "")) for m in GNHK_MARKERS):
                lines.setdefault(word.get("line_idx", 0), []).append(word)
        for _, group in sorted(lines.items()):
            points = [(p[f"x{i}"], p[f"y{i}"]) for w in group if (p := w.get("polygon")) for i in range(4)]
            ordered = sorted(group, key=lambda w: w["polygon"]["x0"])
            text = " ".join(str(w["text"]).strip() for w in ordered).strip()
            crop = _crop(image, points) if points else None
            if crop is not None and len(text) >= MIN_TEXT:
                yield crop, text


def read_parquet(folder: Path) -> Iterator[tuple[np.ndarray, str]]:
    """HuggingFace-style tables: an image column (bytes or a {'bytes': ...} struct) and a text column."""
    import pyarrow.parquet as pq

    for file in sorted(folder.rglob("*.parquet")):
        table = pq.read_table(file)
        names = {name.lower(): name for name in table.column_names}
        image_column = next((names[n] for n in ("image", "img", "pixel_values", "picture") if n in names), None)
        text_column = next((names[n] for n in ("text", "label", "sentence", "transcription", "words") if n in names),
                           None)
        if image_column is None or text_column is None:
            continue
        for image_value, text_value in zip(table[image_column].to_pylist(), table[text_column].to_pylist()):
            raw = image_value.get("bytes") if isinstance(image_value, dict) else image_value
            text = str(text_value or "").strip()
            if not isinstance(raw, (bytes, bytearray)) or len(text) < MIN_TEXT:
                continue
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                yield image, text


def _label_rows(path: Path) -> Iterator[tuple[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in (".csv", ".tsv"):
        rows = list(csv.reader(io.StringIO(text), delimiter="\t" if path.suffix.lower() == ".tsv" else ","))
        header = rows[0] if rows else []
        start = 1 if header and not any(c.lower().endswith(IMAGE_TYPES) for c in header) else 0
        for row in rows[start:]:
            if len(row) >= 2:
                yield row[0].strip(), row[1].strip()
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t") if "\t" in line else line.split(" ", 1)
        if len(parts) >= 2:
            yield parts[0].strip(), parts[-1].strip().replace("|", " ")


def read_labelled_images(folder: Path) -> Iterator[tuple[np.ndarray, str]]:
    label_file = next((f for name in LABEL_FILES for f in folder.rglob(name)), None)
    if label_file is None:
        return
    index = {f.name: f for f in folder.rglob("*") if f.suffix.lower() in IMAGE_TYPES}
    stems = {f.stem: f for f in index.values()}
    for reference, text in _label_rows(label_file):
        name = Path(reference.replace("\\", "/")).name
        path = index.get(name) or stems.get(Path(name).stem)
        if path is None or len(text) < MIN_TEXT:
            continue
        image = _read_gray(path)
        if image is not None:
            yield image, text


READERS = {"GNHK pages": read_gnhk, "parquet tables": read_parquet, "images with a labels file": read_labelled_images}


def detect(folder: Path) -> str:
    folder = Path(folder)
    if any(folder.rglob("*.parquet")):
        return "parquet tables"
    for annotation in folder.rglob("*.json"):
        try:
            first = json.loads(annotation.read_text(encoding="utf-8"))[0]
        except (ValueError, OSError, IndexError, KeyError):
            continue
        if isinstance(first, dict) and "line_idx" in first:
            return "GNHK pages"
        break
    if any(True for name in LABEL_FILES for _ in folder.rglob(name)):
        return "images with a labels file"
    raise UnknownLayout("couldn't tell what this dataset looks like: expected GNHK .json files, .parquet "
                        "tables, or images with a labels file (gt.txt / labels.csv)")


# --- import ----------------------------------------------------------------------------------

def _unpack(source: Path, work: Path) -> Path:
    if source.is_dir():
        return source
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            archive.extractall(work)
        return work
    raise UnknownLayout(f"{source} is neither a folder nor a .zip")


def prepare(source: Path, name: str, *, limit: int = 5000, root: Path | None = None,
            progress: Progress | None = None) -> DatasetInfo:
    """Convert a downloaded dataset into line images + text under data/handwriting/<name>."""
    from datetime import datetime, timezone

    report = progress or (lambda p, m: None)
    source = Path(str(source).strip().strip('"'))
    if not source.exists():
        raise FileNotFoundError(f"{source} does not exist")
    name = "".join(c for c in name.strip().lower().replace(" ", "-") if c.isalnum() or c in "-_") or "dataset"
    folder = Path(root or DATASETS_DIR) / name
    if folder.exists():
        shutil.rmtree(folder)
    (folder / "lines").mkdir(parents=True)

    work = folder / "_unpacked"
    unpacked = _unpack(source, work)
    layout = detect(unpacked)
    report(0.05, f"Reading {layout}")
    kept = 0
    limit = min(limit, MAX_LINES)
    with (folder / "labels.jsonl").open("w", encoding="utf-8") as labels:
        for image, text in READERS[layout](unpacked):
            file = f"{kept:06d}.png"
            cv2.imencode(".png", image)[1].tofile(folder / "lines" / file)
            labels.write(json.dumps({"file": file, "text": text}, ensure_ascii=False) + "\n")
            kept += 1
            if kept % 50 == 0:
                report(min(0.95, 0.05 + 0.9 * kept / limit), f"Prepared {kept} lines")
            if kept >= limit:
                break
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)   # the unpacked copy is not needed once lines exist
    if not kept:
        shutil.rmtree(folder, ignore_errors=True)
        raise UnknownLayout(f"no usable lines found in {source} ({layout})")

    info = DatasetInfo(name=name, lines=kept, layout=layout, source=str(source),
                       prepared_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       bytes=sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()))
    (folder / "meta.json").write_text(json.dumps(info.to_dict(), indent=2), encoding="utf-8")
    report(1.0, f"{kept} lines ready")
    return info


def datasets(root: Path | None = None) -> list[DatasetInfo]:
    folder = Path(root or DATASETS_DIR)
    found = []
    for meta in sorted(folder.glob("*/meta.json")):
        try:
            found.append(DatasetInfo(**json.loads(meta.read_text(encoding="utf-8"))))
        except (ValueError, TypeError):
            continue
    return found


def delete(name: str, root: Path | None = None) -> bool:
    folder = Path(root or DATASETS_DIR) / name
    if not (folder / "meta.json").exists():
        return False
    shutil.rmtree(folder, ignore_errors=True)
    return True


def load_samples(name: str, limit: int | None = None, *, root: Path | None = None,
                 seed: int = 0) -> list[tuple[np.ndarray, str]]:
    """A random sample of the dataset's lines, as (image, text) pairs for training."""
    folder = Path(root or DATASETS_DIR) / name
    rows = [json.loads(line) for line in (folder / "labels.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if limit and len(rows) > limit:
        rows = random.Random(seed).sample(rows, limit)
    samples = []
    for row in rows:
        image = _read_gray(folder / "lines" / row["file"])
        if image is not None:
            samples.append((image, row["text"]))
    return samples
