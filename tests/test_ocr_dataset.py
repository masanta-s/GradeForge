"""Importing an outside handwriting dataset, and training on it alongside the teacher's lines."""
import json
import zipfile

import cv2
import numpy as np
import pytest

from src.learning import ocr_dataset
from src.learning.ocr_dataset import UnknownLayout, datasets, delete, detect, load_samples, prepare
from tests.test_learning_api import api  # noqa: F401  (fixture)
from tests.test_server import client, wait  # noqa: F401  (fixture + helper)


def page(text_lines, path):
    image = np.full((120 * len(text_lines) + 40, 900, 3), 250, np.uint8)
    for i, text in enumerate(text_lines):
        cv2.putText(image, text, (30, 90 + i * 120), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (20, 20, 20), 3)
    cv2.imencode(".png", image)[1].tofile(path)


def gnhk_folder(root):
    """A page plus GNHK-style word boxes: line_idx groups words, %math% and printed text are skipped."""
    folder = root / "gnhk_dataset" / "train"
    folder.mkdir(parents=True)
    page(["photosynthesis happens in", "the chloroplast of plants"], folder / "eng_AA_001.png")
    words = []
    for line, sentence in enumerate(["photosynthesis happens in", "the chloroplast of plants"]):
        x = 30
        for word in sentence.split():
            width = 34 * len(word)
            words.append({"text": word, "line_idx": line, "type": "H",
                          "polygon": {"x0": x, "y0": 40 + line * 120, "x1": x + width, "y1": 40 + line * 120,
                                      "x2": x + width, "y2": 110 + line * 120, "x3": x, "y3": 110 + line * 120}})
            x += width + 12
    words.append({"text": "%math%", "line_idx": 2, "type": "H",
                  "polygon": {f"{a}{i}": 10 for i in range(4) for a in "xy"}})
    words.append({"text": "printed", "line_idx": 3, "type": "M",
                  "polygon": {f"{a}{i}": 20 for i in range(4) for a in "xy"}})
    (folder / "eng_AA_001.json").write_text(json.dumps(words), encoding="utf-8")
    return root


def test_gnhk_pages_become_line_images(tmp_path):
    source = gnhk_folder(tmp_path / "src")
    assert detect(source) == "GNHK pages"
    info = prepare(source, "GNHK Set", root=tmp_path / "out")
    assert (info.name, info.lines, info.layout) == ("gnhk-set", 2, "GNHK pages")
    samples = load_samples("gnhk-set", root=tmp_path / "out")
    assert [text for _, text in samples] == ["photosynthesis happens in", "the chloroplast of plants"]
    assert all(image.ndim == 2 and image.size > 1000 for image, _ in samples)   # real crops, not empty
    assert not (tmp_path / "out" / "gnhk-set" / "_unpacked").exists()           # the unpacked copy is cleaned up


def test_images_with_a_labels_file(tmp_path):
    source = tmp_path / "src"
    (source / "images").mkdir(parents=True)
    for i, text in enumerate(["first line here", "second line here"]):
        page([text], source / "images" / f"line{i}.png")
    (source / "gt.txt").write_text("images/line0.png\tfirst line here\nline1.png\tsecond line here\n#note\n",
                                   encoding="utf-8")
    assert detect(source) == "images with a labels file"
    info = prepare(source, "iam", root=tmp_path / "out")
    assert info.lines == 2
    assert sorted(t for _, t in load_samples("iam", root=tmp_path / "out")) == ["first line here", "second line here"]


def test_labels_csv_with_a_header(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    page(["hello there"], source / "a.png")
    (source / "labels.csv").write_text("filename,text\na.png,hello there\n", encoding="utf-8")
    info = prepare(source, "csvset", root=tmp_path / "out")
    assert info.lines == 1 and load_samples("csvset", root=tmp_path / "out")[0][1] == "hello there"


def test_parquet_tables(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = tmp_path / "src"
    source.mkdir()
    page(["a written line"], tmp_path / "one.png")
    raw = (tmp_path / "one.png").read_bytes()
    table = pa.table({"image": [{"bytes": raw, "path": "one.png"}], "text": ["a written line"]})
    pq.write_table(table, source / "train.parquet")
    assert detect(source) == "parquet tables"
    info = prepare(source, "iam-parquet", root=tmp_path / "out")
    assert info.lines == 1 and load_samples("iam-parquet", root=tmp_path / "out")[0][1] == "a written line"


def test_zip_input_and_unknown_layout(tmp_path):
    source = gnhk_folder(tmp_path / "zipped")
    archive = tmp_path / "gnhk.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for file in source.rglob("*"):
            if file.is_file():
                zipped.write(file, file.relative_to(source))
    assert prepare(archive, "fromzip", root=tmp_path / "out").lines == 2

    empty = tmp_path / "empty"
    (empty / "pictures").mkdir(parents=True)
    page(["no labels anywhere"], empty / "pictures" / "x.png")
    with pytest.raises(UnknownLayout, match="couldn't tell"):
        prepare(empty, "mystery", root=tmp_path / "out")
    with pytest.raises(FileNotFoundError):
        prepare(tmp_path / "nope", "missing", root=tmp_path / "out")


def test_list_limit_and_delete(tmp_path):
    prepare(gnhk_folder(tmp_path / "src"), "gnhk", root=tmp_path / "out")
    listed = datasets(root=tmp_path / "out")
    assert [d.name for d in listed] == ["gnhk"] and listed[0].bytes > 0
    assert len(load_samples("gnhk", limit=1, root=tmp_path / "out")) == 1
    assert delete("gnhk", root=tmp_path / "out") and not datasets(root=tmp_path / "out")
    assert not delete("gnhk", root=tmp_path / "out")        # deleting twice is not an error


# --- through the API -------------------------------------------------------------------------

def test_import_list_and_delete_through_the_api(api, tmp_path, monkeypatch):
    monkeypatch.setattr(ocr_dataset, "DATASETS_DIR", tmp_path / "handwriting")
    source = gnhk_folder(tmp_path / "src")
    assert api.get("/api/learning/ocr/datasets").json()["datasets"] == []

    info = wait(api, api.post("/api/learning/ocr/datasets",
                              json={"path": str(source), "name": "gnhk"}).json()["job_id"])
    assert info["lines"] == 2 and info["layout"] == "GNHK pages"
    listing = api.get("/api/learning/ocr/datasets").json()
    assert listing["total_lines"] == 2 and listing["datasets"][0]["name"] == "gnhk"
    assert api.delete("/api/learning/ocr/datasets/gnhk").json() == {"deleted": "gnhk"}
    assert api.delete("/api/learning/ocr/datasets/gnhk").status_code == 404


def test_training_mixes_the_dataset_but_judges_on_the_teachers_lines(api, tmp_path, monkeypatch):
    from src.learning.ocr_fine_tuner import OCRTrainingResult

    monkeypatch.setattr(ocr_dataset, "DATASETS_DIR", tmp_path / "handwriting")
    prepare(gnhk_folder(tmp_path / "src"), "gnhk", root=tmp_path / "handwriting")
    services = api.app.state.services
    image = tmp_path / "line.png"
    page(["a student line"], image)
    seen = {}

    def fake_train(samples, heldout=None, progress=None):
        seen["train"] = len(samples)
        seen["heldout"] = len(heldout or [])
        progress and progress(0.5, "training")
        return OCRTrainingResult("v1", "base", 0.2, 0.05, True, len(samples), len(heldout or []), 1.0, "dir")

    monkeypatch.setattr("src.learning.ocr_fine_tuner.train_trocr_lora", fake_train)

    # a dataset alone is enough to start, before 20 corrected lines exist
    assert api.post("/api/learning/trocr/train", json={}).status_code == 409
    assert api.post("/api/learning/trocr/train", json={"dataset": "nope"}).status_code == 404
    wait(api, api.post("/api/learning/trocr/train", json={"dataset": "gnhk"}).json()["job_id"])
    assert seen == {"train": 2, "heldout": 0}        # dataset only: its own split judges the result

    for i in range(40):
        services.corrections.add_ocr_correction(exam_id="e", sheet_id="s", writer_id="Riya", page=0, line_index=i,
                                                image_path=str(image), ocr_text="x", corrected_text=f"line {i}")
    wait(api, api.post("/api/learning/trocr/train", json={"dataset": "gnhk"}).json()["job_id"])
    assert seen["heldout"] == 10 and seen["train"] == 30 + 2   # the teacher's 10 lines decide; 30 + dataset train
    history = [h for h in api.get("/api/learning/status").json()["history"] if "gnhk" in h["notes"]]
    assert len(history) == 2 and history[-1]["promoted"] and "+ gnhk dataset" in history[-1]["notes"]
