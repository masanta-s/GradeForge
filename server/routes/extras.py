"""Results exports (CSV, PDF report cards) and the one-click demo exam."""
from __future__ import annotations

import csv
import html
import io

from fastapi import APIRouter, Depends
from fastapi.responses import Response

from server.deps import get_services, job_response, not_found
from server.routes.exams import _question_json
from server.services import Services

router = APIRouter(prefix="/api", tags=["exports"])


@router.get("/exams/{exam_id}/results.csv")
def results_csv(exam_id: str, services: Services = Depends(get_services)) -> Response:
    with not_found("exam"):
        exam = services.storage.get_exam(exam_id)
        key = services.storage.get_key(exam_id)
        sheets = services.storage.list_sheets(exam_id)
    ids = key.question_ids if key else []
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["Student", *[f"Q{q}" for q in ids], "Total", "Out of", "Percent", "To check", "Graded at"])
    for sheet in sheets:
        result = services.storage.get_result(exam_id, sheet["id"])
        if result is None:
            continue
        marks = {q["question_id"]: q["marks"] for q in result["questions"]}
        writer.writerow([sheet["student"], *[f"{marks.get(q, 0):g}" for q in ids], f"{result['total']:g}",
                         f"{result['max_total']:g}", f"{result['percentage']:.1f}",
                         sum(bool(q["review_reasons"]) for q in result["questions"]), result.get("graded_at", "")])
    filename = f"{exam['name']} results.csv".replace('"', "")
    return Response("\ufeff" + out.getvalue(), media_type="text/csv",  # BOM: Excel reads UTF-8 correctly
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


_REPORT_CSS = """
* { font-family: sans-serif; }
body { font-size: 10pt; color: #1e293b; }
h1 { font-size: 17pt; margin: 0 0 2pt 0; }
.muted { color: #64748b; font-size: 9pt; }
.total { font-size: 15pt; font-weight: bold; margin: 10pt 0; }
table { width: 100%; border-collapse: collapse; margin-top: 6pt; }
th { text-align: left; background: #f1f5f9; font-size: 8.5pt; padding: 4pt; }
td { border-bottom: 1px solid #e2e8f0; padding: 4pt; vertical-align: top; }
.marks { white-space: nowrap; font-weight: bold; }
.note { color: #b45309; font-size: 8.5pt; }
"""


def _report_html(exam: dict, result: dict) -> str:
    rows = []
    for q in result["questions"]:
        teacher = " (teacher)" if q.get("teacher_marks") is not None else ""
        feedback = html.escape(q.get("feedback") or ("Not attempted." if q["status"] == "not_attempted" else ""))
        note = f'<div class="note">Teacher: {html.escape(q["teacher_note"])}</div>' if q.get("teacher_note") else ""
        rows.append(f'<tr><td>Q{html.escape(q["question_id"])}</td>'
                    f'<td class="marks">{q["marks"]:g} / {q["max_marks"]:g}{teacher}</td>'
                    f'<td>{feedback}{note}</td></tr>')
    return f"""
<h1>{html.escape(result.get("student", ""))}</h1>
<div class="muted">{html.escape(exam["name"])} &middot; {html.escape(exam["subject"])}</div>
<div class="total">{result["total"]:g} / {result["max_total"]:g} &nbsp; ({result["percentage"]:.0f}%)</div>
<table><tr><th>Question</th><th>Marks</th><th>Feedback</th></tr>{''.join(rows)}</table>
<p class="muted">Marked with GradeForge on this school's own computer and reviewed by the teacher.
Marks labelled (teacher) were set by the teacher.</p>
"""


@router.get("/exams/{exam_id}/sheets/{sheet_id}/report.pdf")
def report_pdf(exam_id: str, sheet_id: str, services: Services = Depends(get_services)) -> Response:
    import pymupdf

    with not_found("result"):
        exam = services.storage.get_exam(exam_id)
        result = services.storage.get_result(exam_id, sheet_id)
        if result is None:
            raise KeyError("result")
    buffer = io.BytesIO()
    story = pymupdf.Story(html=_report_html(exam, result), user_css=_REPORT_CSS)
    writer = pymupdf.DocumentWriter(buffer)
    page = pymupdf.paper_rect("a4")
    more = True
    while more:
        device = writer.begin_page(page)
        more, _ = story.place(page + (40, 40, -40, -40))
        story.draw(device)
        writer.end_page()
    writer.close()
    filename = f"{result.get('student', 'report')} - {exam['name']}.pdf".replace('"', "")
    return Response(buffer.getvalue(), media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.post("/demo")
def create_demo(services: Services = Depends(get_services)) -> dict:
    """A ready-to-grade exam: question paper, the teacher's finalized key and three students."""
    from demo.samples import SAMPLE_KEY, STUDENTS, render_paper_pdf, render_sheet
    from src import config
    from src.knowledge.question_parser import parse_question_paper
    from src.ocr.pipeline import OCRPipeline

    exam = services.storage.create_exam("Demo: " + SAMPLE_KEY.exam_name.replace(" (demo)", ""), SAMPLE_KEY.subject)
    scratch = config.CACHE_DIR / "demo" / exam["id"]
    paper = render_paper_pdf(scratch / "paper.pdf")
    path = services.storage.save_paper(exam["id"], "paper.pdf", paper.read_bytes())
    exam.update(paper_file=path.name, questions=[_question_json(q) for q in
                                                 parse_question_paper(OCRPipeline(detect_diagrams=False).run_file(path))])
    services.storage.update_exam(exam)
    services.storage.save_key(exam["id"], SAMPLE_KEY)

    jobs = []
    for skew, (student, (lines, labels)) in zip((2.5, -1.8, 1.2), STUDENTS.items()):
        image = render_sheet(scratch / f"{student}.png", lines, labels, skew=skew)
        sheet_id, sheet_path = services.storage.create_sheet(exam["id"], student, "sheet.png", image.read_bytes())

        def read(job, sheet_id=sheet_id, sheet_path=sheet_path):
            services.storage.save_ocr(exam["id"], sheet_id, services.ocr.run_file(sheet_path))
            return {"sheet_id": sheet_id}

        jobs.append(job_response(services.jobs.submit("read_sheet", read)))
    return {"exam_id": exam["id"], "jobs": jobs}
