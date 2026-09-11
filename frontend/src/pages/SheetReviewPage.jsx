import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { AlertTriangle, ArrowLeft, Check, Image as ImageIcon, PenLine, RefreshCw, ScanText } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router";
import { api, assetUrl } from "../api";
import { Badge, ErrorNote, JobBar, Modal, StrictnessSlider } from "../components";
import { useJob, useSettings } from "../hooks";

// Approving a checked sheet confirms every AI mark the teacher left alone. Without it GradeForge
// would only ever see its mistakes, and couldn't tell how often it is right.
function ApproveBar({ result, teacher, approve, onApproved }) {
  const [name, setName] = useState(teacher);
  useEffect(() => setName(teacher), [teacher]);
  const mutation = useMutation({ mutationFn: () => approve(name.trim()), onSuccess: onApproved });
  const approved = result.approved_at;
  return (
    <div className={clsx("card flex flex-wrap items-center gap-3 p-4 text-sm", approved && "border-emerald-200 bg-emerald-50/50")}>
      <div className="min-w-60 flex-1 text-slate-600">
        {approved ? (
          <span className="flex items-center gap-1.5 text-emerald-800">
            <Check size={16} /> Approved by {result.approved_by} on {new Date(approved).toLocaleDateString()}.
            Changed a mark since? Approve again.
          </span>
        ) : (
          <>Checked every mark? <b className="text-slate-800">Approve</b> to confirm the marks you left unchanged;
            that is how GradeForge measures how often it agrees with you.</>
        )}
      </div>
      {!teacher && (
        <input className="input w-44 py-1.5" placeholder="Your name" aria-label="Your name" value={name}
          onChange={(e) => setName(e.target.value)} />
      )}
      <button className={approved ? "btn-secondary" : "btn-primary"} disabled={!name.trim() || mutation.isPending}
        onClick={() => mutation.mutate()}>
        <Check size={15} /> {approved ? "Approve again" : "Approve marks"}
      </button>
      {mutation.data && (
        <span className="w-full text-xs text-slate-500">
          {mutation.data.judged ? `You kept ${mutation.data.judged - mutation.data.changed} of ${mutation.data.judged} AI-judged marks.` : "No AI-judged answers on this sheet."}
        </span>
      )}
      <div className="w-full"><ErrorNote>{mutation.error?.message}</ErrorNote></div>
    </div>
  );
}

export default function SheetReviewPage() {
  const { examId, sheetId } = useParams();
  const queryClient = useQueryClient();
  const { data: settings } = useSettings();
  const { data: sheet, error } = useQuery({ queryKey: ["sheet", examId, sheetId], queryFn: () => api.sheet(examId, sheetId) });
  const refresh = () => queryClient.invalidateQueries({ queryKey: ["sheet", examId, sheetId] });
  const job = useJob((j) => { if (j.status === "done") { refresh(); setPreview(null); setTextChanged(false); } });

  const [strictness, setStrictness] = useState(50);
  const [preview, setPreview] = useState(null);
  const [textChanged, setTextChanged] = useState(false);
  const [editing, setEditing] = useState(null);
  const result = preview ?? sheet?.result;
  useEffect(() => { if (sheet?.result) setStrictness(sheet.result.strictness); }, [sheet?.result?.strictness]);

  const whatIf = async (s) => {
    if (!sheet?.result) return;
    setPreview(s === sheet.result.strictness ? null : await api.whatIf(examId, sheetId, s));
  };
  if (error) return <ErrorNote>{error.message}</ErrorNote>;
  if (!sheet) return null;

  return (
    <div className="mx-auto max-w-7xl">
      <Link to={`/exams/${examId}/students`} className="mb-3 inline-flex items-center gap-1 text-sm text-slate-500 hover:text-slate-700">
        <ArrowLeft size={15} /> Students
      </Link>
      <div className="mb-5 flex flex-wrap items-end gap-4">
        <div className="flex-1">
          <h1 className="text-2xl font-semibold text-slate-900">{sheet.student}</h1>
          {result && (
            <p className="mt-1 text-sm text-slate-500">
              Graded by {result.model} · {new Date(result.graded_at).toLocaleString()} ·{" "}
              <a className="text-brand-600 hover:underline" href={`/api/exams/${examId}/sheets/${sheetId}/report.pdf`}>
                Report card (PDF)
              </a>
            </p>
          )}
        </div>
        {result && (
          <div className="text-right">
            <div className={clsx("text-3xl font-semibold tabular-nums", preview ? "text-brand-600" : "text-slate-900")}>
              {result.total}<span className="text-lg text-slate-400">/{result.max_total}</span>
            </div>
            <div className="text-sm text-slate-500">{Math.round(result.percentage)}%{preview && " (preview)"}</div>
          </div>
        )}
      </div>

      <JobBar {...job} />
      {textChanged && (
        <div className="mb-4 flex items-center gap-3 rounded-lg border border-brand-200 bg-brand-50 px-4 py-3 text-sm text-brand-900">
          <PenLine size={16} /> You corrected the scanned text. Re-grade to use it.
          <button className="btn-primary ml-auto py-1.5" disabled={job.running}
            onClick={async () => job.start((await api.grade(examId, sheetId, strictness)).job_id)}>
            <RefreshCw size={14} /> Re-grade
          </button>
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_380px]">
        <div className="space-y-4">
          {result ? (
            <>
              <div className="card flex flex-wrap items-center gap-4 p-4">
                <div className="min-w-64 flex-1">
                  <StrictnessSlider compact value={strictness} onChange={setStrictness} onCommit={whatIf} />
                </div>
                {preview && (
                  <button className="btn-primary"
                    onClick={async () => { await api.whatIf(examId, sheetId, strictness, true); setPreview(null); refresh(); }}>
                    Apply strictness {strictness}
                  </button>
                )}
              </div>
              {result.questions.map((q) => (
                <QuestionResult key={q.question_id} q={q} onEdit={() => setEditing(q)} />
              ))}
              {result.unplaced_text && (
                <div className="card p-4 text-sm">
                  <div className="label">Text not matched to any question</div>
                  <pre className="whitespace-pre-wrap font-sans text-slate-700">{result.unplaced_text}</pre>
                </div>
              )}
              {!preview && (
                <ApproveBar result={result} teacher={settings?.teacher_name ?? ""}
                  approve={(name) => api.approveSheet(examId, sheetId, name)} onApproved={refresh} />
              )}
            </>
          ) : (
            <div className="card p-6 text-sm text-slate-600">
              Not graded yet. Check the scanned text on the right, then grade from the Students tab.
            </div>
          )}
        </div>
        <ScanPanel sheet={sheet} examId={examId} sheetId={sheetId} onChanged={() => { setTextChanged(true); refresh(); }} />
      </div>

      {editing && (
        <MarksDialog q={editing} teacher={settings?.teacher_name ?? ""} onClose={() => setEditing(null)}
          save={(body) => api.fixMarks(examId, sheetId, editing.question_id, body)}
          onSaved={() => { setEditing(null); setPreview(null); refresh(); }} />
      )}
    </div>
  );
}

function QuestionResult({ q, onEdit }) {
  const detail = q.detail ?? {};
  const written = q.qtype === "mixed" ? detail.justification : q.qtype !== "mcq" ? detail : null;
  const pct = q.max_marks ? q.marks / q.max_marks : 0;
  return (
    <div className={clsx("card p-5", q.review_reasons.length && "ring-2 ring-amber-200")}>
      <div className="flex flex-wrap items-start gap-3">
        <span className="font-semibold text-slate-500">Q{q.question_id}</span>
        <div className="min-w-0 flex-1">
          {q.status === "not_attempted" ? <span className="text-sm italic text-slate-400">Not attempted</span>
            : <pre className="whitespace-pre-wrap font-sans text-sm text-slate-800">{q.answer_text || "(drawing only)"}</pre>}
        </div>
        <button onClick={onEdit} title="Change marks"
          className={clsx("rounded-lg px-3 py-1.5 text-right tabular-nums hover:bg-slate-100",
            pct >= 0.8 ? "text-emerald-700" : pct >= 0.4 ? "text-amber-700" : "text-rose-700")}>
          <span className="text-xl font-semibold">{q.marks}</span>
          <span className="text-sm text-slate-400">/{q.max_marks}</span>
          {q.teacher_marks != null && <div className="text-[11px] text-slate-500">set by {q.corrected_by}</div>}
        </button>
      </div>

      {q.review_reasons.map((r) => (
        <div key={r} className="mt-3 flex items-start gap-2 rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-900">
          <AlertTriangle size={15} className="mt-0.5 shrink-0" /> {r}
        </div>
      ))}
      {q.feedback && <p className="mt-3 text-sm text-slate-600">{q.feedback}</p>}

      <div className="mt-3 flex flex-wrap gap-1.5">
        {q.qtype === "mcq" && detail.detected_option && (
          <Badge tone={detail.correct ? "green" : "red"}>chose {detail.detected_option}</Badge>)}
        {q.qtype === "mixed" && detail.choice?.detected_option && (
          <Badge tone={detail.choice.correct ? "green" : "red"}>chose {detail.choice.detected_option}</Badge>)}
        {written?.llm_quality != null && <Badge tone="blue">AI judged {Math.round(written.llm_quality * 100)}% correct</Badge>}
        {written?.matched_keywords?.map((k) => <Badge key={k} tone="green"><Check size={11} /> {k}</Badge>)}
        {written?.missing_keywords?.map((k) => <Badge key={k} tone="gray" className="line-through">{k}</Badge>)}
      </div>

      {q.diagram && (
        <div className="mt-4 rounded-lg border border-slate-200 p-3">
          <div className="mb-2 flex items-center gap-2 text-sm font-medium text-slate-700">
            <ImageIcon size={15} /> Diagram {q.diagram.marks}/{q.diagram.max_marks}
          </div>
          <div className="flex flex-wrap gap-1.5">
            {q.diagram.matched_labels.map((l) => <Badge key={l} tone="green"><Check size={11} /> {l}</Badge>)}
            {q.diagram.missing_labels.map((l) => <Badge key={l} tone="red">missing: {l}</Badge>)}
          </div>
          <div className="mt-2 grid grid-cols-3 gap-2 text-xs text-slate-500">
            {Object.entries(q.diagram.components).map(([k, v]) => (
              <div key={k}>
                <div className="capitalize">{k}</div>
                <div className="mt-1 h-1.5 rounded-full bg-slate-100">
                  <div className="h-full rounded-full bg-brand-500" style={{ width: `${v * 100}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ScanPanel({ sheet, examId, sheetId, onChanged }) {
  return (
    <div className="card h-fit lg:sticky lg:top-6">
      <div className="flex items-center gap-2 border-b border-slate-100 px-4 py-3 text-sm font-medium text-slate-700">
        <ScanText size={16} /> Scanned answer sheet
        <a className="ml-auto text-xs font-normal text-brand-600 hover:underline" target="_blank" rel="noreferrer"
          href={`/api/exams/${examId}/sheets/${sheetId}/source`}>original</a>
      </div>
      <div className="max-h-[75vh] space-y-3 overflow-y-auto p-4">
        {sheet.pages.map((page) => (
          <div key={page.page_index} className="space-y-3">
            {page.lines.map((line, i) => (
              <LineEditor key={i} line={line} src={line.crop && assetUrl(examId, sheetId, "lines", line.crop)}
                save={(text) => api.fixLine(examId, sheetId, page.page_index, i, text)} onSaved={onChanged} />
            ))}
            {page.diagrams.map((d) => (
              <div key={d.stem}>
                <img alt="drawing" className="w-full rounded border border-slate-200"
                  src={assetUrl(examId, sheetId, "diagrams", `${d.stem}_image.png`)} />
                <div className="mt-1 text-xs text-slate-500">Drawing · detection {Math.round(d.confidence * 100)}%</div>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

function LineEditor({ line, src, save, onSaved }) {
  const [text, setText] = useState(line.text);
  useEffect(() => setText(line.text), [line.text]);
  const mutation = useMutation({ mutationFn: () => save(text), onSuccess: onSaved });
  const low = line.confidence < 0.6;
  return (
    <div className={clsx("rounded-lg border p-2", low ? "border-amber-300 bg-amber-50/50" : "border-slate-200")}>
      {src && <img alt="" src={src} className="mb-1.5 max-h-14 w-full rounded object-contain object-left" />}
      <div className="flex items-center gap-1.5">
        <input className="input py-1 text-sm" value={text} onChange={(e) => setText(e.target.value)}
          onBlur={() => text !== line.text && mutation.mutate()} />
        {line.corrected ? <Badge tone="blue">fixed</Badge> : low ? <Badge tone="amber">check</Badge> : null}
      </div>
    </div>
  );
}

function MarksDialog({ q, teacher: initialTeacher, save, onSaved, onClose }) {
  const [marks, setMarks] = useState(q.marks);
  const [note, setNote] = useState(q.teacher_note ?? "");
  const [teacher, setTeacher] = useState(initialTeacher);
  const mutation = useMutation({ mutationFn: () => save({ marks: Number(marks), note, teacher_name: teacher }), onSuccess: onSaved });
  return (
    <Modal title={`Change marks for Q${q.question_id}`} onClose={onClose}>
      <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); mutation.mutate(); }}>
        <p className="text-sm text-slate-600">
          AI gave {q.ai_marks ?? q.marks}/{q.max_marks}. Your mark is kept as teacher intent and helps future grading.
        </p>
        <div>
          <label className="label" htmlFor="marks">Marks (out of {q.max_marks})</label>
          <input id="marks" type="number" step={0.5} min={0} max={q.max_marks} className="input" value={marks}
            onChange={(e) => setMarks(e.target.value)} autoFocus />
        </div>
        <div>
          <label className="label" htmlFor="note">Why (optional)</label>
          <input id="note" className="input" value={note} onChange={(e) => setNote(e.target.value)}
            placeholder="e.g. accepted: CO2 written as carbon dioxide" />
        </div>
        <div>
          <label className="label" htmlFor="by">Your name</label>
          <input id="by" className="input" value={teacher} onChange={(e) => setTeacher(e.target.value)} />
        </div>
        <ErrorNote>{mutation.error?.message}</ErrorNote>
        <div className="flex justify-end gap-2">
          <button type="button" className="btn-secondary" onClick={onClose}>Cancel</button>
          <button className="btn-primary" disabled={!teacher.trim() || mutation.isPending}>Save marks</button>
        </div>
      </form>
    </Modal>
  );
}
