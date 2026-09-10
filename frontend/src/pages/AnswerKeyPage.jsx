import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Bot, CheckCheck, FileText, Lock, MessageSquareWarning, PencilLine, Save, ShieldCheck, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";
import { useOutletContext } from "react-router";
import { api } from "../api";
import { Badge, ConfidenceBadge, ErrorNote, FileDrop, JobBar } from "../components";
import { useJob } from "../hooks";
import DisputeDialog from "./DisputeDialog";

const QTYPE_LABEL = { mcq: "MCQ", mixed: "MCQ + justify", short: "Short answer", descriptive: "Long answer" };

export default function AnswerKeyPage() {
  const { exam } = useOutletContext();
  const queryClient = useQueryClient();
  const refresh = () => queryClient.invalidateQueries({ queryKey: ["exam", exam.id] });
  const job = useJob((j) => { if (j.status === "done") refresh(); });

  const [draft, setDraft] = useState(exam.answer_key);
  const [dirty, setDirty] = useState(false);
  const [disputeQid, setDisputeQid] = useState(null);
  useEffect(() => { if (!dirty) setDraft(exam.answer_key); }, [exam.answer_key, dirty]);

  const run = (fn) => async () => job.start((await fn()).job_id);
  const save = useMutation({
    mutationFn: () => api.saveKey(exam.id, draft),
    onSuccess: () => { setDirty(false); refresh(); },
  });
  const finalize = useMutation({ mutationFn: () => api.finalizeKey(exam.id), onSuccess: refresh });
  const blankKey = useMutation({
    mutationFn: () => api.saveKey(exam.id, {
      exam_name: exam.name, subject: exam.subject, finalized: false,
      questions: exam.questions.map((q) => ({
        id: q.id, text: q.text, qtype: q.qtype, max_marks: q.marks ?? 1, options: q.options,
        correct_option: null, model_answer: "", keywords: [], source: "teacher",
      })),
    }),
    onSuccess: refresh,
  });

  const updateQuestion = (qid, changes) => {
    setDraft((d) => ({ ...d, finalized: false, questions: d.questions.map((q) => (q.id === qid ? { ...q, ...changes } : q)) }));
    setDirty(true);
  };

  const validations = exam.validations ?? {};
  const openDisputes = Object.values(validations).filter((v) => v.status === "open");
  const finalizeError = finalize.error?.detail;

  // 1. no question paper yet
  if (!exam.questions.length) {
    return (
      <div className="max-w-2xl space-y-4">
        <p className="text-sm text-slate-600">
          Upload the question paper. Digital PDFs are read directly; scans and photos go through OCR.
        </p>
        <FileDrop accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff,.webp" disabled={job.running}
          hint="PDF, PNG or JPG" onFile={async (f) => job.start((await api.uploadPaper(exam.id, f)).job_id)} />
        <JobBar {...job} />
      </div>
    );
  }

  // 2. questions found, no key yet
  if (!draft) {
    return (
      <div className="space-y-5">
        <QuestionList questions={exam.questions} />
        <div className="card flex flex-wrap items-center gap-4 p-5">
          <div className="flex-1">
            <div className="font-medium text-slate-900">Prepare the answer key</div>
            <div className="text-sm text-slate-500">
              Let the local AI draft every answer for you to review, or write the key yourself.
            </div>
          </div>
          <button className="btn-secondary" disabled={job.running || blankKey.isPending} onClick={() => blankKey.mutate()}>
            <PencilLine size={16} /> Write it myself
          </button>
          <button className="btn-primary" disabled={job.running} onClick={run(() => api.generateKey(exam.id))}>
            <Sparkles size={16} /> Draft with AI
          </button>
        </div>
        <JobBar {...job} />
      </div>
    );
  }

  // 3. review / edit the key
  return (
    <div className="space-y-4">
      <div className="card flex flex-wrap items-center gap-3 px-4 py-3">
        <div className="flex-1 text-sm text-slate-600">
          {draft.questions.length} questions · {draft.questions.reduce((s, q) => s + Number(q.max_marks || 0), 0)} marks
          {openDisputes.length > 0 && <> · <b className="text-amber-700">{openDisputes.length} flagged by AI</b></>}
        </div>
        {dirty && (
          <button className="btn-primary" disabled={save.isPending} onClick={() => save.mutate()}>
            <Save size={16} /> Save changes
          </button>
        )}
        <button className="btn-secondary" disabled={dirty || job.running} onClick={run(() => api.validateKey(exam.id))}
          title={dirty ? "Save your changes first" : ""}>
          <ShieldCheck size={16} /> Check with AI
        </button>
        <button className="btn-primary" disabled={dirty || draft.finalized || finalize.isPending}
          onClick={() => finalize.mutate()}>
          {draft.finalized ? <><Lock size={16} /> Finalized</> : <><CheckCheck size={16} /> Finalize key</>}
        </button>
      </div>
      <JobBar {...job} />
      <ErrorNote>{save.error?.message}</ErrorNote>
      {finalizeError && (
        <ErrorNote>
          {finalizeError.open_disputes ? `Resolve the AI's flags first (questions ${finalizeError.open_disputes.join(", ")}).`
            : finalizeError.problems ? `Fix these first: ${finalizeError.problems.join("; ")}` : String(finalizeError)}
        </ErrorNote>
      )}
      {exam.key_problems?.length > 0 && !dirty && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
          Incomplete: {exam.key_problems.join("; ")}
        </div>
      )}

      {draft.questions.map((q) => (
        <KeyEntry key={q.id} question={q} validation={validations[q.id]}
          onChange={(changes) => updateQuestion(q.id, changes)} onDispute={() => setDisputeQid(q.id)} />
      ))}

      {disputeQid && (
        <DisputeDialog exam={exam} question={draft.questions.find((q) => q.id === disputeQid)}
          validation={validations[disputeQid]} onClose={() => { setDisputeQid(null); refresh(); }} />
      )}
    </div>
  );
}

function QuestionList({ questions }) {
  return (
    <div className="card divide-y divide-slate-100">
      <div className="flex items-center gap-2 px-5 py-3 text-sm font-medium text-slate-700">
        <FileText size={16} /> {questions.length} questions found in the paper
      </div>
      {questions.map((q) => (
        <div key={q.id} className="flex gap-3 px-5 py-3 text-sm">
          <span className="w-8 shrink-0 font-semibold text-slate-500">Q{q.id}</span>
          <div className="flex-1">
            <div className="text-slate-800">{q.text}</div>
            {Object.keys(q.options).length > 0 && (
              <div className="mt-1 text-slate-500">{Object.entries(q.options).map(([k, v]) => `${k}) ${v}`).join("   ")}</div>
            )}
          </div>
          <div className="flex shrink-0 items-start gap-1.5">
            <Badge>{QTYPE_LABEL[q.qtype]}</Badge>
            {q.wants_diagram && <Badge tone="blue">diagram</Badge>}
            <Badge>{q.marks ?? "?"} m</Badge>
          </div>
        </div>
      ))}
    </div>
  );
}

// Comma-separated list editor: keeps its own text while typing, syncs when the data changes.
function ListInput({ items, onCommit }) {
  const joined = (items ?? []).join(", ");
  const [text, setText] = useState(joined);
  useEffect(() => setText(joined), [joined]);
  return (
    <input className="input" value={text} onChange={(e) => setText(e.target.value)}
      onBlur={() => onCommit(text.split(",").map((s) => s.trim()).filter(Boolean))} />
  );
}

function KeyEntry({ question: q, validation, onChange, onDispute }) {
  const flagged = validation?.status === "open";
  const hasOptions = q.qtype === "mcq" || q.qtype === "mixed";

  return (
    <div className={`card p-5 ${flagged ? "ring-2 ring-amber-300" : ""}`}>
      <div className="mb-3 flex flex-wrap items-start gap-3">
        <span className="font-semibold text-slate-500">Q{q.id}</span>
        <div className="min-w-0 flex-1 text-slate-900">{q.text}</div>
        <Badge>{QTYPE_LABEL[q.qtype]}</Badge>
        {q.source === "ai_generated" && <Badge tone="blue"><Bot size={12} /> AI draft <ConfidenceBadge value={q.ai_confidence} /></Badge>}
        {q.source === "ai_suggestion_accepted" && <Badge tone="blue">AI suggestion accepted</Badge>}
        {q.source === "teacher_override" && <Badge tone="gray">Teacher's answer kept</Badge>}
        {validation?.verdict === "agrees" && <Badge tone="green"><ShieldCheck size={12} /> AI agrees</Badge>}
        {validation?.status === "resolved" && validation?.flagged && <Badge tone="gray">Dispute resolved</Badge>}
        <label className="flex items-center gap-1.5 text-sm text-slate-500">
          <input type="number" min={0} step={0.5} value={q.max_marks} className="input w-20 py-1"
            onChange={(e) => onChange({ max_marks: Number(e.target.value) })} /> marks
        </label>
      </div>

      {flagged && (
        <div className="mb-4 flex flex-wrap items-center gap-3 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm">
          <MessageSquareWarning size={17} className="text-amber-700" />
          <div className="min-w-0 flex-1 text-amber-900">
            <b>The AI thinks this may be wrong.</b> It suggests: <i>{validation.ai_answer}</i>
          </div>
          <button className="btn-secondary py-1.5" onClick={onDispute}>Review</button>
        </div>
      )}

      <div className="grid gap-4 md:grid-cols-2">
        {hasOptions && (
          <div>
            <span className="label">Correct option</span>
            <div className="space-y-1.5">
              {Object.entries(q.options).map(([k, v]) => (
                <label key={k} className="flex items-center gap-2 text-sm">
                  <input type="radio" name={`opt-${q.id}`} checked={q.correct_option === k}
                    onChange={() => onChange({ correct_option: k, source: "teacher" })} className="accent-brand-600" />
                  <b className="w-4">{k}</b> {v}
                </label>
              ))}
            </div>
            {q.explanation && <p className="mt-2 text-xs text-slate-500">{q.explanation}</p>}
          </div>
        )}
        {q.qtype !== "mcq" && (q.qtype === "mixed" || !q.diagram || q.diagram.marks < q.max_marks) && (
          <div className={hasOptions ? "" : "md:col-span-2"}>
            <label className="label">{q.qtype === "mixed" ? "Model justification" : "Model answer"}</label>
            <textarea className="input min-h-24" value={q.model_answer}
              onChange={(e) => onChange({ model_answer: e.target.value, source: "teacher" })} />
            <label className="label mt-3">Keywords (comma separated)</label>
            <ListInput items={q.keywords} onCommit={(keywords) => onChange({ keywords })} />
          </div>
        )}
        {q.diagram && (
          <div className="md:col-span-2">
            <label className="label">Diagram: required labels ({q.diagram.marks} marks)</label>
            <ListInput items={q.diagram.required_labels}
              onCommit={(required_labels) => onChange({ diagram: { ...q.diagram, required_labels } })} />
            {q.diagram.description && <p className="mt-1.5 text-xs text-slate-500">{q.diagram.description}</p>}
          </div>
        )}
      </div>
    </div>
  );
}
