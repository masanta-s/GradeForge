import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { BookMarked, CloudUpload, Gauge, PenTool, Sparkles } from "lucide-react";
import { useState } from "react";
import { api } from "../api";
import { Badge, ErrorNote, JobBar, PageHeader } from "../components";
import { useJob } from "../hooks";

export default function LearningPage() {
  const queryClient = useQueryClient();
  const { data: status } = useQuery({ queryKey: ["learning"], queryFn: api.learning });
  const job = useJob((j) => { if (j.status === "done") queryClient.invalidateQueries({ queryKey: ["learning"] }); });
  const [writer, setWriter] = useState("");
  const [consent, setConsent] = useState(false);
  const [trainError, setTrainError] = useState(null);
  const exportNotebook = useMutation({ mutationFn: api.exportNotebook });
  if (!status) return null;

  const ocrCount = writer ? status.ocr.by_writer[writer] ?? 0 : status.ocr.corrections;
  const train = async () => {
    setTrainError(null);
    try { job.start((await api.trainTrocr(writer || null)).job_id); } catch (e) { setTrainError(e.message); }
  };

  return (
    <div className="mx-auto max-w-5xl">
      <PageHeader title="Learning"
        subtitle="GradeForge learns from your corrections. They are kept as your intent, so they work with every model." />

      <div className="grid gap-5 md:grid-cols-2">
        <Mechanism n={1} icon={BookMarked} title="Examples from your past marking"
          active={status.few_shot.active}
          status={status.few_shot.active ? `${plural(status.few_shot.corrections, "correction")} used` : "Correct a mark to start"}>
          When grading, the AI is shown how you marked similar answers before. Works with every model,
          from your first correction.
        </Mechanism>

        <Mechanism n={2} icon={Gauge} title="Score calibration"
          active={status.calibration.some((c) => c.active)}
          status={status.calibration.length ? null : "Needs 15 corrected written answers"}>
          Learns how much each model over- or under-marks compared with you, per subject, and corrects for it.
          {status.calibration.map((c) => (
            <div key={`${c.model}-${c.subject}`} className="mt-3">
              <div className="flex justify-between text-xs text-slate-600">
                <span>{c.model} · {c.subject}</span>
                <span>{c.active ? `model ${c.mean_bias > 0 ? "+" : ""}${Math.round(c.mean_bias * 100)}% vs you`
                  : `${c.needed} more to activate`}</span>
              </div>
              <Progress value={c.corrections / 15} />
            </div>
          ))}
        </Mechanism>

        <Mechanism n={3} icon={PenTool} title="Reading students' handwriting"
          active={!!status.ocr.active_version}
          status={status.ocr.active_version
            ? `Using ${status.ocr.active_version} · ${(status.ocr.active_cer * 100).toFixed(1)}% character errors`
            : `${plural(status.ocr.corrections, "corrected line")}`}>
          Fine-tunes the handwriting reader on lines you corrected. A new version is used only if it reads
          held-out lines more accurately than the current one.
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <select className="input w-auto py-1.5" value={writer} onChange={(e) => setWriter(e.target.value)}>
              <option value="">All students</option>
              {Object.entries(status.ocr.by_writer).map(([w, n]) => <option key={w} value={w}>{w} ({n})</option>)}
            </select>
            <button className="btn-primary py-1.5" onClick={train}
              disabled={job.running || ocrCount < status.ocr.min_samples}>
              Train on this handwriting
            </button>
          </div>
          {ocrCount < status.ocr.min_samples && (
            <p className="mt-2 text-xs text-slate-500">
              {plural(status.ocr.min_samples - ocrCount, "more corrected line")} needed.
            </p>
          )}
          <div className="mt-3"><JobBar {...job} /></div>
          <ErrorNote>{trainError}</ErrorNote>
        </Mechanism>

        <Mechanism n={4} icon={Sparkles} title="Fine-tuning the grading model"
          active={false} status={`${status.llm.examples} of ${status.llm.recommended} recommended examples`}>
          This computer's GPU is too small to fine-tune current models, so GradeForge prepares a notebook that
          trains on a free Colab or Kaggle GPU. You then import the result into Ollama.
          <Progress value={status.llm.examples / status.llm.recommended} />
          <label className="mt-3 flex items-start gap-2 text-xs text-slate-700">
            <input type="checkbox" className="mt-0.5 accent-brand-600" checked={consent}
              onChange={(e) => setConsent(e.target.checked)} />
            I understand the notebook contains students' answers (no names) and will leave this computer.
          </label>
          <button className="btn-secondary mt-3 py-1.5" disabled={!consent || !status.llm.can_export || exportNotebook.isPending}
            onClick={() => exportNotebook.mutate()}>
            <CloudUpload size={15} /> Export Colab notebook
          </button>
          <ErrorNote>{exportNotebook.error?.message}</ErrorNote>
        </Mechanism>
      </div>

      {status.history.length > 0 && (
        <div className="card mt-6 overflow-x-auto">
          <div className="border-b border-slate-100 px-5 py-3 text-sm font-medium text-slate-700">Training history</div>
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
              <tr>{["When", "What", "Version", "Lines", "Errors before", "Errors after", "Result"]
                .map((h) => <th key={h} className="px-4 py-2 font-medium">{h}</th>)}</tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {status.history.slice().reverse().map((h) => (
                <tr key={h.id}>
                  <td className="px-4 py-2 text-slate-500">{new Date(h.created_at).toLocaleString()}</td>
                  <td className="px-4 py-2">{h.mechanism === "trocr_lora" ? "Handwriting" : h.mechanism}</td>
                  <td className="px-4 py-2">{h.version}</td>
                  <td className="px-4 py-2 tabular-nums">{h.num_samples}</td>
                  <td className="px-4 py-2 tabular-nums">{(h.metric_before * 100).toFixed(1)}%</td>
                  <td className="px-4 py-2 tabular-nums">{(h.metric_after * 100).toFixed(1)}%</td>
                  <td className="px-4 py-2">{h.promoted ? <Badge tone="green">in use</Badge> : <Badge>not better, discarded</Badge>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function Mechanism({ n, icon: Icon, title, active, status, children }) {
  return (
    <div className="card p-5">
      <div className="mb-2 flex items-center gap-2.5">
        <div className={clsx("grid h-8 w-8 place-items-center rounded-lg", active ? "bg-emerald-50 text-emerald-600" : "bg-slate-100 text-slate-500")}>
          <Icon size={17} />
        </div>
        <div className="flex-1 font-medium text-slate-900">{n}. {title}</div>
        {active ? <Badge tone="green">active</Badge> : <Badge>waiting</Badge>}
      </div>
      {status && <div className="mb-2 text-xs font-medium text-slate-500">{status}</div>}
      <div className="text-sm text-slate-600">{children}</div>
    </div>
  );
}

const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

function Progress({ value }) {
  return (
    <div className="mt-1.5 h-1.5 rounded-full bg-slate-100">
      <div className="h-full rounded-full bg-brand-500" style={{ width: `${Math.min(100, value * 100)}%` }} />
    </div>
  );
}
