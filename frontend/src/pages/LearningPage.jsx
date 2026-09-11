import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { BookMarked, CloudUpload, Gauge, PenTool, Search, Sparkles } from "lucide-react";
import { useState } from "react";
import { api } from "../api";
import { Badge, ErrorNote, JobBar, PageHeader } from "../components";
import { useJob, useSettings } from "../hooks";
import { PlanOptions, TierBadge } from "../modelParts";

export default function LearningPage() {
  const queryClient = useQueryClient();
  const { data: status } = useQuery({ queryKey: ["learning"], queryFn: api.learning });
  const job = useJob((j) => { if (j.status === "done") queryClient.invalidateQueries({ queryKey: ["learning"] }); });
  const [writer, setWriter] = useState("");
  const [trainError, setTrainError] = useState(null);
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
          Trains the grading model's own weights on your corrections. GradeForge works out where that can run
          for each model: this computer, or a free Colab/Kaggle GPU through a notebook.
          <Progress value={status.llm.examples / status.llm.recommended} />
          <FineTunePanel canExport={status.llm.can_export} />
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
                  <td className="px-4 py-2"><ErrorBar value={h.metric_before} tone="bg-rose-400" /></td>
                  <td className="px-4 py-2"><ErrorBar value={h.metric_after} tone={h.promoted ? "bg-emerald-500" : "bg-slate-400"} /></td>
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

// Mechanism 4: pick a model, see where (or whether) it can be fine-tuned, export the notebook,
// and import the trained GGUF back into Ollama.
function FineTunePanel({ canExport }) {
  const queryClient = useQueryClient();
  const { data: settings } = useSettings();
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const [model, setModel] = useState(null);
  const chosen = model ?? settings?.model;
  const plan = useQuery({ queryKey: ["llm-plan", chosen], queryFn: () => api.llmPlan(chosen), enabled: !!chosen });
  const resolve = useJob((j) => j.status === "done" && queryClient.invalidateQueries({ queryKey: ["llm-plan"] }));
  const [consent, setConsent] = useState(false);
  const [error, setError] = useState(null);
  const exportNotebook = useMutation({ mutationFn: () => api.exportNotebook(chosen) });
  const p = plan.data?.plan;
  const trainable = models.data?.filter((m) => m.tier.level === "green" && m.name !== chosen) ?? [];

  const findSource = async () => {
    setError(null);
    try { resolve.start((await api.resolveModel(chosen)).job_id); } catch (e) { setError(e.message); }
  };

  return (
    <div className="mt-3 space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <label className="text-xs text-slate-600" htmlFor="ft-model">Model to fine-tune</label>
        <select id="ft-model" className="input w-auto py-1.5" value={chosen ?? ""} onChange={(e) => setModel(e.target.value)}>
          {models.data?.map((m) => <option key={m.name} value={m.name}>{m.name}{m.active ? " (grading now)" : ""}</option>)}
        </select>
        {plan.data && <TierBadge tier={plan.data.tier} />}
      </div>
      <ErrorNote>{plan.error?.message || models.error?.message}</ErrorNote>

      {plan.data?.needs_source ? (
        <div className="rounded-lg bg-slate-50 p-3 text-xs text-slate-600">
          Fine-tuning needs this model's original weights. GradeForge can find them on HuggingFace (it sends only
          the model's name).
          <button className="btn-secondary mt-2 py-1" disabled={resolve.running} onClick={findSource}>
            <Search size={14} /> Find training source
          </button>
          <div className="mt-2"><JobBar {...resolve} /></div>
        </div>
      ) : p && (
        <div className="rounded-lg bg-slate-50 p-3">
          {p.repo && <div className="mb-2 text-xs text-slate-500">Trains <b className="text-slate-700">{p.repo}</b></div>}
          <PlanOptions plan={p} />
          {!p.recommended && trainable.length > 0 && (
            <p className="mt-2 text-xs text-slate-600">
              Tip: {trainable.map((m) => m.name).join(" or ")} can be fine-tuned ({trainable[0].tier.where}). Few-shot
              examples and calibration keep learning either way.
            </p>
          )}
        </div>
      )}

      {p?.recommended && (
        <>
          <label className="flex items-start gap-2 text-xs text-slate-700">
            <input type="checkbox" className="mt-0.5 accent-brand-600" checked={consent}
              onChange={(e) => setConsent(e.target.checked)} />
            I understand the notebook contains students' answers (no names) and will leave this computer.
          </label>
          <button className="btn-secondary py-1.5" disabled={!consent || !canExport || exportNotebook.isPending}
            onClick={() => exportNotebook.mutate()}>
            <CloudUpload size={15} /> Export {p.recommended === "local" ? "training" : "Colab/Kaggle"} notebook
          </button>
        </>
      )}
      <ErrorNote>{exportNotebook.error?.message || error}</ErrorNote>
      <ImportModel base={chosen} />
    </div>
  );
}

function ImportModel({ base }) {
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState({ path: "", name: "gradeforge-grader" });
  const [error, setError] = useState(null);
  const job = useJob((j) => j.status === "done" && queryClient.invalidateQueries({ queryKey: ["models"] }));
  const start = async (e) => {
    e.preventDefault();
    setError(null);
    try { job.start((await api.importModel({ ...form, base })).job_id); } catch (err) { setError(err.message); }
  };
  if (!open) {
    return <button className="text-xs text-brand-700 hover:underline" onClick={() => setOpen(true)}>Import a trained model (.gguf)…</button>;
  }
  return (
    <form className="space-y-2 rounded-lg border border-slate-200 p-3" onSubmit={start}>
      <div className="text-xs font-medium text-slate-700">Import the .gguf downloaded from Colab/Kaggle</div>
      <input className="input py-1.5" placeholder="F:\Downloads\gradeforge-grader.Q4_K_M.gguf" aria-label="Path to the .gguf file"
        value={form.path} onChange={(e) => setForm({ ...form, path: e.target.value })} />
      <div className="flex flex-wrap items-center gap-2">
        <input className="input w-auto flex-1 py-1.5" aria-label="Name in Ollama" value={form.name}
          onChange={(e) => setForm({ ...form, name: e.target.value })} />
        <span className="text-xs text-slate-500">uses {base}'s chat template</span>
        <button className="btn-primary py-1.5" disabled={!form.path.trim() || job.running}>Import</button>
      </div>
      <JobBar {...job} />
      {job.job?.status === "done" && (
        <p className="text-xs text-emerald-700">
          Registered {job.job.result.ollama_name}. Run its check under Models & settings, then compare it on a few corrected sheets.
        </p>
      )}
      <ErrorNote>{error}</ErrorNote>
    </form>
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

// Character error rate as a bar (0-50 % scale) plus the number: before/after at a glance.
function ErrorBar({ value, tone }) {
  return (
    <div className="flex items-center gap-2">
      <div className="h-2 w-24 rounded-full bg-slate-100">
        <div className={clsx("h-full rounded-full", tone)} style={{ width: `${Math.min(100, value * 200)}%` }} />
      </div>
      <span className="tabular-nums">{(value * 100).toFixed(1)}%</span>
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
