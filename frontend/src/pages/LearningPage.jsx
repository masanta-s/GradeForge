import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { BookMarked, CloudUpload, Cpu, Download, Gauge, Pause, PenTool, Search, Sparkles } from "lucide-react";
import { useState } from "react";
import { api, formatBytes } from "../api";
import { Badge, ErrorNote, JobBar, PageHeader } from "../components";
import { useJob, useSettings } from "../hooks";
import { PlanOptions, TierBadge } from "../modelParts";
import ProgressPanel from "./ProgressPanel";

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
      <ProgressPanel />

      <div className="grid gap-5 md:grid-cols-2">
        <Mechanism n={1} icon={BookMarked} title="Examples from your past marking"
          active={status.few_shot.active}
          status={status.few_shot.active
            ? `${plural(status.few_shot.corrections, "checked answer")} used (${status.few_shot.changed} changed by you)`
            : "Correct a mark or approve a sheet to start"}>
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

        <Mechanism n={4} icon={Sparkles} title="Fine-tuning the grading model" wide
          active={status.history.some((h) => h.mechanism === "llm_lora" && h.promoted)}
          status={`${status.llm.examples} of ${status.llm.recommended} recommended examples`}>
          Trains the grading model's own weights (LoRA) on your checked answers. GradeForge works out where that can
          run for each model: this computer, or a free Colab/Kaggle GPU through a notebook.
          <Progress value={status.llm.examples / status.llm.recommended} />
          <FineTunePanel canExport={status.llm.can_export} />
        </Mechanism>
      </div>

      {status.history.length > 0 && (
        <div className="card mt-6 overflow-x-auto">
          <div className="border-b border-slate-100 px-5 py-3 text-sm font-medium text-slate-700">Training history</div>
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
              <tr>{["When", "What", "Version", "Examples", "Error before", "Error after", "Result"]
                .map((h) => <th key={h} className="px-4 py-2 font-medium">{h}</th>)}</tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {status.history.slice().reverse().map((h) => (
                <tr key={h.id}>
                  <td className="px-4 py-2 text-slate-500">{new Date(h.created_at).toLocaleString()}</td>
                  <td className="px-4 py-2">{h.mechanism === "trocr_lora" ? "Handwriting" : `Grading model (${h.model})`}</td>
                  <td className="px-4 py-2">{h.version}</td>
                  <td className="px-4 py-2 tabular-nums">{h.num_samples}</td>
                  {h.mechanism === "llm_lora" ? (
                    // average mark difference, as % of each question's marks (held-out answers)
                    <>
                      <td className="px-4 py-2 tabular-nums">{h.metric_before != null ? `${h.metric_before.toFixed(1)}%` : "–"}</td>
                      <td className="px-4 py-2 tabular-nums">{h.metric_after != null ? `${h.metric_after.toFixed(1)}%` : "–"}</td>
                    </>
                  ) : (
                    <>
                      <td className="px-4 py-2"><ErrorBar value={h.metric_before} tone="bg-rose-400" /></td>
                      <td className="px-4 py-2"><ErrorBar value={h.metric_after} tone={h.promoted ? "bg-emerald-500" : "bg-slate-400"} /></td>
                    </>
                  )}
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

// Mechanism 4: pick a model, see where it can be fine-tuned, then train it here (layers streamed
// through the GPU) or with a Colab/Kaggle notebook. Either way GradeForge merges the result, checks it
// and only switches to it if it matches the teacher better on held-out answers.
function FineTunePanel({ canExport }) {
  const queryClient = useQueryClient();
  const { data: settings } = useSettings();
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const [model, setModel] = useState(null);
  const chosen = model ?? models.data?.find((m) => m.active)?.name ?? settings?.model;
  const plan = useQuery({ queryKey: ["llm-plan", chosen], queryFn: () => api.llmPlan(chosen), enabled: !!chosen });
  const refresh = () => ["llm-plan", "llm-versions", "learning", "progress", "models", "settings"]
    .forEach((k) => queryClient.invalidateQueries({ queryKey: [k] }));
  const resolve = useJob((j) => j.status === "done" && refresh());
  const [error, setError] = useState(null);
  const p = plan.data?.plan;
  const option = (target) => p?.options.find((o) => o.target === target && o.available);
  const cloud = option("colab") ?? option("kaggle");
  const trainable = models.data?.filter((m) => m.tier.level === "green" && m.name !== chosen) ?? [];
  const base = plan.data?.model ?? chosen;

  const findSource = async () => {
    setError(null);
    try { resolve.start((await api.resolveModel(chosen)).job_id); } catch (e) { setError(e.message); }
  };

  return (
    <div className="mt-3 space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <label className="text-xs text-slate-600" htmlFor="ft-model">Model to fine-tune</label>
        <select id="ft-model" className="input w-auto py-1.5" value={chosen ?? ""} onChange={(e) => setModel(e.target.value)}>
          {models.data?.filter((m) => !m.name.startsWith("gradeforge-"))
            .map((m) => <option key={m.name} value={m.name}>{m.name}{m.active ? " (grading now)" : ""}</option>)}
        </select>
        {plan.data && <TierBadge tier={plan.data.tier} />}
      </div>
      <ErrorNote>{plan.error?.message || models.error?.message}</ErrorNote>

      {plan.data?.needs_source ? (
        <div className="rounded-lg bg-slate-50 p-3 text-xs text-slate-600">
          Fine-tuning needs this model's original weights. GradeForge can find them on HuggingFace (it sends only
          the model's name).
          <div><button className="btn-secondary mt-2 py-1" disabled={resolve.running} onClick={findSource}>
            <Search size={14} /> Find training source
          </button></div>
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

      {option("stream") && <LocalTraining model={base} weightsReady={plan.data.weights_ready} canTrain={canExport}
        onChanged={refresh} />}
      {cloud && <CloudTraining model={base} target={cloud} canExport={canExport}
        weightsReady={plan.data.weights_ready} showDownload={!option("stream")} onChanged={refresh} />}
      {p && <Versions model={base} />}
      <ErrorNote>{error}</ErrorNote>
    </div>
  );
}

function WeightsDownload({ model, onDone }) {
  const size = useQuery({ queryKey: ["weights", model], queryFn: () => api.weights(model, true) });
  const job = useJob((j) => j.status === "done" && onDone());
  const [error, setError] = useState(null);
  const gb = size.data?.total_bytes ? formatBytes(size.data.total_bytes) : "about 19 GB for a 9B model";
  const start = async () => {
    setError(null);
    try { job.start((await api.downloadWeights(model)).job_id); } catch (e) { setError(e.message); }
  };
  return (
    <div className="text-xs text-slate-600">
      Training and merging here need the model's original weights: {gb}, downloaded once from HuggingFace into{" "}
      <code>models/hf_weights</code> on this drive. An interrupted download resumes.
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <button className="btn-secondary py-1" disabled={job.running} onClick={start}>
          <Download size={14} /> Download {size.data?.total_bytes ? formatBytes(size.data.total_bytes) : "weights"}
        </button>
        {size.data?.downloaded_bytes > 0 && !size.data.complete && (
          <span>{formatBytes(size.data.downloaded_bytes)} already downloaded</span>
        )}
      </div>
      <div className="mt-2"><JobBar {...job} /></div>
      <ErrorNote>{error}</ErrorNote>
    </div>
  );
}

function LocalTraining({ model, weightsReady, canTrain, onChanged }) {
  const job = useJob((j) => j.status === "done" && onChanged());
  const [fromScratch, setFromScratch] = useState(false);
  const [error, setError] = useState(null);
  const start = async () => {
    setError(null);
    try { job.start((await api.retrain(model, fromScratch)).job_id); } catch (e) { setError(e.message); }
  };
  const result = job.job?.status === "done" ? job.job.result : null;
  return (
    <div className="rounded-lg border border-slate-200 p-3">
      <div className="mb-1 flex items-center gap-2 text-sm font-medium text-slate-800"><Cpu size={15} /> Train on this computer</div>
      <p className="mb-2 text-xs text-slate-500">
        Private: nothing leaves the computer. The model's layers take turns on the GPU, so it is slow but works on
        8 GB. It stops by itself when held-back answers stop improving, can be paused and resumed, and the new
        version is used only if it marks the held-out answers more like you.
      </p>
      {!weightsReady ? <WeightsDownload model={model} onDone={onChanged} /> : (
        <>
          <div className="flex flex-wrap items-center gap-3">
            <button className="btn-primary py-1.5" disabled={job.running || !canTrain} onClick={start}>
              <Sparkles size={15} /> Train now
            </button>
            {job.running && <button className="btn-ghost py-1.5" onClick={() => api.pauseTraining()}><Pause size={15} /> Pause</button>}
            <label className="flex items-center gap-1.5 text-xs text-slate-600">
              <input type="checkbox" className="accent-brand-600" checked={fromScratch}
                onChange={(e) => setFromScratch(e.target.checked)} />
              start from the original model (otherwise it continues from the version in use)
            </label>
          </div>
          {!canTrain && <p className="mt-2 text-xs text-slate-500">Check or approve some graded written answers first.</p>}
          <div className="mt-2"><JobBar {...job} /></div>
          {job.running && <p className="mt-1 text-xs text-slate-500">Grading jobs wait until training finishes (one GPU).</p>}
          {result && <Verdict meta={result} />}
        </>
      )}
      <ErrorNote>{error}</ErrorNote>
    </div>
  );
}

function CloudTraining({ model, target, canExport, weightsReady, showDownload, onChanged }) {
  const [consent, setConsent] = useState(false);
  const [path, setPath] = useState("");
  const [error, setError] = useState(null);
  const exportNotebook = useMutation({ mutationFn: () => api.exportNotebook(model) });
  const job = useJob((j) => j.status === "done" && onChanged());
  const importAdapter = async (e) => {
    e.preventDefault();
    setError(null);
    try { job.start((await api.importAdapter(model, path.trim())).job_id); } catch (err) { setError(err.message); }
  };
  return (
    <div className="rounded-lg border border-slate-200 p-3">
      <div className="mb-1 flex items-center gap-2 text-sm font-medium text-slate-800"><CloudUpload size={15} /> {target.label}</div>
      <p className="mb-2 text-xs text-slate-500">{target.reason}. The notebook trains the same adapter; you bring it back here.</p>
      <label className="flex items-start gap-2 text-xs text-slate-700">
        <input type="checkbox" className="mt-0.5 accent-brand-600" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
        I understand the notebook contains students' answers (no names) and will leave this computer.
      </label>
      <button className="btn-secondary mt-2 py-1.5" disabled={!consent || !canExport || exportNotebook.isPending}
        onClick={() => exportNotebook.mutate()}>
        <CloudUpload size={15} /> Export notebook
      </button>
      <form className="mt-3 space-y-2 border-t border-slate-100 pt-3" onSubmit={importAdapter}>
        <div className="text-xs font-medium text-slate-700">Import the trained adapter (unzip gradeforge_adapter.zip first)</div>
        {!weightsReady && (showDownload ? <WeightsDownload model={model} onDone={onChanged} />
          : <p className="text-xs text-slate-500">Download the original weights above first: the adapter is merged into them here.</p>)}
        <div className="flex flex-wrap gap-2">
          <input className="input w-auto flex-1 py-1.5" placeholder="F:\Downloads\gradeforge_adapter" aria-label="Adapter folder"
            value={path} onChange={(e) => setPath(e.target.value)} />
          <button className="btn-primary py-1.5" disabled={!path.trim() || !weightsReady || job.running}>Import</button>
        </div>
      </form>
      <div className="mt-2"><JobBar {...job} /></div>
      {job.job?.status === "done" && <Verdict meta={job.job.result} />}
      <ErrorNote>{exportNotebook.error?.message || error}</ErrorNote>
    </div>
  );
}

function Verdict({ meta }) {
  const good = meta.promoted;
  return (
    <p className={clsx("mt-2 text-sm", good ? "text-emerald-700" : "text-slate-700")}>
      {meta.ollama_name}: {meta.verdict}.{" "}
      {meta.before && meta.after && (
        <>Average difference {meta.before.mae.toFixed(2)} → {meta.after.mae.toFixed(2)} marks;
          within half a mark {Math.round(meta.before.close * 100)}% → {Math.round(meta.after.close * 100)}%.</>
      )}
      {good ? " Now grading with it." : " Still grading with the previous model."}
    </p>
  );
}

function Versions({ model }) {
  const { data } = useQuery({ queryKey: ["llm-versions", model], queryFn: () => api.llmVersions(model) });
  if (!data?.length) return null;
  return (
    <div className="overflow-x-auto">
      <div className="mb-1 text-xs font-medium uppercase tracking-wide text-slate-500">Fine-tuned versions</div>
      <table className="w-full text-xs">
        <thead className="text-left text-slate-500">
          <tr>{["Version", "Trained on", "How", "Avg difference", "Within ½ mark", "Result"]
            .map((h) => <th key={h} className="py-1 pr-3 font-medium">{h}</th>)}</tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {data.slice().reverse().map((v) => (
            <tr key={v.ollama_name}>
              <td className="py-1.5 pr-3 font-medium text-slate-800">{v.ollama_name.split(":")[1]}</td>
              <td className="py-1.5 pr-3 tabular-nums">{plural(v.examples ?? 0, "answer")}</td>
              <td className="py-1.5 pr-3">{v.route === "stream" ? "this computer" : "notebook"}</td>
              <td className="py-1.5 pr-3 tabular-nums">{v.before && v.after ? `${v.before.mae.toFixed(2)} → ${v.after.mae.toFixed(2)}` : "–"}</td>
              <td className="py-1.5 pr-3 tabular-nums">{v.before && v.after ? `${Math.round(v.before.close * 100)}% → ${Math.round(v.after.close * 100)}%` : "–"}</td>
              <td className="py-1.5 pr-3">{v.promoted ? <Badge tone="green">used</Badge> : <Badge>{v.verdict ?? "kept aside"}</Badge>}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}


function Mechanism({ n, icon: Icon, title, active, status, children, wide }) {
  return (
    <div className={clsx("card p-5", wide && "md:col-span-2")}>
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
