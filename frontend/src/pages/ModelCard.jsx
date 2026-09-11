import { useQuery } from "@tanstack/react-query";
import { ExternalLink, PlayCircle, Search } from "lucide-react";
import { useState } from "react";
import { api } from "../api";
import { Badge, ErrorNote, JobBar } from "../components";
import { useJob } from "../hooks";
import { CheckList, PlanOptions, TierBadge } from "../modelParts";

const CONFIDENCE_TONE = { high: "green", medium: "amber", low: "red", none: "gray" };

// Everything GradeForge knows about one installed model: the capability check, its trainable
// HuggingFace source, whether the fine-tuning pipeline supports it, and where it could train.
export default function ModelCard({ name, onChanged }) {
  const card = useQuery({ queryKey: ["model-card", name], queryFn: () => api.modelCard(name) });
  const refresh = () => { card.refetch(); onChanged?.(); };
  const probe = useJob((j) => j.status === "done" && refresh());
  const resolve = useJob((j) => j.status === "done" && refresh());
  const use = useJob((j) => j.status === "done" && refresh());
  const [error, setError] = useState(null);
  const run = (job, call) => async () => {
    setError(null);
    try { job.start((await call()).job_id); } catch (e) { setError(e.message); }
  };

  if (card.error) return <ErrorNote>{card.error.message}</ErrorNote>;
  const m = card.data;
  if (!m) return <div className="px-4 py-6 text-sm text-slate-500">Loading…</div>;
  const busy = probe.running || resolve.running || use.running;

  return (
    <div className="space-y-4 border-t border-slate-100 bg-slate-50/60 px-4 py-4">
      <div className="flex flex-wrap items-center gap-2">
        <TierBadge tier={m.tier} />
        <p className="flex-1 text-sm text-slate-600">{m.tier.reason}</p>
        {m.active ? <Badge tone="blue">grading with this model</Badge> : (
          <button className="btn-primary py-1.5" disabled={busy || m.tier.level === "red"}
            onClick={run(use, () => api.switchModel(name))}>
            Use for grading
          </button>
        )}
      </div>
      <JobBar {...use} />

      <div className="grid gap-4 lg:grid-cols-2">
        <section className="card p-4">
          <div className="mb-3 flex items-center justify-between gap-2">
            <h3 className="text-sm font-semibold text-slate-900">Capability check</h3>
            <button className="btn-secondary py-1 text-xs" disabled={busy} onClick={run(probe, () => api.probeModel(name))}>
              <PlayCircle size={14} /> {m.probe ? "Check again" : "Run check (~30 s)"}
            </button>
          </div>
          {m.probe ? (
            <>
              <CheckList items={m.probe.results.map((r) => ({
                title: r.label, status: r.status, detail: r.detail,
                extra: r.critical ? `${r.seconds}s · required` : `${r.seconds}s · optional`,
              }))} />
              <p className="mt-3 text-xs text-slate-400">
                {m.probe.native_json ? "Returns valid JSON natively." : "Needs its JSON cleaned up or repaired."}{" "}
                Checked {new Date(m.probe.created_at * 1000).toLocaleString()}.
              </p>
            </>
          ) : (
            <p className="text-sm text-slate-500">
              Tests JSON output, following instructions, and marking a known answer before the model grades
              anything, plus whether it can really read images (for diagrams).
            </p>
          )}
          <div className="mt-3"><JobBar {...probe} /></div>
        </section>

        <section className="card p-4">
          <h3 className="mb-3 text-sm font-semibold text-slate-900">Fine-tuning</h3>
          <SourcePanel model={m} busy={busy} onResolve={run(resolve, () => api.resolveModel(name))}
            onSetRepo={(repo) => run(resolve, () => api.setRepo(name, repo))()} />
          <div className="mt-3"><JobBar {...resolve} /></div>
          {m.trainability && (
            <div className="mt-4 border-t border-slate-100 pt-3">
              <div className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">Pipeline support</div>
              <CheckList items={m.trainability.checks.map((c) => ({ title: c.component, status: c.status, detail: c.detail }))} />
            </div>
          )}
          {m.training_plan && (
            <div className="mt-4 border-t border-slate-100 pt-3">
              <div className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">Where it can train</div>
              <PlanOptions plan={m.training_plan} showBlocked={false} />
            </div>
          )}
        </section>
      </div>
      <ErrorNote>{error}</ErrorNote>
    </div>
  );
}

function SourcePanel({ model, busy, onResolve, onSetRepo }) {
  const r = model.resolution;
  const [editing, setEditing] = useState(false);
  const [repo, setRepo] = useState("");

  if (!r) {
    return (
      <div className="text-sm text-slate-600">
        <p>
          Fine-tuning needs this model's original weights, which Ollama doesn't have. GradeForge can look them
          up on HuggingFace (it sends only the model's name).
        </p>
        <button className="btn-secondary mt-3 py-1.5" disabled={busy} onClick={onResolve}>
          <Search size={15} /> Find training source
        </button>
      </div>
    );
  }
  return (
    <div className="text-sm">
      <div className="flex flex-wrap items-center gap-2">
        {r.repo ? (
          <a className="inline-flex items-center gap-1 font-medium text-brand-700 hover:underline"
            href={`https://huggingface.co/${r.repo}`} target="_blank" rel="noreferrer">
            {r.repo} <ExternalLink size={13} />
          </a>
        ) : <span className="font-medium text-slate-700">No trainable source found</span>}
        <Badge tone={CONFIDENCE_TONE[r.confidence]}>{r.confidence} confidence</Badge>
        {r.source !== "auto" && <Badge>{r.source === "manual" ? "chosen by you" : "registry fix"}</Badge>}
      </div>
      {r.reasons.length > 0 && (
        <ul className="mt-1.5 list-disc space-y-0.5 pl-5 text-xs text-slate-500">
          {r.reasons.map((reason) => <li key={reason}>{reason}</li>)}
        </ul>
      )}
      {editing ? (
        <form className="mt-3 flex flex-wrap gap-2" onSubmit={(e) => { e.preventDefault(); onSetRepo(repo.trim()); setEditing(false); }}>
          <input className="input w-auto flex-1 py-1.5" placeholder="organisation/model-name" value={repo}
            onChange={(e) => setRepo(e.target.value)} aria-label="HuggingFace repository" autoFocus />
          <button className="btn-primary py-1.5" disabled={!repo.includes("/") || busy}>Use this repo</button>
          <button type="button" className="btn-ghost py-1.5" onClick={() => setEditing(false)}>Cancel</button>
        </form>
      ) : (
        <div className="mt-2 flex flex-wrap gap-3 text-xs">
          <button className="text-brand-700 hover:underline" onClick={() => { setRepo(r.repo ?? ""); setEditing(true); }}>Not right?</button>
          {r.source === "manual" && <button className="text-brand-700 hover:underline" disabled={busy} onClick={() => onSetRepo(null)}>Use automatic</button>}
          <button className="text-brand-700 hover:underline" disabled={busy} onClick={onResolve}>Look up again</button>
        </div>
      )}
    </div>
  );
}
