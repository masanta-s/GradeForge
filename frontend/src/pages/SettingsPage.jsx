import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { ChevronDown, Cpu, Download, Eye, EyeOff, KeyRound, Save, Star } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../api";
import { Badge, ErrorNote, JobBar, PageHeader, StrictnessSlider } from "../components";
import { useJob, useSettings } from "../hooks";
import { TierBadge } from "../modelParts";
import CloudSettings from "./CloudSettings";
import ModelCard from "./ModelCard";
import StoragePanel from "./StoragePanel";

export default function SettingsPage() {
  const { data: settings } = useSettings();
  const { data: health } = useQuery({ queryKey: ["health"], queryFn: api.health });
  if (!settings) return null;
  const subtitle = health?.provider === "cloud"
    ? `Grading with ${health.model} through its cloud API.`
    : `Grading on this computer with ${settings.model} through Ollama${health && !health.ollama ? " (Ollama is not running)" : ""}.`;

  return (
    <div className="mx-auto max-w-5xl">
      <PageHeader title="Models & settings" subtitle={subtitle} />
      <div className="space-y-5">
        <ModelsSection ollama={health?.ollama} />
        <CloudSettings />
        <GeneralSettings settings={settings} />
        <StoragePanel />
      </div>
    </div>
  );
}

function ModelsSection({ ollama }) {
  const queryClient = useQueryClient();
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const [open, setOpen] = useState(null);
  const invalidate = () => ["models", "settings", "health", "gpu"].forEach((k) => queryClient.invalidateQueries({ queryKey: [k] }));

  return (
    <div className="card overflow-hidden">
      <div className="p-5 pb-3">
        <div className="mb-1 flex items-center gap-2 font-medium text-slate-900"><Cpu size={17} /> Local models</div>
        <p className="text-sm text-slate-500">
          Every model Ollama has installed{ollama ? ` (Ollama ${ollama})` : ""}. A model is checked before it grades;
          open one to see its check, its training source and where it could be fine-tuned.
        </p>
        <GpuBar />
      </div>
      <div className="px-5 pb-2"><ErrorNote>{models.error?.message}</ErrorNote></div>
      <ul className="divide-y divide-slate-100 border-t border-slate-100">
        {models.data?.map((m) => (
          <li key={m.name}>
            <button className={clsx("flex w-full flex-wrap items-center gap-x-3 gap-y-1.5 px-5 py-3 text-left hover:bg-slate-50",
              m.active && "bg-brand-50/50")} onClick={() => setOpen(open === m.name ? null : m.name)} aria-expanded={open === m.name}>
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2 font-medium text-slate-900">
                  {m.name}
                  {m.active && <Badge tone="blue">in use</Badge>}
                  {m.default && !m.active && <Badge><Star size={11} /> recommended</Badge>}
                </div>
                <div className="text-xs text-slate-500">
                  {m.architecture} · {m.parameter_size} · {m.quantization} · {m.context_length?.toLocaleString()} context
                </div>
              </div>
              <TierBadge tier={m.tier} />
              <VisionBadge model={m} />
              {m.vram_gib != null && (
                <Badge tone={m.fully_on_gpu ? "green" : "amber"}>{m.vram_gib} GiB {m.fully_on_gpu ? "on GPU" : "partly on CPU"}</Badge>
              )}
              <ChevronDown size={16} className={clsx("text-slate-400 transition-transform", open === m.name && "rotate-180")} />
            </button>
            {open === m.name && <ModelCard name={m.name} onChanged={invalidate} />}
          </li>
        ))}
      </ul>
      <div className="space-y-4 border-t border-slate-100 p-5">
        <PullModel onDone={invalidate} />
        <HfToken />
      </div>
    </div>
  );
}

function VisionBadge({ model }) {
  if (model.vision_ok === true) return <Badge tone="green"><Eye size={11} /> reads images</Badge>;
  if (model.vision_ok === false && model.capabilities.includes("vision")) {
    return <Badge tone="amber" className="whitespace-nowrap"><EyeOff size={11} /> vision failed check</Badge>;
  }
  if (model.capabilities.includes("vision")) return <Badge><Eye size={11} /> vision</Badge>;
  return null;
}

function GpuBar() {
  const { data } = useQuery({ queryKey: ["gpu"], queryFn: api.gpu, refetchInterval: 10_000 });
  if (!data?.gpu) return null;
  const pct = Math.min(100, (data.used_gb / data.total_gb) * 100);
  return (
    <div className="mt-3">
      <div className="flex justify-between text-xs text-slate-500">
        <span>{data.gpu}</span><span className="tabular-nums">{data.used_gb.toFixed(1)} / {data.total_gb.toFixed(1)} GB in use</span>
      </div>
      <div className="mt-1 h-2 overflow-hidden rounded-full bg-slate-100" role="meter" aria-valuenow={Math.round(pct)}
        aria-valuemin={0} aria-valuemax={100} aria-label="GPU memory in use">
        <div className={clsx("h-full rounded-full", pct > 90 ? "bg-amber-500" : "bg-brand-500")} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function PullModel({ onDone }) {
  const [name, setName] = useState("");
  const [error, setError] = useState(null);
  const job = useJob((j) => { if (j.status === "done") { setName(""); onDone(); } });
  const start = async (e) => {
    e.preventDefault();
    setError(null);
    try { job.start((await api.pullModel(name.trim())).job_id); } catch (err) { setError(err.message); }
  };
  return (
    <div>
      <label className="label" htmlFor="pull">Download a model from the Ollama library</label>
      <form className="flex flex-wrap gap-2" onSubmit={start}>
        <input id="pull" className="input w-auto flex-1" placeholder="e.g. qwen3.5:4b" value={name}
          onChange={(e) => setName(e.target.value)} />
        <button className="btn-secondary" disabled={!name.trim() || job.running}><Download size={15} /> Download</button>
      </form>
      <p className="mt-1.5 text-xs text-slate-500">Saved in Ollama's own model folder; GPU memory is measured afterwards.</p>
      <div className="mt-2"><JobBar {...job} /></div>
      <ErrorNote>{error}</ErrorNote>
    </div>
  );
}

function HfToken() {
  const queryClient = useQueryClient();
  const status = useQuery({ queryKey: ["hf-token"], queryFn: api.hfToken });
  const [token, setToken] = useState("");
  const done = () => { setToken(""); queryClient.invalidateQueries({ queryKey: ["hf-token"] }); };
  const save = useMutation({ mutationFn: () => api.setHfToken(token), onSuccess: done });
  const remove = useMutation({ mutationFn: api.deleteHfToken, onSuccess: done });
  return (
    <div>
      <label className="label" htmlFor="hf-token"><KeyRound size={13} className="mr-1 inline" />HuggingFace token (optional)</label>
      {status.data?.has_hf_token ? (
        <div className="flex items-center gap-2 text-sm">
          <Badge tone="green">saved in Windows Credential Manager</Badge>
          <button className="btn-ghost py-1 text-rose-700" onClick={() => remove.mutate()}>Remove</button>
        </div>
      ) : (
        <form className="flex flex-wrap gap-2" onSubmit={(e) => { e.preventDefault(); save.mutate(); }}>
          <input id="hf-token" type="password" autoComplete="off" className="input w-auto flex-1" placeholder="hf_…"
            value={token} onChange={(e) => setToken(e.target.value)} />
          <button className="btn-secondary" disabled={!token.trim() || save.isPending}>Save</button>
        </form>
      )}
      <p className="mt-1.5 text-xs text-slate-500">Only needed for gated models or if HuggingFace rate-limits the look-ups.</p>
      <ErrorNote>{save.error?.message || remove.error?.message}</ErrorNote>
    </div>
  );
}

function GeneralSettings({ settings }) {
  const queryClient = useQueryClient();
  const [form, setForm] = useState({ strictness: settings.strictness, teacher_name: settings.teacher_name });
  useEffect(() => { setForm({ strictness: settings.strictness, teacher_name: settings.teacher_name }); },
    [settings.strictness, settings.teacher_name]);
  const save = useMutation({
    mutationFn: () => api.saveSettings(form),
    onSuccess: (s) => queryClient.setQueryData(["settings"], s),
  });
  const dirty = form.strictness !== settings.strictness || form.teacher_name !== settings.teacher_name;

  return (
    <div className="grid gap-5 md:grid-cols-2">
      <div>
        <StrictnessSlider value={form.strictness} onChange={(v) => setForm({ ...form, strictness: v })} />
        <p className="mt-2 text-xs text-slate-500">Default for new grading. You can still preview any strictness per sheet.</p>
      </div>
      <div className="card h-fit p-5">
        <label className="label" htmlFor="teacher">Your name</label>
        <input id="teacher" className="input" value={form.teacher_name}
          onChange={(e) => setForm({ ...form, teacher_name: e.target.value })} placeholder="Mrs. Iyer" />
        <p className="mt-2 text-xs text-slate-500">Pre-filled when you resolve disputes or change marks (audit trail).</p>
        <button className="btn-primary mt-4" disabled={!dirty || save.isPending} onClick={() => save.mutate()}>
          <Save size={16} /> Save
        </button>
        <div className="mt-3"><ErrorNote>{save.error?.message}</ErrorNote></div>
      </div>
    </div>
  );
}
