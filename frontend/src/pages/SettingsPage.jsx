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
        <Credentials />
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

// Credentials all come from the .env file in the project folder, so this is read-only: it shows
// which are set (masked) and can prove the Kaggle one works.
function Credentials() {
  const { data } = useQuery({ queryKey: ["secrets"], queryFn: api.secrets });
  const { data: kaggle } = useQuery({ queryKey: ["kaggle"], queryFn: api.kaggle });
  const check = useMutation({ mutationFn: api.checkKaggle });
  const quota = check.data;
  if (!data) return null;

  return (
    <div>
      <div className="label"><KeyRound size={13} className="mr-1 inline" />Accounts and API keys</div>
      <p className="mb-2 text-sm text-slate-500">
        All optional: grading needs no account. They are read from the <code>.env</code> file in the project
        folder (copy <code>.env.example</code>, fill in what you use, restart GradeForge). GradeForge never
        writes them anywhere.
      </p>
      <ul className="space-y-1.5 text-sm">
        {data.map((s) => (
          <li key={s.name} className="flex flex-wrap items-center gap-2">
            <code className="w-52 shrink-0 text-xs text-slate-600">{s.variable}</code>
            {s.set ? <Badge tone="green">set · {s.masked}</Badge> : <Badge>not set</Badge>}
            <span className="text-xs text-slate-500">{LABELS[s.name]}</span>
            {s.name === "kaggle" && s.set && (
              <>
                <button className="btn-secondary py-0.5 text-xs" disabled={check.isPending} onClick={() => check.mutate()}>
                  {check.isPending ? "Checking…" : "Test connection"}
                </button>
                {quota && (
                  <span className="text-xs text-slate-600">
                    {quota.username}
                    {quota.gpu_hours_left != null && ` · ${quota.gpu_hours_left} of ${quota.gpu_hours_total} GPU hours left this week`}
                  </span>
                )}
              </>
            )}
          </li>
        ))}
      </ul>
      {kaggle && !kaggle.has_token && (
        <p className="mt-2 text-xs text-slate-500">
          Kaggle token: kaggle.com → your avatar → Settings → API → Create New Token. Notebooks also need
          Phone Verification on your Kaggle account.
        </p>
      )}
      <ErrorNote>{check.error?.message}</ErrorNote>
    </div>
  );
}

const LABELS = {
  kaggle: "run the fine-tuning notebook on Kaggle's free GPUs",
  huggingface: "gated models and higher download limits",
  openai: "cloud grading with OpenAI",
  anthropic: "cloud grading with Anthropic",
  gemini: "cloud grading with Google Gemini",
};

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
