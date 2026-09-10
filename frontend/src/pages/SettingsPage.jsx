import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { Cpu, Eye, Save, Star } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../api";
import { Badge, ErrorNote, PageHeader, StrictnessSlider } from "../components";
import { useSettings } from "../hooks";

export default function SettingsPage() {
  const queryClient = useQueryClient();
  const { data: settings } = useSettings();
  const { data: models, error: modelsError } = useQuery({ queryKey: ["models"], queryFn: api.models });
  const { data: health } = useQuery({ queryKey: ["health"], queryFn: api.health });
  const [form, setForm] = useState(null);
  useEffect(() => { if (settings && !form) setForm(settings); }, [settings, form]);
  const save = useMutation({
    mutationFn: () => api.saveSettings(form),
    onSuccess: (s) => { queryClient.setQueryData(["settings"], s); setForm(s); },
  });
  if (!form) return null;
  const dirty = JSON.stringify(form) !== JSON.stringify(settings);

  return (
    <div className="mx-auto max-w-4xl">
      <PageHeader title="Models & settings" subtitle="Everything runs on this computer through Ollama.">
        <button className="btn-primary" disabled={!dirty || save.isPending} onClick={() => save.mutate()}>
          <Save size={16} /> Save
        </button>
      </PageHeader>

      <div className="space-y-5">
        <div className="card p-5">
          <div className="mb-1 flex items-center gap-2 font-medium text-slate-900"><Cpu size={17} /> Grading model</div>
          <p className="mb-4 text-sm text-slate-500">
            Ollama {health?.ollama ?? "not reachable"}. VRAM figures are measured on this GPU, not estimated.
          </p>
          <ErrorNote>{modelsError?.message}</ErrorNote>
          <div className="space-y-2">
            {models?.map((m) => (
              <label key={m.name} className={clsx("flex cursor-pointer items-center gap-3 rounded-lg border p-3",
                form.model === m.name ? "border-brand-500 bg-brand-50" : "border-slate-200 hover:bg-slate-50")}>
                <input type="radio" className="accent-brand-600" checked={form.model === m.name}
                  onChange={() => setForm({ ...form, model: m.name })} />
                <div className="flex-1">
                  <div className="flex items-center gap-2 font-medium text-slate-900">
                    {m.name} {m.default && <Badge tone="blue"><Star size={11} /> recommended</Badge>}
                  </div>
                  <div className="text-xs text-slate-500">
                    {m.architecture} · {m.parameter_size} · {m.quantization} · {m.context_length?.toLocaleString()} context
                  </div>
                </div>
                {m.capabilities.includes("vision") && <Badge><Eye size={11} /> vision</Badge>}
                {m.vram_gib != null && (
                  <Badge tone={m.fully_on_gpu ? "green" : "amber"}>
                    {m.vram_gib} GiB {m.fully_on_gpu ? "on GPU" : "partly on CPU"}
                  </Badge>
                )}
              </label>
            ))}
          </div>
        </div>

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
          </div>
        </div>
        <ErrorNote>{save.error?.message}</ErrorNote>
      </div>
    </div>
  );
}
