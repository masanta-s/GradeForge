import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { Cloud, HardDrive, KeyRound, ShieldAlert } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../api";
import { Badge, ErrorNote } from "../components";

// Opt-in cloud grading. The API key comes from the project's .env file (shown masked only);
// switching to cloud needs an explicit consent tick because exam data leaves the computer.
export default function CloudSettings() {
  const queryClient = useQueryClient();
  const { data } = useQuery({ queryKey: ["cloud"], queryFn: api.cloud });
  const [form, setForm] = useState(null);
  useEffect(() => { if (data && !form) setForm({ ...data.settings, consent: data.settings.cloud_consent }); }, [data, form]);

  const save = useMutation({
    mutationFn: () => api.saveCloudSettings({
      provider: form.provider, cloud_provider: form.cloud_provider, cloud_model: form.cloud_model, consent: form.consent,
    }),
    onSuccess: () => ["cloud", "settings", "health", "models"].forEach((k) => queryClient.invalidateQueries({ queryKey: [k] })),
  });
  if (!data || !form) return null;

  const provider = data.providers.find((p) => p.id === form.cloud_provider);
  const cloud = form.provider === "cloud";
  const dirty = form.provider !== data.settings.provider || (cloud && (form.cloud_provider !== data.settings.cloud_provider
    || form.cloud_model !== data.settings.cloud_model || form.consent !== data.settings.cloud_consent));

  return (
    <div className="card p-5">
      <div className="mb-1 flex items-center gap-2 font-medium text-slate-900"><Cloud size={17} /> Where grading runs</div>
      <p className="mb-4 text-sm text-slate-500">
        {data.settings.active ? `Grading with ${data.settings.cloud_model} (cloud).` : "Grading on this computer."}
      </p>

      <div className="grid gap-2 sm:grid-cols-2">
        <Choice active={!cloud} icon={HardDrive} title="On this computer (Ollama)" onClick={() => setForm({ ...form, provider: "ollama" })}>
          Free and private: nothing leaves the computer.
        </Choice>
        <Choice active={cloud} icon={Cloud} title="Cloud API" onClick={() => setForm({ ...form, provider: "cloud" })}>
          Your own API key. Needs your consent: exam data goes to the provider.
        </Choice>
      </div>

      {cloud && (
        <div className="mt-5 space-y-4">
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label className="label" htmlFor="cloud-provider">Provider</label>
              <select id="cloud-provider" className="input" value={form.cloud_provider}
                onChange={(e) => setForm({ ...form, cloud_provider: e.target.value, cloud_model: "" })}>
                {data.providers.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
              </select>
            </div>
            <div>
              <label className="label" htmlFor="cloud-model">Model</label>
              <input id="cloud-model" className="input" list="cloud-models" value={form.cloud_model}
                placeholder="Type or choose a model" onChange={(e) => setForm({ ...form, cloud_model: e.target.value })} />
              <datalist id="cloud-models">
                {(data.models[form.cloud_provider] ?? []).map((m) => <option key={m} value={m} />)}
              </datalist>
            </div>
          </div>

          <ApiKey provider={provider} model={form.cloud_model} />
          {form.cloud_model && <ModelFacts model={form.cloud_model} />}

          <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
            <div className="mb-1 flex items-center gap-2 font-medium"><ShieldAlert size={16} /> Privacy</div>
            Cloud grading sends questions, answer keys and students' answers to {provider?.label}. Few-shot examples
            and calibration keep learning from your corrections; the model's weights can't be fine-tuned.
            <label className="mt-3 flex items-start gap-2">
              <input type="checkbox" className="mt-0.5 accent-brand-600" checked={form.consent}
                onChange={(e) => setForm({ ...form, consent: e.target.checked })} />
              I understand that student data leaves this computer.
            </label>
          </div>
        </div>
      )}

      <div className="mt-4 flex items-center gap-3">
        <button className="btn-primary" disabled={!dirty || save.isPending || (cloud && !form.consent)} onClick={() => save.mutate()}>
          {cloud ? "Use cloud model" : "Use this computer"}
        </button>
        {save.isSuccess && !dirty && <span className="text-sm text-emerald-700">Saved.</span>}
      </div>
      <div className="mt-3"><ErrorNote>{save.error?.message}</ErrorNote></div>
    </div>
  );
}

function Choice({ active, icon: Icon, title, onClick, children }) {
  return (
    <button type="button" onClick={onClick} aria-pressed={active}
      className={clsx("rounded-lg border p-3 text-left", active ? "border-brand-500 bg-brand-50" : "border-slate-200 hover:bg-slate-50")}>
      <div className="flex items-center gap-2 text-sm font-medium text-slate-900"><Icon size={16} /> {title}</div>
      <div className="mt-0.5 text-xs text-slate-500">{children}</div>
    </button>
  );
}

// The key comes from .env; the app never stores one. This says whether it is there and tests it.
function ApiKey({ provider, model }) {
  const test = useMutation({ mutationFn: () => api.testCloudKey(provider.id, model) });
  if (!provider) return null;
  return (
    <div>
      <div className="label"><KeyRound size={13} className="mr-1 inline" />{provider.label} API key</div>
      {provider.has_key ? (
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <code className="rounded bg-slate-100 px-2 py-1">{provider.masked_key}</code>
          <Badge tone="green">from {provider.source}</Badge>
          <button className="btn-secondary py-1" disabled={!model || test.isPending} onClick={() => test.mutate()}>
            {test.isPending ? "Testing…" : "Test"}
          </button>
        </div>
      ) : (
        <p className="text-sm text-slate-600">
          Put your key in <code>{provider.variable}</code> in the <code>.env</code> file in the project folder
          (copy <code>.env.example</code>), then restart GradeForge.
        </p>
      )}
      {test.data && (
        <p className={clsx("mt-2 text-sm", test.data.ok ? "text-emerald-700" : "text-rose-700")}>{test.data.detail}</p>
      )}
      <div className="mt-2"><ErrorNote>{test.error?.message}</ErrorNote></div>
    </div>
  );
}

function ModelFacts({ model }) {
  const [papers, setPapers] = useState(30);
  const info = useQuery({ queryKey: ["cloud-model", model], queryFn: () => api.cloudModelInfo(model) });
  const estimate = useQuery({
    queryKey: ["cloud-estimate", model, papers], queryFn: () => api.cloudEstimate(model, papers), enabled: papers > 0,
  });
  const i = info.data;
  const e = estimate.data;
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-lg bg-slate-50 px-4 py-3 text-sm text-slate-600">
      {i?.known ? (
        <>
          <span>${i.input_per_million.toFixed(2)} / ${i.output_per_million.toFixed(2)} per million tokens (in / out)</span>
          <Badge tone={i.vision ? "green" : "gray"}>{i.vision ? "reads diagrams" : "no image input"}</Badge>
        </>
      ) : <span>Not in LiteLLM's price list: check the provider's pricing.</span>}
      <span className="flex items-center gap-1.5">
        Grading
        <input type="number" min={1} className="input w-20 py-1" value={papers} aria-label="Number of papers"
          onChange={(ev) => setPapers(Math.max(1, Number(ev.target.value) || 1))} />
        papers ≈ <b className="text-slate-800">{e?.usd != null ? `$${e.usd.toFixed(2)}` : "?"}</b>
      </span>
      {e && <span className="text-xs text-slate-400">{e.calls_per_paper} AI calls per paper, based on {e.basis}</span>}
    </div>
  );
}
