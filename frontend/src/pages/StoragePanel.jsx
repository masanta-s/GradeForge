import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { HardDrive, Trash2 } from "lucide-react";
import { useState } from "react";
import { api, formatBytes } from "../api";
import { ErrorNote, Modal } from "../components";

const KIND = { llm_checkpoint: "Fine-tuned grading model", trocr_adapter: "Handwriting adapter", transient: "Temporary training file" };

// Disk used by fine-tuning output, and a clean-up the teacher reviews before anything is deleted.
export default function StoragePanel() {
  const queryClient = useQueryClient();
  const { data } = useQuery({ queryKey: ["storage"], queryFn: () => api.storage(2) });
  const [confirming, setConfirming] = useState(false);
  const clean = useMutation({
    mutationFn: () => api.cleanStorage(2),
    onSuccess: () => { setConfirming(false); queryClient.invalidateQueries({ queryKey: ["storage"] }); },
  });
  if (!data) return null;
  const d = data.disk;

  return (
    <div className="card p-5">
      <div className="mb-1 flex items-center gap-2 font-medium text-slate-900"><HardDrive size={17} /> Storage</div>
      <p className="mb-4 text-sm text-slate-500">
        Fine-tuning output on drive {d.drive}: {formatBytes(d.free_bytes)} free of {formatBytes(d.total_bytes)}.
        Your corrections are never cleaned up.
      </p>
      <dl className="grid gap-3 text-sm sm:grid-cols-3">
        {[["Fine-tuned grading models", d.llm_checkpoints_bytes], ["Handwriting adapters", d.trocr_adapters_bytes],
          ["Temporary training files", d.transient_bytes]].map(([label, bytes]) => (
          <div key={label} className="rounded-lg bg-slate-50 px-3 py-2">
            <dt className="text-xs text-slate-500">{label}</dt>
            <dd className="font-medium tabular-nums text-slate-800">{formatBytes(bytes)}</dd>
          </div>
        ))}
      </dl>
      <div className="mt-4 flex flex-wrap items-center gap-3">
        <button className="btn-secondary" disabled={!data.plan.length} onClick={() => setConfirming(true)}>
          <Trash2 size={15} /> Clean up{data.plan.length ? ` (frees ${formatBytes(data.reclaimable_bytes)})` : ""}
        </button>
        <span className="text-xs text-slate-500">
          {data.plan.length ? `${data.plan.length} old item${data.plan.length === 1 ? "" : "s"}` : "Nothing to clean up."}{" "}
          Keeps the newest {data.keep} versions of each model and anything in use.
        </span>
      </div>
      {clean.data?.errors?.length > 0 && <div className="mt-3"><ErrorNote>{clean.data.errors.join("; ")}</ErrorNote></div>}

      {confirming && (
        <Modal title="Delete these files?" onClose={() => setConfirming(false)}>
          <ul className="mb-4 max-h-64 space-y-1.5 overflow-y-auto text-sm">
            {data.plan.map((item) => (
              <li key={item.path} className="flex justify-between gap-3">
                <span className="min-w-0">
                  <span className="text-slate-800">{KIND[item.kind]}</span>{" "}
                  <span className="break-all text-xs text-slate-500">{item.ollama_name ?? item.path.split(/[\\/]/).slice(-2).join("/")}</span>
                  <span className="block text-xs text-slate-400">{item.reason}</span>
                </span>
                <span className="shrink-0 tabular-nums text-slate-600">{formatBytes(item.size_bytes)}</span>
              </li>
            ))}
          </ul>
          <p className="mb-4 text-xs text-slate-500">Old fine-tuned models are also removed from Ollama. This can't be undone.</p>
          <div className="flex justify-end gap-2">
            <button className="btn-ghost" onClick={() => setConfirming(false)}>Cancel</button>
            <button className="btn-primary bg-rose-600 hover:bg-rose-700" disabled={clean.isPending} onClick={() => clean.mutate()}>
              Delete {formatBytes(data.reclaimable_bytes)}
            </button>
          </div>
          <div className="mt-3"><ErrorNote>{clean.error?.message}</ErrorNote></div>
        </Modal>
      )}
    </div>
  );
}
