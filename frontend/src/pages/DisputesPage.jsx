import { useQuery } from "@tanstack/react-query";
import { Download, MessagesSquare, Search } from "lucide-react";
import { useState } from "react";
import { api } from "../api";
import { Badge, Empty, PageHeader } from "../components";

const DECISION = {
  accepted_ai: <Badge tone="blue">AI answer accepted</Badge>,
  save_correction: <Badge tone="green">Teacher kept · saved for learning</Badge>,
  one_time_override: <Badge tone="gray">Teacher kept · one-time</Badge>,
};

export default function DisputesPage() {
  const [filters, setFilters] = useState({ teacher: "", subject: "", text: "" });
  const [applied, setApplied] = useState(filters);
  const params = Object.fromEntries(Object.entries(applied).filter(([, v]) => v.trim()));
  const { data: records } = useQuery({ queryKey: ["disputes", params], queryFn: () => api.disputes(params) });
  const [open, setOpen] = useState(null);

  return (
    <div className="mx-auto max-w-5xl">
      <PageHeader title="Dispute log" subtitle="Every disagreement between a teacher and the AI about an answer key.">
        <a className="btn-secondary" href="/api/disputes/export.csv"><Download size={16} /> Export CSV</a>
      </PageHeader>

      <form className="card mb-4 grid gap-3 p-4 sm:grid-cols-[1fr_1fr_1.5fr_auto]"
        onSubmit={(e) => { e.preventDefault(); setApplied(filters); }}>
        {[["teacher", "Teacher"], ["subject", "Subject"], ["text", "Question or answer text"]].map(([key, label]) => (
          <input key={key} className="input" placeholder={label} value={filters[key]}
            onChange={(e) => setFilters({ ...filters, [key]: e.target.value })} />
        ))}
        <button className="btn-primary"><Search size={16} /> Search</button>
      </form>

      {records?.length ? (
        <div className="card divide-y divide-slate-100">
          {records.map((r) => (
            <div key={r.id} className="px-5 py-4">
              <button className="w-full text-left" onClick={() => setOpen(open === r.id ? null : r.id)}>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium text-slate-900">{r.exam_name} · Q{r.question_id}</span>
                  {DECISION[r.teacher_decision]}
                  <span className="ml-auto text-xs text-slate-500">
                    {r.teacher_name} · {new Date(r.timestamp).toLocaleString()}
                  </span>
                </div>
                <div className="mt-1 text-sm text-slate-600">{r.question_text}</div>
              </button>
              {open === r.id && (
                <div className="mt-3 space-y-2 text-sm">
                  <div><b>Teacher:</b> {r.teacher_answer}</div>
                  <div><b>AI:</b> {r.ai_answer} <span className="text-slate-500">({r.ai_justification})</span></div>
                  <div className="space-y-1 rounded-lg bg-slate-50 p-3">
                    {r.conversation.map((t, i) => (
                      <div key={i}><b className="capitalize">{t.role}:</b> {t.content}</div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <Empty icon={MessagesSquare} title="No disputes recorded">
          When you disagree with the AI's check of an answer key, the decision and conversation appear here.
        </Empty>
      )}
    </div>
  );
}
