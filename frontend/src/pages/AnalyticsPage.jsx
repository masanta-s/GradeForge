import { useQuery } from "@tanstack/react-query";
import { BarChart3 } from "lucide-react";
import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api";
import { Empty, PageHeader } from "../components";

const BUCKETS = ["0-20", "20-40", "40-60", "60-80", "80-100"];

export default function AnalyticsPage() {
  const { data: exams } = useQuery({ queryKey: ["exams"], queryFn: api.exams });
  const [examId, setExamId] = useState(null);
  const selected = examId ?? exams?.[0]?.id;
  const { data } = useQuery({ queryKey: ["analytics", selected], queryFn: () => api.analytics(selected), enabled: !!selected });

  const spread = BUCKETS.map((range, i) => ({
    range, students: data?.percentages.filter((p) => (i === 4 ? p >= 80 : p >= i * 20 && p < (i + 1) * 20)).length ?? 0,
  }));
  const questions = (data?.questions ?? []).map((q) => ({ ...q, pct: Math.round((100 * q.average) / q.max_marks) }));

  return (
    <div className="mx-auto max-w-5xl">
      <PageHeader title="Analytics" subtitle="How the class did, and which questions were hardest.">
        {exams?.length > 0 && (
          <select className="input w-72" value={selected ?? ""} onChange={(e) => setExamId(e.target.value)}>
            {exams.map((e) => <option key={e.id} value={e.id}>{e.name}</option>)}
          </select>
        )}
      </PageHeader>

      {!data?.students ? (
        <Empty icon={BarChart3} title="No graded sheets yet">Grade some answer sheets to see class results here.</Empty>
      ) : (
        <div className="space-y-5">
          <div className="grid gap-4 sm:grid-cols-3">
            <Stat label="Students graded" value={data.students} />
            <Stat label="Class average" value={`${Math.round(data.average_percentage)}%`} />
            <Stat label="Answers flagged for review" value={questions.reduce((s, q) => s + q.flagged, 0)} />
          </div>
          <div className="grid gap-5 lg:grid-cols-2">
            <div className="card p-5">
              <div className="mb-3 text-sm font-medium text-slate-700">Score distribution (%)</div>
              <div className="h-56">
                <ResponsiveContainer>
                  <BarChart data={spread} margin={{ left: -20 }}>
                    <CartesianGrid vertical={false} stroke="#f1f5f9" />
                    <XAxis dataKey="range" tick={{ fontSize: 12 }} />
                    <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
                    <Tooltip cursor={{ fill: "#f8fafc" }} />
                    <Bar dataKey="students" fill="#6366f1" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="card p-5">
              <div className="mb-3 text-sm font-medium text-slate-700">Average score per question (%)</div>
              <div className="h-56">
                <ResponsiveContainer>
                  <BarChart data={questions} margin={{ left: -20 }}>
                    <CartesianGrid vertical={false} stroke="#f1f5f9" />
                    <XAxis dataKey="question_id" tickFormatter={(v) => `Q${v}`} tick={{ fontSize: 12 }} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} />
                    <Tooltip formatter={(v) => `${v}%`} labelFormatter={(v) => `Question ${v}`} cursor={{ fill: "#f8fafc" }} />
                    <Bar dataKey="pct" radius={[4, 4, 0, 0]} isAnimationActive={false}>
                      {questions.map((q) => <Cell key={q.question_id} fill={q.pct < 40 ? "#f43f5e" : q.pct < 70 ? "#f59e0b" : "#10b981"} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
          <div className="card overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
                <tr>{["Question", "Type", "Average", "Full marks", "Zero", "Not attempted", "Flagged", "Changed by teacher"]
                  .map((h) => <th key={h} className="px-4 py-2.5 font-medium">{h}</th>)}</tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {questions.map((q) => (
                  <tr key={q.question_id}>
                    <td className="px-4 py-2.5 font-medium">Q{q.question_id}</td>
                    <td className="px-4 py-2.5 text-slate-500">{q.qtype}</td>
                    <td className="px-4 py-2.5 tabular-nums">{q.average.toFixed(1)}/{q.max_marks} ({q.pct}%)</td>
                    {["full_marks", "zero", "not_attempted", "flagged", "teacher_changed"].map((k) => (
                      <td key={k} className="px-4 py-2.5 tabular-nums">{q[k]}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="card p-5">
      <div className="label">{label}</div>
      <div className="text-2xl font-semibold text-slate-900">{value}</div>
    </div>
  );
}
