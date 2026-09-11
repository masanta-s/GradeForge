import { useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { Download, FlaskConical, TrendingUp } from "lucide-react";
import { useState } from "react";
import { api } from "../api";
import { ErrorNote, JobBar } from "../components";
import { useJob } from "../hooks";

const pct = (x) => `${Math.round(x * 100)}%`;

// "Is it getting better?" Two honest measures: marks the teacher kept on approved sheets, week by
// week, and a fixed held-out test (answers never used for learning) re-marked by each model setup.
export default function ProgressPanel() {
  const queryClient = useQueryClient();
  const { data } = useQuery({ queryKey: ["progress"], queryFn: api.progress });
  const job = useJob((j) => j.status === "done" && queryClient.invalidateQueries({ queryKey: ["progress"] }));
  const [error, setError] = useState(null);
  if (!data) return null;

  const learned = data.benchmarks.filter((b) => b.setup === "learned");
  const firstTest = learned[0], lastTest = learned.at(-1);
  const { first_week: firstWeek, latest_week: lastWeek } = data;
  const minutes = Math.max(1, Math.round((2 * data.holdout.measured_per_run * 3) / 60));
  const measure = async () => {
    setError(null);
    try { job.start((await api.benchmark()).job_id); } catch (e) { setError(e.message); }
  };

  return (
    <div className="card mb-6 p-5">
      <div className="mb-1 flex items-center gap-2 font-medium text-slate-900"><TrendingUp size={17} /> How closely the AI marks like you</div>
      <p className="mb-4 text-sm text-slate-500">
        {data.checked.total} answer{data.checked.total === 1 ? "" : "s"} checked by you ({data.checked.changed} changed,{" "}
        {data.checked.confirmed} confirmed on {data.approved_sheets} approved sheet{data.approved_sheets === 1 ? "" : "s"}).
        One in five is kept aside as a test the AI never learns from.
      </p>

      <div className="grid gap-3 sm:grid-cols-3">
        <Stat label="Marks you kept, latest week" value={lastWeek ? `${lastWeek.accepted_pct}%` : "–"}
          change={firstWeek && lastWeek && firstWeek !== lastWeek ? lastWeek.accepted_pct - firstWeek.accepted_pct : null}
          unit=" pts since the first week" empty="Approve checked sheets to start" />
        <Stat label="Held-out test: within half a mark" value={lastTest ? pct(lastTest.close) : "–"}
          change={learned.length > 1 ? 100 * (lastTest.close - firstTest.close) : null} unit=" pts since the first test"
          empty="Run a test below" />
        <Stat label="Held-out test: average difference" value={lastTest ? `${lastTest.mae.toFixed(2)} marks` : "–"}
          change={learned.length > 1 ? lastTest.mae - firstTest.mae : null} unit=" marks since the first test"
          lowerIsBetter decimals={2} empty={lastTest ? "" : "Run a test below"} />
      </div>

      {data.weekly.length > 0 && (
        <div className="mt-5">
          <div className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">Marks kept as-is, by week</div>
          <div className="space-y-1.5">
            {data.weekly.map((w) => (
              <div key={w.week} className="flex items-center gap-3 text-sm">
                <span className="w-20 shrink-0 tabular-nums text-slate-500">{w.week}</span>
                <div className="h-2.5 flex-1 rounded-full bg-slate-100">
                  <div className="h-full rounded-full bg-emerald-500" style={{ width: `${w.accepted_pct}%` }} />
                </div>
                <span className="w-44 shrink-0 text-right tabular-nums text-slate-600">
                  {w.accepted_pct}% · {w.sheets} sheet{w.sheets === 1 ? "" : "s"} · ±{w.avg_change}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {data.benchmarks.length > 0 && (
        <div className="mt-5 overflow-x-auto">
          <div className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">Held-out tests</div>
          <table className="w-full text-sm">
            <thead className="text-left text-xs text-slate-500">
              <tr>{["When", "Setup", "Answers", "Exact", "Within ½ mark", "Avg difference", "Leans"]
                .map((h) => <th key={h} className="py-1.5 pr-4 font-medium">{h}</th>)}</tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {data.benchmarks.slice().reverse().map((b) => (
                <tr key={b.id}>
                  <td className="py-1.5 pr-4 text-slate-500">{new Date(b.created_at).toLocaleDateString()}</td>
                  <td className="py-1.5 pr-4">{b.label}</td>
                  <td className="py-1.5 pr-4 tabular-nums">{b.n}</td>
                  <td className="py-1.5 pr-4 tabular-nums">{pct(b.exact)}</td>
                  <td className="py-1.5 pr-4 tabular-nums">{pct(b.close)}</td>
                  <td className="py-1.5 pr-4 tabular-nums">{b.mae.toFixed(2)} marks</td>
                  <td className="py-1.5 pr-4 text-slate-600">
                    {Math.abs(b.bias) < 0.1 ? "balanced" : b.bias > 0 ? `+${b.bias.toFixed(2)} generous` : `${b.bias.toFixed(2)} strict`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="mt-4 flex flex-wrap items-center gap-3">
        <button className="btn-secondary" disabled={job.running || !data.holdout.total} onClick={measure}>
          <FlaskConical size={15} /> Run the held-out test (~{minutes} min)
        </button>
        <a className="btn-ghost" href="/api/learning/progress.csv"><Download size={15} /> Download as CSV</a>
        <span className="text-xs text-slate-500">
          {data.holdout.total
            ? `${data.holdout.measured_per_run} of ${data.holdout.total} held-out answers are re-marked, by the model alone and with what it learned from you.`
            : "No held-out answers yet: they build up as you check more marks."}
        </span>
      </div>
      <div className="mt-3"><JobBar {...job} /></div>
      <ErrorNote>{error}</ErrorNote>
    </div>
  );
}

function Stat({ label, value, change, unit, lowerIsBetter, decimals = 1, empty }) {
  const better = change != null && (lowerIsBetter ? change < 0 : change > 0);
  return (
    <div className="rounded-lg bg-slate-50 px-4 py-3">
      <div className="text-xs text-slate-500">{label}</div>
      <div className="text-2xl font-semibold tabular-nums text-slate-900">{value}</div>
      {change != null ? (
        <div className={clsx("text-xs", change === 0 ? "text-slate-500" : better ? "text-emerald-700" : "text-rose-700")}>
          {change > 0 ? "+" : ""}{change.toFixed(decimals)}{unit}
        </div>
      ) : value === "–" && <div className="text-xs text-slate-400">{empty}</div>}
    </div>
  );
}
