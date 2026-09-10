import { useQueryClient } from "@tanstack/react-query";
import { ChevronRight, ClipboardCheck, Lock, Play, Users } from "lucide-react";
import { useState } from "react";
import { Link, useOutletContext } from "react-router";
import { api } from "../api";
import { Badge, Empty, ErrorNote, FileDrop, JobBar } from "../components";
import { useJob, useSettings } from "../hooks";

const STATUS = {
  uploaded: <Badge>Reading…</Badge>,
  read: <Badge tone="blue">Ready to grade</Badge>,
  graded: <Badge tone="green">Graded</Badge>,
};

export default function StudentsPage() {
  const { exam } = useOutletContext();
  const { data: settings } = useSettings();
  const queryClient = useQueryClient();
  const refresh = () => queryClient.invalidateQueries({ queryKey: ["exam", exam.id] });
  const job = useJob((j) => { if (j.status === "done") refresh(); });
  const [student, setStudent] = useState("");
  const [error, setError] = useState(null);
  const keyReady = exam.answer_key?.finalized;

  const upload = async (file) => {
    setError(null);
    try {
      const r = await api.uploadSheet(exam.id, student.trim(), file);
      setStudent("");
      refresh();
      job.start(r.job_id);
    } catch (e) { setError(e.message); }
  };
  const grade = async (sheetId) => {
    setError(null);
    try { job.start((await api.grade(exam.id, sheetId, settings?.strictness ?? 50)).job_id); }
    catch (e) { setError(e.message); }
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
      <div className="space-y-4">
        {!keyReady && (
          <div className="flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
            <Lock size={16} /> Finalize the answer key before grading. You can already upload sheets.
          </div>
        )}
        <JobBar {...job} />
        <ErrorNote>{error}</ErrorNote>
        {exam.sheets.length ? (
          <div className="card divide-y divide-slate-100">
            {exam.sheets.map((s) => (
              <div key={s.id} className="flex items-center gap-4 px-5 py-3.5">
                <div className="min-w-0 flex-1">
                  <div className="truncate font-medium text-slate-900">{s.student}</div>
                  <div className="mt-0.5 flex items-center gap-2 text-xs">
                    {STATUS[s.status]}
                    {s.review_count > 0 && <Badge tone="amber">{s.review_count} to check</Badge>}
                  </div>
                </div>
                {s.total != null && (
                  <div className="text-right">
                    <div className="text-lg font-semibold tabular-nums text-slate-900">{s.total}/{s.max_total}</div>
                    <div className="text-xs text-slate-500">{Math.round((100 * s.total) / s.max_total)}%</div>
                  </div>
                )}
                {s.status !== "uploaded" && keyReady && (
                  <button className="btn-secondary py-1.5" disabled={job.running} onClick={() => grade(s.id)}>
                    <Play size={14} /> {s.status === "graded" ? "Re-grade" : "Grade"}
                  </button>
                )}
                {s.status !== "uploaded" && (
                  <Link to={`/exams/${exam.id}/sheets/${s.id}`} className="btn-ghost py-1.5">
                    Review <ChevronRight size={15} />
                  </Link>
                )}
              </div>
            ))}
          </div>
        ) : (
          <Empty icon={Users} title="No answer sheets yet">Add each student's sheet on the right.</Empty>
        )}
      </div>

      <div className="card h-fit space-y-3 p-5">
        <div className="flex items-center gap-2 font-medium text-slate-900"><ClipboardCheck size={17} /> Add answer sheet</div>
        <div>
          <label className="label" htmlFor="student">Student name or roll no.</label>
          <input id="student" className="input" value={student} onChange={(e) => setStudent(e.target.value)} />
        </div>
        <FileDrop accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff,.webp" disabled={!student.trim() || job.running}
          hint={student.trim() ? "Photo, scan or PDF" : "Enter the student first"} onFile={upload} />
      </div>
    </div>
  );
}
