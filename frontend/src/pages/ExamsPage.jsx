import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { BookOpen, ChevronRight, Plus } from "lucide-react";
import { useState } from "react";
import { Link, useNavigate } from "react-router";
import { api } from "../api";
import { Empty, ErrorNote, Modal, PageHeader } from "../components";

export default function ExamsPage() {
  const { data: exams, isLoading } = useQuery({ queryKey: ["exams"], queryFn: api.exams });
  const [creating, setCreating] = useState(false);

  return (
    <div className="mx-auto max-w-5xl">
      <PageHeader title="Exams" subtitle="Upload a question paper, prepare the answer key, then grade answer sheets.">
        <button className="btn-primary" onClick={() => setCreating(true)}><Plus size={16} /> New exam</button>
      </PageHeader>

      {isLoading ? null : exams?.length ? (
        <div className="card divide-y divide-slate-100">
          {exams.map((exam) => (
            <Link key={exam.id} to={`/exams/${exam.id}`}
              className="flex items-center gap-4 px-5 py-4 hover:bg-slate-50">
              <div className="grid h-10 w-10 place-items-center rounded-lg bg-brand-50 text-brand-600">
                <BookOpen size={18} />
              </div>
              <div className="min-w-0 flex-1">
                <div className="truncate font-medium text-slate-900">{exam.name}</div>
                <div className="text-sm text-slate-500">
                  {exam.subject} · {exam.questions.length ? `${exam.questions.length} questions` : "no paper yet"} ·{" "}
                  created {new Date(exam.created_at).toLocaleDateString()}
                </div>
              </div>
              <ChevronRight className="text-slate-400" size={18} />
            </Link>
          ))}
        </div>
      ) : (
        <Empty icon={BookOpen} title="No exams yet">
          Create an exam, upload its question paper, and GradeForge will draft the answer key for you to review.
        </Empty>
      )}

      {creating && <NewExamDialog onClose={() => setCreating(false)} />}
    </div>
  );
}

function NewExamDialog({ onClose }) {
  const [name, setName] = useState("");
  const [subject, setSubject] = useState("");
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const create = useMutation({
    mutationFn: () => api.createExam(name, subject),
    onSuccess: (exam) => {
      queryClient.invalidateQueries({ queryKey: ["exams"] });
      navigate(`/exams/${exam.id}`);
    },
  });

  return (
    <Modal title="New exam" onClose={onClose}>
      <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); create.mutate(); }}>
        <div>
          <label className="label" htmlFor="exam-name">Exam name</label>
          <input id="exam-name" className="input" autoFocus value={name} onChange={(e) => setName(e.target.value)}
            placeholder="Class 10 Biology - Unit Test 2" />
        </div>
        <div>
          <label className="label" htmlFor="exam-subject">Subject</label>
          <input id="exam-subject" className="input" value={subject} onChange={(e) => setSubject(e.target.value)}
            placeholder="Biology" />
        </div>
        <ErrorNote>{create.error?.message}</ErrorNote>
        <div className="flex justify-end gap-2">
          <button type="button" className="btn-secondary" onClick={onClose}>Cancel</button>
          <button className="btn-primary" disabled={!name.trim() || !subject.trim() || create.isPending}>
            Create exam
          </button>
        </div>
      </form>
    </Modal>
  );
}
