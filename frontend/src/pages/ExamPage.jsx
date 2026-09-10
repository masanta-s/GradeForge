import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { ArrowLeft } from "lucide-react";
import { Link, NavLink, Outlet, useParams } from "react-router";
import { api } from "../api";
import { Badge, ErrorNote } from "../components";

export default function ExamPage() {
  const { examId } = useParams();
  const { data: exam, error } = useQuery({
    queryKey: ["exam", examId],
    queryFn: () => api.exam(examId),
    // keep refreshing while sheets are still being read in the background (e.g. the demo)
    refetchInterval: (q) => (q.state.data?.sheets.some((s) => s.status === "uploaded") ? 2000 : false),
  });
  if (error) return <ErrorNote>{error.message}</ErrorNote>;
  if (!exam) return null;

  const key = exam.answer_key;
  const tabs = [
    { to: "", label: "Paper & answer key", end: true },
    { to: "students", label: `Students (${exam.sheets.length})` },
  ];
  return (
    <div className="mx-auto max-w-6xl">
      <Link to="/" className="mb-3 inline-flex items-center gap-1 text-sm text-slate-500 hover:text-slate-700">
        <ArrowLeft size={15} /> All exams
      </Link>
      <div className="mb-5 flex flex-wrap items-center gap-3">
        <h1 className="text-2xl font-semibold text-slate-900">{exam.name}</h1>
        <Badge>{exam.subject}</Badge>
        {key?.finalized ? <Badge tone="green">Answer key final</Badge>
          : key ? <Badge tone="amber">Answer key draft</Badge> : <Badge>No answer key</Badge>}
      </div>
      <div className="mb-6 flex gap-1 border-b border-slate-200">
        {tabs.map((t) => (
          <NavLink key={t.label} to={t.to} end={t.end}
            className={({ isActive }) => clsx("-mb-px border-b-2 px-4 py-2.5 text-sm",
              isActive ? "border-brand-600 font-medium text-brand-700" : "border-transparent text-slate-500 hover:text-slate-700")}>
            {t.label}
          </NavLink>
        ))}
      </div>
      <Outlet context={{ exam }} />
    </div>
  );
}
