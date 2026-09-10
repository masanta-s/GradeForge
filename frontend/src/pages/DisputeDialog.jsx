import { useMutation } from "@tanstack/react-query";
import clsx from "clsx";
import { Bot, Send, User } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../api";
import { ErrorNote, Modal } from "../components";
import { useSettings } from "../hooks";

export default function DisputeDialog({ exam, question, validation, onClose }) {
  const { data: settings } = useSettings();
  const [conversation, setConversation] = useState(validation.conversation?.length ? validation.conversation : [
    { role: "ai", content: `This answer may be incorrect. Suggested: ${validation.ai_answer}. Reason: ${validation.justification}` },
  ]);
  const [conceded, setConceded] = useState(!!validation.ai_conceded);
  const [message, setMessage] = useState("");
  const [teacher, setTeacher] = useState(settings?.teacher_name ?? "");
  const [saveCorrection, setSaveCorrection] = useState(true);
  useEffect(() => { if (settings?.teacher_name) setTeacher((t) => t || settings.teacher_name); }, [settings]);

  const discuss = useMutation({
    mutationFn: () => api.discuss(exam.id, question.id, message),
    onSuccess: (r) => { setConversation(r.conversation); setConceded(r.ai_conceded); setMessage(""); },
  });
  const resolve = useMutation({
    mutationFn: (action) => api.resolve(exam.id, question.id, { teacher_name: teacher, action, save_correction: saveCorrection }),
    onSuccess: onClose,
  });

  return (
    <Modal title={`Question ${question.id}: answer-key dispute`} onClose={onClose} wide>
      <p className="mb-4 text-sm text-slate-700">{question.text}</p>
      <div className="mb-4 grid gap-3 sm:grid-cols-2">
        <div className="rounded-lg bg-slate-50 p-3">
          <div className="label">Your answer</div>
          <div className="text-sm text-slate-800">{validation.teacher_answer || "(empty)"}</div>
        </div>
        <div className="rounded-lg bg-brand-50 p-3">
          <div className="label">AI's suggestion</div>
          <div className="text-sm text-slate-800">{validation.ai_answer}</div>
        </div>
      </div>

      <div className="mb-3 max-h-64 space-y-3 overflow-y-auto">
        {conversation.map((turn, i) => (
          <div key={i} className={clsx("flex gap-2.5", turn.role === "teacher" && "flex-row-reverse")}>
            <div className={clsx("grid h-7 w-7 shrink-0 place-items-center rounded-full",
              turn.role === "ai" ? "bg-brand-100 text-brand-700" : "bg-slate-200 text-slate-600")}>
              {turn.role === "ai" ? <Bot size={15} /> : <User size={15} />}
            </div>
            <div className={clsx("max-w-[80%] rounded-xl px-3.5 py-2 text-sm",
              turn.role === "ai" ? "bg-slate-100 text-slate-800" : "bg-brand-600 text-white")}>
              {turn.content}
            </div>
          </div>
        ))}
      </div>
      {conceded && (
        <div className="mb-3 rounded-lg bg-emerald-50 px-3 py-2 text-sm text-emerald-800">
          The AI now agrees with you. Keep your answer below.
        </div>
      )}
      <form className="mb-5 flex gap-2" onSubmit={(e) => { e.preventDefault(); if (message.trim()) discuss.mutate(); }}>
        <input className="input" value={message} onChange={(e) => setMessage(e.target.value)}
          placeholder="Explain why your answer is right…" disabled={discuss.isPending} />
        <button className="btn-secondary" disabled={!message.trim() || discuss.isPending}><Send size={15} /> Reply</button>
      </form>

      <div className="space-y-3 border-t border-slate-100 pt-4">
        <div>
          <label className="label" htmlFor="teacher">Your name (recorded in the audit log)</label>
          <input id="teacher" className="input" value={teacher} onChange={(e) => setTeacher(e.target.value)} />
        </div>
        <label className="flex items-start gap-2 text-sm text-slate-700">
          <input type="checkbox" className="mt-0.5 accent-brand-600" checked={saveCorrection}
            onChange={(e) => setSaveCorrection(e.target.checked)} />
          <span>If I keep my answer, save it as a correction so future grading learns from it
            <span className="block text-xs text-slate-500">Unticked: a one-time override for this exam only.</span></span>
        </label>
        <ErrorNote>{resolve.error?.message || discuss.error?.message}</ErrorNote>
        <div className="flex flex-wrap justify-end gap-2">
          <button className="btn-secondary" disabled={!teacher.trim() || resolve.isPending}
            onClick={() => resolve.mutate("insist")}>Keep my answer</button>
          <button className="btn-primary" disabled={!teacher.trim() || resolve.isPending}
            onClick={() => resolve.mutate("accept")}>Use the AI's answer</button>
        </div>
      </div>
    </Modal>
  );
}
