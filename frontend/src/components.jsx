import clsx from "clsx";
import {
  AlertTriangle, BarChart3, CheckCircle2, FileUp, GraduationCap, Loader2, MessagesSquare, Settings, X,
} from "lucide-react";
import { lazy, Suspense, useRef, useState } from "react";
import { NavLink, Outlet } from "react-router";

const StrictnessCurve = lazy(() => import("./StrictnessCurve"));

const NAV = [
  { to: "/", label: "Exams", icon: GraduationCap, end: true },
  { to: "/disputes", label: "Dispute log", icon: MessagesSquare },
  { to: "/analytics", label: "Analytics", icon: BarChart3 },
  { to: "/settings", label: "Models & settings", icon: Settings },
];

export function Layout() {
  return (
    <div className="min-h-screen md:flex">
      {/* Narrow screens: a compact top bar instead of the sidebar. */}
      <header className="sticky top-0 z-40 flex items-center gap-1 overflow-x-auto border-b border-slate-200 bg-white px-3 py-2 md:hidden">
        <div className="mr-2 grid h-7 w-7 shrink-0 place-items-center rounded-lg bg-brand-600 text-white">
          <CheckCircle2 size={16} />
        </div>
        {NAV.map(({ to, label, icon: Icon, end }) => (
          <NavLink key={to} to={to} end={end} title={label}
            className={({ isActive }) => clsx("flex shrink-0 items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-sm",
              isActive ? "bg-brand-50 font-medium text-brand-700" : "text-slate-600")}>
            <Icon size={16} /> <span className="sr-only sm:not-sr-only">{label}</span>
          </NavLink>
        ))}
      </header>
      <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col border-r border-slate-200 bg-white md:flex">
        <div className="flex items-center gap-2.5 px-5 py-5">
          <div className="grid h-8 w-8 place-items-center rounded-lg bg-brand-600 text-white">
            <CheckCircle2 size={18} />
          </div>
          <div>
            <div className="font-semibold text-slate-900">GradeForge</div>
            <div className="text-xs text-slate-500">Local AI exam grading</div>
          </div>
        </div>
        <nav className="flex flex-col gap-0.5 px-3">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end}
              className={({ isActive }) => clsx("flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm",
                isActive ? "bg-brand-50 font-medium text-brand-700" : "text-slate-600 hover:bg-slate-100")}>
              <Icon size={17} /> {label}
            </NavLink>
          ))}
        </nav>
        <div className="mt-auto px-5 py-4 text-xs leading-relaxed text-slate-400">
          Student data stays on this computer.
        </div>
      </aside>
      <main className="min-w-0 flex-1 px-4 py-5 sm:px-8 sm:py-7">
        <Outlet />
      </main>
    </div>
  );
}

export function PageHeader({ title, subtitle, children }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-2xl font-semibold text-slate-900">{title}</h1>
        {subtitle && <p className="mt-1 text-sm text-slate-500">{subtitle}</p>}
      </div>
      {children && <div className="flex items-center gap-2">{children}</div>}
    </div>
  );
}

const BADGE = {
  green: "bg-emerald-50 text-emerald-700 ring-emerald-600/20",
  amber: "bg-amber-50 text-amber-800 ring-amber-600/20",
  red: "bg-rose-50 text-rose-700 ring-rose-600/20",
  blue: "bg-brand-50 text-brand-700 ring-brand-600/20",
  gray: "bg-slate-100 text-slate-600 ring-slate-500/20",
};

export function Badge({ tone = "gray", children, className }) {
  return (
    <span className={clsx("inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-medium ring-1 ring-inset",
      BADGE[tone], className)}>
      {children}
    </span>
  );
}

export function ConfidenceBadge({ value }) {
  if (value == null) return null;
  const tone = value >= 0.85 ? "green" : value >= 0.6 ? "amber" : "red";
  return <Badge tone={tone}>{Math.round(value * 100)}%</Badge>;
}

export function JobBar({ job, running }) {
  if (!job && !running) return null;
  if (job?.status === "failed") {
    return <ErrorNote>{job.error}</ErrorNote>;
  }
  if (job?.status === "done") return null;
  const pct = Math.round((job?.progress ?? 0) * 100);
  return (
    <div className="card flex items-center gap-3 px-4 py-3 text-sm">
      <Loader2 size={16} className="animate-spin text-brand-600" />
      <div className="flex-1">
        <div className="text-slate-700">{job?.message || "Starting..."}</div>
        <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-slate-100">
          <div className="h-full rounded-full bg-brand-500 transition-all" style={{ width: `${pct}%` }} />
        </div>
      </div>
      <span className="w-10 text-right tabular-nums text-slate-500">{pct}%</span>
    </div>
  );
}

export function ErrorNote({ children }) {
  if (!children) return null;
  return (
    <div className="flex items-start gap-2 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">
      <AlertTriangle size={16} className="mt-0.5 shrink-0" />
      <div className="min-w-0 break-words">{String(children)}</div>
    </div>
  );
}

export function FileDrop({ accept, onFile, disabled, hint }) {
  const input = useRef(null);
  const [over, setOver] = useState(false);
  return (
    <button type="button" disabled={disabled}
      onClick={() => input.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files[0]; if (f) onFile(f); }}
      className={clsx("flex w-full flex-col items-center gap-2 rounded-xl border-2 border-dashed px-6 py-8 text-sm transition-colors",
        over ? "border-brand-500 bg-brand-50" : "border-slate-300 bg-white hover:border-brand-400",
        disabled && "cursor-not-allowed opacity-60")}>
      <FileUp className="text-brand-600" />
      <span className="font-medium text-slate-700">Drop a file here or click to choose</span>
      {hint && <span className="text-xs text-slate-500">{hint}</span>}
      <input ref={input} type="file" accept={accept} hidden
        onChange={(e) => { const f = e.target.files[0]; if (f) onFile(f); e.target.value = ""; }} />
    </button>
  );
}

export function Modal({ title, onClose, children, wide }) {
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-slate-900/40 p-4" onClick={onClose}>
      <div className={clsx("card max-h-[90vh] w-full overflow-y-auto", wide ? "max-w-3xl" : "max-w-lg")}
        onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between border-b border-slate-100 px-5 py-3.5">
          <h2 className="font-semibold text-slate-900">{title}</h2>
          <button className="btn-ghost p-1.5" onClick={onClose} aria-label="Close"><X size={18} /></button>
        </div>
        <div className="p-5">{children}</div>
      </div>
    </div>
  );
}

export function Empty({ icon: Icon, title, children }) {
  return (
    <div className="card flex flex-col items-center px-6 py-12 text-center">
      {Icon && <Icon className="mb-3 text-slate-300" size={36} />}
      <div className="font-medium text-slate-700">{title}</div>
      {children && <div className="mt-1 max-w-md text-sm text-slate-500">{children}</div>}
    </div>
  );
}

// Same formula as src/grading/strictness_curve.py
export const strictnessExponent = (s) => 0.3 * (2.5 / 0.3) ** (s / 100);
export const strictnessLabel = (s) =>
  s < 12.5 ? "Very lenient" : s < 37.5 ? "Lenient" : s < 62.5 ? "Medium" : s < 87.5 ? "Strict" : "Very strict";

export function StrictnessSlider({ value, onChange, onCommit, compact }) {
  const exponent = strictnessExponent(value);
  const at70 = 100 * 0.7 ** exponent;
  return (
    <div className={clsx(!compact && "card p-4")}>
      <div className="flex items-baseline justify-between">
        <span className="label mb-0">Strictness</span>
        <span className="text-sm font-medium text-slate-700">{Math.round(value)} · {strictnessLabel(value)}</span>
      </div>
      <input type="range" min={0} max={100} value={value} className="mt-2 w-full accent-brand-600" aria-label="Strictness"
        onChange={(e) => onChange(Number(e.target.value))}
        onPointerUp={(e) => onCommit?.(Number(e.target.value))}
        onKeyUp={(e) => onCommit?.(Number(e.target.value))} />
      {!compact && (
        <>
          <div className="mt-2 h-28">
            <Suspense fallback={null}><StrictnessCurve exponent={exponent} /></Suspense>
          </div>
          <p className="mt-1 text-xs text-slate-500">
            An answer judged 70% correct earns <b className="text-slate-700">{at70.toFixed(0)}%</b> of the marks.
          </p>
        </>
      )}
    </div>
  );
}
