// Pieces shared by the Settings and Learning pages: tier badges, check lists, training plans.
import clsx from "clsx";
import { CheckCircle2, CircleHelp, MinusCircle, XCircle } from "lucide-react";
import { Badge } from "./components";

const TIER = {
  green: { tone: "green", dot: "bg-emerald-500" },
  yellow: { tone: "amber", dot: "bg-amber-500" },
  red: { tone: "red", dot: "bg-rose-500" },
  unknown: { tone: "gray", dot: "bg-slate-400" },
};

export function TierBadge({ tier }) {
  if (!tier) return null;
  const style = TIER[tier.level] ?? TIER.unknown;
  return (
    <Badge tone={style.tone} className="whitespace-nowrap">
      <span className={clsx("h-1.5 w-1.5 rounded-full", style.dot)} aria-hidden />
      {tier.label}{tier.where ? ` · ${tier.where}` : ""}
    </Badge>
  );
}

const STATUS = {
  pass: { icon: CheckCircle2, className: "text-emerald-600", label: "passed" },
  fail: { icon: XCircle, className: "text-rose-600", label: "failed" },
  skipped: { icon: MinusCircle, className: "text-slate-400", label: "skipped" },
  unknown: { icon: CircleHelp, className: "text-slate-400", label: "not checked here" },
};

export function StatusIcon({ status }) {
  const s = STATUS[status] ?? STATUS.unknown;
  return <s.icon size={16} className={clsx("mt-0.5 shrink-0", s.className)} aria-label={s.label} />;
}

// A list of {title, status, detail} rows: probe results, architecture checks, route options.
export function CheckList({ items }) {
  return (
    <ul className="space-y-2">
      {items.map((item) => (
        <li key={item.title} className="flex gap-2 text-sm">
          <StatusIcon status={item.status} />
          <div className="min-w-0">
            <span className="font-medium text-slate-800">{item.title}</span>
            {item.extra && <span className="ml-1.5 text-xs text-slate-400">{item.extra}</span>}
            {item.detail && <div className="text-xs text-slate-500">{item.detail}</div>}
          </div>
        </li>
      ))}
    </ul>
  );
}

// showBlocked: false where the tier reason above already says why nothing is available.
export function PlanOptions({ plan, showBlocked = true }) {
  if (!plan) return null;
  return (
    <div className="space-y-3">
      {plan.estimate && (
        <div className="text-sm text-slate-700">
          <span className="font-medium">{plan.estimate.label}</span> · needs ~{plan.estimate.vram_gb} GB of GPU memory
          <span className="text-xs text-slate-400"> ({plan.estimate.source})</span>
          {plan.estimate.note && <div className="text-xs text-slate-500">{plan.estimate.note}</div>}
        </div>
      )}
      {plan.options.length > 0 && (
        <CheckList items={plan.options.map((o) => ({
          title: o.label, status: o.available ? "pass" : "fail", detail: o.reason,
          extra: plan.recommended === o.target ? "recommended" : null,
        }))} />
      )}
      {showBlocked && plan.blocked_reason && <p className="text-sm text-slate-600">{plan.blocked_reason}</p>}
    </div>
  );
}
