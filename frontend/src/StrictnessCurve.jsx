// Loaded lazily: recharts is large and only needed where the curve is shown.
import { Area, AreaChart, ReferenceDot, ResponsiveContainer, XAxis, YAxis } from "recharts";

export default function StrictnessCurve({ exponent }) {
  const curve = Array.from({ length: 21 }, (_, i) => ({ q: i * 5, marks: 100 * (i / 20) ** exponent }));
  return (
    <ResponsiveContainer>
      <AreaChart data={curve} margin={{ top: 6, right: 6, bottom: 0, left: -18 }}>
        <XAxis dataKey="q" tick={{ fontSize: 10 }} tickFormatter={(v) => `${v}%`} />
        <YAxis tick={{ fontSize: 10 }} domain={[0, 100]} />
        <Area dataKey="marks" type="monotone" stroke="#4f46e5" fill="#e0e7ff" isAnimationActive={false} />
        <ReferenceDot x={70} y={100 * 0.7 ** exponent} r={4} fill="#4f46e5" stroke="white" />
      </AreaChart>
    </ResponsiveContainer>
  );
}
