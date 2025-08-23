export const BinaryPredictionDisplay: React.FC<{
  probOver: number | null;
  accuracy: number | null;
  loading: boolean;
  backend: string | null;
  count: number;
  mode?: "binary" | "roulette";
}> = ({ probOver, accuracy, loading, backend, count, mode = "binary" }) => {
  const rec = (() => {
    if (probOver == null) return "—";
    if (mode === "roulette") return probOver >= 0.5 ? "RED" : "BLACK";
    return probOver >= 0.5 ? "OVER" : "UNDER";
  })();
  return (
    <div className="mt-4 p-4 bg-slate-800/60 border border-slate-700 rounded-lg">
      <div className="flex items-baseline gap-4">
        <div className="text-4xl font-semibold tabular-nums bg-gradient-to-r from-teal-300 to-cyan-400 bg-clip-text text-transparent">
          {probOver == null ? "—" : (probOver * 100).toFixed(2) + "%"}
        </div>
        <div className="text-xs text-slate-400 uppercase tracking-wide">
          {probOver == null
            ? "No data"
            : mode === "roulette"
            ? "P(RED)"
            : "P(OVER)"}
        </div>
      </div>
      <div className="mt-2 text-sm font-mono">
        {rec === "—" ? "Waiting for samples" : rec}
      </div>
      <div className="mt-3 flex gap-6 text-[11px] text-slate-400">
        <div>
          <span className="block text-slate-500">Samples</span>
          <span className="font-mono text-slate-200">{count}</span>
        </div>
        <div>
          <span className="block text-slate-500">Accuracy</span>
          <span className="font-mono text-slate-200">
            {accuracy == null ? "—" : (accuracy * 100).toFixed(1) + "%"}
          </span>
        </div>
        <div>
          <span className="block text-slate-500">Status</span>
          <span
            className={`font-mono ${
              loading ? "text-amber-400" : "text-emerald-400"
            }`}
          >
            {loading ? "train" : "idle"}
          </span>
        </div>
        <div>
          <span className="block text-slate-500">Backend</span>
          <span className="font-mono text-slate-200">{backend || "—"}</span>
        </div>
      </div>
    </div>
  );
};
