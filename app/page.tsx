"use client";
// HistoryList replaced by inline chips for binary outcomes
import { BinaryButtons } from "./components/BinaryButtons";
import { BinaryPredictionDisplay } from "./components/BinaryPredictionDisplay";
import { ControlPanel } from "./components/ControlPanel";
import { useBinaryModel } from "./hooks/useBinaryModel";

function StatsBar({
  records,
  metrics,
}: {
  records: { correct: boolean | null }[];
  metrics: {
    longestWin: number;
    longestLose: number;
    overPredSuccess: number;
    overPredTotal: number;
    sumWin: number;
    countWin: number;
    sumLose: number;
    countLose: number;
    sumActualRuns: number;
    countActualRuns: number;
    sumOverRuns: number;
    countOverRuns: number;
    sumBelowRuns: number;
    countBelowRuns: number;
  };
}) {
  const decided = records.filter((r) => r.correct != null);
  const wins = decided.filter((r) => r.correct).length;
  const losses = decided.filter((r) => r.correct === false).length;
  const winPct = decided.length ? (wins / decided.length) * 100 : 0;
  const overHitPct = metrics.overPredTotal
    ? (metrics.overPredSuccess / metrics.overPredTotal) * 100
    : 0;
  const avgWinStreak = metrics.countWin
    ? metrics.sumWin / metrics.countWin
    : metrics.longestWin || 0;
  const avgLoseStreak = metrics.countLose
    ? metrics.sumLose / metrics.countLose
    : metrics.longestLose || 0;
  const avgActualRun = metrics.countActualRuns
    ? metrics.sumActualRuns / metrics.countActualRuns
    : 0;
  const avgOverRun = metrics.countOverRuns
    ? metrics.sumOverRuns / metrics.countOverRuns
    : 0;
  const avgBelowRun = metrics.countBelowRuns
    ? metrics.sumBelowRuns / metrics.countBelowRuns
    : 0;
  return (
    <div className="mt-2 grid grid-cols-3 gap-2 text-[11px]">
      <div className="rounded border border-emerald-600/40 bg-emerald-700/10 px-3 py-2 flex flex-col items-start">
        <span className="text-[9px] uppercase tracking-wide text-emerald-300">
          Wins
        </span>
        <span className="text-emerald-200 font-semibold tabular-nums">
          {wins}
        </span>
      </div>
      <div className="rounded border border-rose-600/40 bg-rose-700/10 px-3 py-2 flex flex-col items-start">
        <span className="text-[9px] uppercase tracking-wide text-rose-300">
          Losses
        </span>
        <span className="text-rose-200 font-semibold tabular-nums">
          {losses}
        </span>
      </div>
      <div className="rounded border border-sky-600/40 bg-sky-700/10 px-3 py-2 flex flex-col items-start">
        <span className="text-[9px] uppercase tracking-wide text-sky-300">
          Win %
        </span>
        <span className="text-sky-200 font-semibold tabular-nums">
          {winPct.toFixed(1)}%
        </span>
      </div>
      <div className="rounded border border-teal-600/40 bg-teal-700/10 px-3 py-2 flex flex-col items-start">
        <span className="text-[9px] uppercase tracking-wide text-teal-300">
          Over Prediction Hit %
        </span>
        <span className="text-teal-200 font-semibold tabular-nums">
          {overHitPct.toFixed(1)}% ({metrics.overPredSuccess}/
          {metrics.overPredTotal})
        </span>
      </div>
      <div className="rounded border border-indigo-600/40 bg-indigo-700/10 px-3 py-2 flex flex-col items-start">
        <span className="text-[9px] uppercase tracking-wide text-indigo-300">
          Longest Win Streak
        </span>
        <span className="text-indigo-200 font-semibold tabular-nums">
          {metrics.longestWin}
        </span>
      </div>
      <div className="rounded border border-fuchsia-600/40 bg-fuchsia-700/10 px-3 py-2 flex flex-col items-start">
        <span className="text-[9px] uppercase tracking-wide text-fuchsia-300">
          Longest Lose Streak
        </span>
        <span className="text-fuchsia-200 font-semibold tabular-nums">
          {metrics.longestLose}
        </span>
      </div>
      <div className="rounded border border-zinc-600/40 bg-zinc-700/10 px-3 py-2 flex flex-col items-start">
        <span className="text-[9px] uppercase tracking-wide text-zinc-300">
          Decisions
        </span>
        <span className="text-zinc-200 font-semibold tabular-nums">
          {decided.length}
        </span>
      </div>
      <div className="rounded border border-lime-600/40 bg-lime-700/10 px-3 py-2 flex flex-col items-start">
        <span className="text-[9px] uppercase tracking-wide text-lime-300">
          Avg Win Streak
        </span>
        <span className="text-lime-200 font-semibold tabular-nums">
          {avgWinStreak.toFixed(2)}
        </span>
      </div>
      <div className="rounded border border-amber-600/40 bg-amber-700/10 px-3 py-2 flex flex-col items-start">
        <span className="text-[9px] uppercase tracking-wide text-amber-300">
          Avg Lose Streak
        </span>
        <span className="text-amber-200 font-semibold tabular-nums">
          {avgLoseStreak.toFixed(2)}
        </span>
      </div>
      <div className="rounded border border-purple-600/40 bg-purple-700/10 px-3 py-2 flex flex-col items-start">
        <span className="text-[9px] uppercase tracking-wide text-purple-300">
          Avg Actual Run
        </span>
        <span className="text-purple-200 font-semibold tabular-nums">
          {avgActualRun.toFixed(2)}
        </span>
      </div>
      <div className="rounded border border-emerald-600/40 bg-emerald-700/10 px-3 py-2 flex flex-col items-start">
        <span className="text-[9px] uppercase tracking-wide text-emerald-300">
          Avg OVER Run
        </span>
        <span className="text-emerald-200 font-semibold tabular-nums">
          {avgOverRun.toFixed(2)}
        </span>
      </div>
      <div className="rounded border border-cyan-600/40 bg-cyan-700/10 px-3 py-2 flex flex-col items-start">
        <span className="text-[9px] uppercase tracking-wide text-cyan-300">
          Avg BELOW Run
        </span>
        <span className="text-cyan-200 font-semibold tabular-nums">
          {avgBelowRun.toFixed(2)}
        </span>
      </div>
    </div>
  );
}

export default function Home() {
  const {
    addObservation,
    history,
    probOver,
    loading,
    backend,
    accuracy,
    controls,
    setControls,
    resetModel,
    records,
    rlWeights,
    biasInfo,
    metrics,
    suggestedThreshold,
    probParts,
  } = useBinaryModel();
  const diagnostics = (() => {
    if (!probParts) return null;
    const sources = [
      {
        key: "model",
        label: "Model",
        p: probParts.model,
        mw: controls.wModel,
        rw: rlWeights.model,
      },
      {
        key: "markov",
        label: "Markov",
        p: probParts.markov,
        mw: controls.wMarkov,
        rw: rlWeights.markov,
      },
      {
        key: "streak",
        label: "Streak",
        p: probParts.streak,
        mw: controls.wStreak,
        rw: rlWeights.streak,
      },
      {
        key: "pattern",
        label: "Pattern",
        p: probParts.pattern,
        mw: controls.wPattern,
        rw: rlWeights.pattern,
      },
      {
        key: "ewma",
        label: "EWMA",
        p: probParts.ewma,
        mw: controls.wEwma,
        rw: rlWeights.ewma,
      },
      {
        key: "bayes",
        label: "Bayes",
        p: probParts.bayes,
        mw: controls.wBayes,
        rw: rlWeights.bayes ?? 1,
      },
      {
        key: "entropy",
        label: "Entropy",
        p: probParts.entropyMod,
        mw: controls.wEntropy,
        rw: rlWeights.entropyMod ?? 1,
      },
    ];
    const rawCombined = sources.map((s) => ({ ...s, cw: s.mw * s.rw }));
    const sum = rawCombined.reduce((a, b) => a + b.cw, 0) || 1;
    const norm = rawCombined.map((s) => ({ ...s, wn: s.cw / sum }));
    return norm;
  })();
  return (
    <div className="min-h-screen w-full bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 text-slate-200 px-5 py-10">
      <div className="max-w-xl mx-auto">
        <header className="mb-10">
          <h1 className="text-2xl font-semibold tracking-tight bg-gradient-to-r from-teal-300 via-cyan-300 to-sky-400 text-transparent bg-clip-text">
            Realtime Forecast
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Incremental sequence model predicting next numeric input.
            GPU-enabled (WebGPU/WebGL fallback).
          </p>
        </header>
        <section className="space-y-6">
          <div className="sticky top-0 z-20 pt-2 pb-3 bg-gradient-to-b from-slate-950/95 via-slate-950/80 to-transparent backdrop-blur supports-[backdrop-filter]:backdrop-blur-sm">
            <BinaryButtons
              onBelow={() => addObservation(0)}
              onOver={() => addObservation(1)}
              disabled={loading}
              probOver={probOver}
              threshold={controls.threshold}
            />
          </div>
          <BinaryPredictionDisplay
            probOver={probOver}
            accuracy={accuracy}
            loading={loading}
            backend={backend}
            count={history.length}
          />
          <StatsBar records={records} metrics={metrics} />
          {diagnostics && (
            <div className="mt-4 text-left max-w-xl mx-auto">
              <h3 className="text-xs font-semibold tracking-wide text-slate-300 mb-2">
                Probability Components
              </h3>
              <div className="grid grid-cols-7 gap-2 text-[10px]">
                {diagnostics.map((d) => (
                  <div
                    key={d.key}
                    className="p-2 rounded border border-slate-700 bg-slate-900/40 flex flex-col gap-0.5"
                  >
                    <span className="text-slate-400 text-[9px] uppercase tracking-wide">
                      {d.label}
                    </span>
                    <span className="text-teal-300 tabular-nums">
                      p {(d.p * 100).toFixed(1)}%
                    </span>
                    <span className="text-cyan-300/90 tabular-nums">
                      w {d.wn.toFixed(2)}
                    </span>
                    <span className="text-[9px] text-slate-500">
                      m{d.mw.toFixed(2)} · r{d.rw.toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
              <p className="mt-2 text-[9px] text-slate-500 leading-snug">
                Weights shown are normalized blend fractions (m=manual, r=RL).
                Final probability also subject to anti-streak & inversion
                heuristics.
              </p>
            </div>
          )}
          <div className="mt-4 p-3 border border-slate-700 rounded bg-slate-900/40 h-48 overflow-y-auto text-[10px] font-mono flex flex-col gap-1">
            {records
              .slice()
              .reverse()
              .map((r, i) => {
                const res =
                  r.correct == null ? "—" : r.correct ? "WIN" : "LOSS";
                const color =
                  r.correct == null
                    ? "bg-slate-600/30 border-slate-600/50 text-slate-300"
                    : r.correct
                    ? "bg-emerald-600/30 border-emerald-500/40 text-emerald-200"
                    : "bg-rose-700/30 border-rose-600/40 text-rose-200";
                return (
                  <div
                    key={i}
                    className={`flex items-center gap-2 px-2 py-1 rounded border ${color}`}
                  >
                    <span
                      className={`text-[9px] px-1.5 py-0.5 rounded ${
                        r.label
                          ? "bg-teal-500/30 text-teal-200"
                          : "bg-cyan-500/20 text-cyan-200"
                      }`}
                    >
                      {r.label ? "OVER" : "BELOW"}
                    </span>
                    <span className="text-slate-400">
                      pred:{" "}
                      {r.predLabel == null
                        ? "n/a"
                        : r.predLabel
                        ? "OVER"
                        : "BELOW"}
                    </span>
                    <span className="text-slate-400">
                      p={r.prob == null ? "n/a" : (r.prob * 100).toFixed(1)}%
                    </span>
                    <span className="ml-auto font-semibold tracking-wide">
                      {res}
                    </span>
                  </div>
                );
              })}
            {records.length === 0 && (
              <span className="text-slate-600">No observations yet.</span>
            )}
          </div>
          <ControlPanel
            controls={controls}
            setControls={setControls}
            rlWeights={rlWeights}
            resetModel={resetModel}
            loading={loading}
          />
        </section>
        <footer className="mt-12 text-center text-[10px] text-slate-600 space-y-1">
          <div>
            Binary model: click actual outcome each round (Below 2 or Over 2).
            Model adapts online using streak & pattern features.
          </div>
        </footer>
      </div>
    </div>
  );
}
