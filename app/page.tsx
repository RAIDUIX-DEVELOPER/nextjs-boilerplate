"use client";
// HistoryList replaced by inline chips for binary outcomes
import { BinaryButtons } from "./components/BinaryButtons";
import { BinaryPredictionDisplay } from "./components/BinaryPredictionDisplay";
import { ControlPanel } from "./components/ControlPanel";
import { useBinaryModel } from "./hooks/useBinaryModel";

function DozensPanel({
  probs,
  samples,
  pred,
}: {
  probs: [number, number, number];
  samples: number;
  pred: number | null;
}) {
  return (
    <div className="grid grid-cols-3 gap-3 text-[11px]">
      {probs.map((p, i) => {
        const active = pred === i;
        return (
          <div
            key={i}
            className={`p-3 rounded border flex flex-col gap-1 ${
              active
                ? "border-teal-500/70 bg-teal-600/10"
                : "border-slate-700 bg-slate-900/50"
            }`}
          >
            <span className="text-slate-400 uppercase tracking-wide text-[9px]">
              {i === 0 ? "1st 12" : i === 1 ? "2nd 12" : "3rd 12"}
            </span>
            <span className="text-teal-300 font-mono">
              {(p * 100).toFixed(1)}%
            </span>
          </div>
        );
      })}
      <div className="col-span-3 text-[9px] text-slate-500">
        Samples: {samples}
      </div>
    </div>
  );
}

function DozensStats({
  records,
  accuracy,
  samples,
  history,
}: {
  records: {
    label?: number;
    predLabel?: number | null;
    correct: boolean | null;
  }[];
  accuracy: number | null;
  samples: number;
  history: (0 | 1 | 2)[];
}) {
  const decided = records.filter((r) => r.correct != null);
  const wins = decided.filter((r) => r.correct).length;
  const winPct = decided.length ? (wins / decided.length) * 100 : 0;
  // Per-dozen counts & per-dozen accuracy
  const counts: number[] = [0, 0, 0];
  history.forEach((d) => (counts[d] += 1));
  const perClassCorrect = [0, 0, 0];
  const perClassTotal = [0, 0, 0];
  records.forEach((r) => {
    if (typeof r.label === "number") {
      perClassTotal[r.label] += 1;
      if (r.correct) perClassCorrect[r.label] += 1;
    }
  });
  // Average streak lengths per dozen
  const streakSums = [0, 0, 0];
  const streakCounts = [0, 0, 0];
  if (history.length) {
    let cur = history[0];
    let len = 1;
    for (let i = 1; i < history.length; i++) {
      if (history[i] === cur) len++;
      else {
        streakSums[cur] += len;
        streakCounts[cur] += 1;
        cur = history[i];
        len = 1;
      }
    }
    streakSums[cur] += len;
    streakCounts[cur] += 1;
  }
  const avgStreak = [0, 1, 2].map((i) =>
    streakCounts[i] ? streakSums[i] / streakCounts[i] : 0
  );
  return (
    <div className="mt-2 flex flex-col gap-2 text-[11px]">
      <div className="grid grid-cols-2 gap-2">
        <MetricCard title="Samples" color="sky" value={samples} />
        <MetricCard title="Decisions" color="emerald" value={decided.length} />
        <MetricCard
          title="Win %"
          color="teal"
          value={winPct.toFixed(1) + "%"}
        />
        <MetricCard
          title="Accuracy"
          color="indigo"
          value={accuracy == null ? "—" : (accuracy * 100).toFixed(1) + "%"}
        />
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 lg:grid-cols-7 gap-2">
        {/* Per-class frequency */}
        {[0, 1, 2].map((i) => (
          <MetricCard
            key={"freq-" + i}
            title={(i === 0 ? "1st" : i === 1 ? "2nd" : "3rd") + " Freq"}
            color={i === 0 ? "cyan" : i === 1 ? "violet" : "fuchsia"}
            value={
              samples ? ((counts[i] / samples) * 100).toFixed(1) + "%" : "—"
            }
          />
        ))}
        {[0, 1, 2].map((i) => (
          <MetricCard
            key={"acc-" + i}
            title={(i === 0 ? "1st" : i === 1 ? "2nd" : "3rd") + " Acc"}
            color={i === 0 ? "cyan" : i === 1 ? "violet" : "fuchsia"}
            value={
              perClassTotal[i]
                ? ((perClassCorrect[i] / perClassTotal[i]) * 100).toFixed(1) +
                  "%"
                : "—"
            }
          />
        ))}
        {[0, 1, 2].map((i) => (
          <MetricCard
            key={"streak-" + i}
            title={(i === 0 ? "1st" : i === 1 ? "2nd" : "3rd") + " Avg Run"}
            color={i === 0 ? "cyan" : i === 1 ? "violet" : "fuchsia"}
            value={avgStreak[i].toFixed(2)}
          />
        ))}
        <div className="col-span-full rounded border border-fuchsia-600/40 bg-fuchsia-700/10 px-3 py-2 flex flex-col items-start">
          <span className="text-[9px] uppercase tracking-wide text-fuchsia-300">
            Note
          </span>
          <span className="text-fuchsia-200 text-[10px] leading-snug">
            Per-dozen frequency, accuracy, and average run lengths help spot
            emerging biases or streak structures.
          </span>
        </div>
      </div>
    </div>
  );
}

// Reusable tiny metric card for dozens stats
function MetricCard({
  title,
  value,
  color,
}: {
  title: string;
  value: string | number;
  color: string;
}) {
  const base = {
    sky: "border-sky-600/40 bg-sky-700/10 text-sky-200",
    emerald: "border-emerald-600/40 bg-emerald-700/10 text-emerald-200",
    teal: "border-teal-600/40 bg-teal-700/10 text-teal-200",
    indigo: "border-indigo-600/40 bg-indigo-700/10 text-indigo-200",
    cyan: "border-cyan-600/40 bg-cyan-700/10 text-cyan-200",
    violet: "border-violet-600/40 bg-violet-700/10 text-violet-200",
    fuchsia: "border-fuchsia-600/40 bg-fuchsia-700/10 text-fuchsia-200",
  } as const;
  return (
    <div
      className={`rounded border px-3 py-2 flex flex-col items-start ${
        // @ts-ignore
        base[color] || base.teal
      }`}
    >
      <span className="text-[9px] uppercase tracking-wide opacity-80">
        {title}
      </span>
      <span className="font-semibold tabular-nums text-[11px] leading-tight">
        {value}
      </span>
    </div>
  );
}

function StatsBar({
  records,
  metrics,
  mode,
  variant,
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
  mode: string;
  variant: string;
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
  const isRB = mode === "roulette" && variant === "redblack";
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
          {isRB ? "Red Prediction Hit %" : "Over Prediction Hit %"}
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
          {isRB ? "Avg RED Run" : "Avg OVER Run"}
        </span>
        <span className="text-emerald-200 font-semibold tabular-nums">
          {avgOverRun.toFixed(2)}
        </span>
      </div>
      <div className="rounded border border-cyan-600/40 bg-cyan-700/10 px-3 py-2 flex flex-col items-start">
        <span className="text-[9px] uppercase tracking-wide text-cyan-300">
          {isRB ? "Avg BLACK Run" : "Avg UNDER Run"}
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
    addDozenObservation,
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
    rouletteDozenProbs,
    rouletteDozenHistory,
    rouletteDozenPred,
    rouletteDozenRecords,
    rouletteDozenAccuracy,
    rouletteDozenVersion,
    controlsVersion,
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
          {/* Game Mode Selector */}
          <div className="mt-4 flex flex-wrap gap-2 text-[11px]">
            {[
              {
                key: "binary",
                label: "Binary (Over/Under)",
                mode: "binary",
                variant: "redblack",
              },
              {
                key: "rb",
                label: "Roulette Red/Black",
                mode: "roulette",
                variant: "redblack",
              },
              {
                key: "dz",
                label: "Roulette Dozens",
                mode: "roulette",
                variant: "dozens",
              },
            ].map((item) => {
              const active =
                item.mode === controls.mode &&
                (item.mode === "binary" ||
                  controls.rouletteVariant === item.variant);
              return (
                <button
                  key={item.key}
                  onClick={() => {
                    if (item.mode === "binary") setControls({ mode: "binary" });
                    else
                      setControls({
                        mode: "roulette",
                        rouletteVariant: item.variant as "redblack" | "dozens",
                      });
                  }}
                  className={`px-3 py-1.5 rounded border transition-colors ${
                    active
                      ? "border-teal-500/70 bg-teal-600/10 text-teal-300"
                      : "border-slate-700 bg-slate-800/60 hover:border-teal-400 hover:text-teal-300 text-slate-300"
                  }`}
                >
                  {item.label}
                </button>
              );
            })}
          </div>
        </header>
        <section className="space-y-6">
          <div className="sticky top-0 z-20 pt-2 pb-3 bg-gradient-to-b from-slate-950/95 via-slate-950/80 to-transparent backdrop-blur supports-[backdrop-filter]:backdrop-blur-sm">
            <BinaryButtons
              mode={controls.mode}
              rouletteVariant={controls.rouletteVariant}
              onBelow={() => addObservation(0)}
              onOver={() => addObservation(1)}
              onRed={() => addObservation(1)}
              onBlack={() => addObservation(0)}
              onDozen1={() => addDozenObservation(0)}
              onDozen2={() => addDozenObservation(1)}
              onDozen3={() => addDozenObservation(2)}
              disabled={loading}
              probOver={probOver}
              threshold={controls.threshold}
            />
          </div>
          {controls.mode === "roulette" &&
          controls.rouletteVariant === "dozens" ? (
            <div
              key={"dozens-panel-" + rouletteDozenVersion}
              className="mt-4 p-4 bg-slate-800/60 border border-slate-700 rounded-lg text-[11px] flex flex-col gap-2"
            >
              <div className="flex items-baseline gap-6">
                <h3 className="text-xs tracking-wide text-slate-300 font-semibold">
                  Dozens Prediction
                </h3>
                {rouletteDozenPred != null && (
                  <span className="text-[10px] text-teal-300">
                    Pred:{" "}
                    {rouletteDozenPred === 0
                      ? "1st"
                      : rouletteDozenPred === 1
                      ? "2nd"
                      : "3rd"}{" "}
                    12
                  </span>
                )}
                {rouletteDozenAccuracy != null && (
                  <span className="text-[10px] text-sky-300">
                    Acc {(rouletteDozenAccuracy * 100).toFixed(1)}%
                  </span>
                )}
              </div>
              <DozensPanel
                probs={rouletteDozenProbs || [1 / 3, 1 / 3, 1 / 3]}
                samples={rouletteDozenHistory?.length || 0}
                pred={rouletteDozenPred ?? null}
              />
            </div>
          ) : (
            <BinaryPredictionDisplay
              probOver={probOver}
              accuracy={accuracy}
              loading={loading}
              backend={backend}
              count={history.length}
              mode={controls.mode}
            />
          )}
          {!(
            controls.mode === "roulette" &&
            controls.rouletteVariant === "dozens"
          ) && (
            <StatsBar
              records={records}
              metrics={metrics}
              mode={controls.mode}
              variant={controls.rouletteVariant}
            />
          )}
          {controls.mode === "roulette" &&
            controls.rouletteVariant === "dozens" && (
              <DozensStats
                records={rouletteDozenRecords}
                accuracy={rouletteDozenAccuracy}
                samples={rouletteDozenHistory?.length || 0}
                history={(rouletteDozenHistory as (0 | 1 | 2)[]) || []}
              />
            )}
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
          {controls.mode === "roulette" &&
          controls.rouletteVariant === "dozens" ? (
            <div className="mt-4 p-3 border border-slate-700 rounded bg-slate-900/40 h-48 overflow-y-auto text-[10px] font-mono flex flex-col gap-1">
              {rouletteDozenRecords
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
                  const labelTxt =
                    r.label === 0
                      ? "1st 12"
                      : r.label === 1
                      ? "2nd 12"
                      : "3rd 12";
                  const predTxt =
                    r.predLabel == null
                      ? "n/a"
                      : r.predLabel === 0
                      ? "1st 12"
                      : r.predLabel === 1
                      ? "2nd 12"
                      : "3rd 12";
                  return (
                    <div
                      key={i}
                      className={`flex items-center gap-2 px-2 py-1 rounded border ${color}`}
                    >
                      <span
                        className={`text-[9px] px-1.5 py-0.5 rounded bg-teal-600/20 text-teal-200`}
                      >
                        {labelTxt}
                      </span>
                      <span className="text-slate-400">pred: {predTxt}</span>
                      {r.probs && (
                        <span className="text-slate-400">
                          p={(Math.max(...r.probs) * 100).toFixed(1)}%
                        </span>
                      )}
                      <span className="ml-auto font-semibold tracking-wide">
                        {res}
                      </span>
                    </div>
                  );
                })}
              {rouletteDozenRecords.length === 0 && (
                <span className="text-slate-600">No observations yet.</span>
              )}
            </div>
          ) : (
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
                  const isRoulette = controls.mode === "roulette";
                  const labelTxt = r.label
                    ? isRoulette
                      ? "RED"
                      : "OVER"
                    : isRoulette
                    ? "BLACK"
                    : "UNDER";
                  const predTxt =
                    r.predLabel == null
                      ? "n/a"
                      : r.predLabel
                      ? isRoulette
                        ? "RED"
                        : "OVER"
                      : isRoulette
                      ? "BLACK"
                      : "UNDER";
                  return (
                    <div
                      key={i}
                      className={`flex items-center gap-2 px-2 py-1 rounded border ${color}`}
                    >
                      <span
                        className={`text-[9px] px-1.5 py-0.5 rounded ${
                          r.label
                            ? isRoulette
                              ? "bg-red-600/30 text-red-300"
                              : "bg-teal-500/30 text-teal-200"
                            : isRoulette
                            ? "bg-slate-500/30 text-slate-200"
                            : "bg-cyan-500/20 text-cyan-200"
                        }`}
                      >
                        {labelTxt}
                      </span>
                      <span className="text-slate-400">pred: {predTxt}</span>
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
          )}
          <ControlPanel
            controls={controls}
            setControls={setControls}
            rlWeights={rlWeights}
            resetModel={resetModel}
            loading={loading}
          />
        </section>
        <footer className="mt-12 text-center text-[10px] text-slate-600 space-y-1">
          {controls.mode === "roulette" &&
          controls.rouletteVariant === "dozens" ? (
            <div>
              Dozens mode: record each spin's dozen (1st/2nd/3rd 12).
              Multi-class ensemble updates online with streak, pattern &
              frequency heuristics.
            </div>
          ) : controls.mode === "roulette" ? (
            <div>
              Roulette red/black: click actual color each spin. Binary ensemble
              adapts via streak & pattern features.
            </div>
          ) : (
            <div>
              Binary model: click actual outcome each round (Under or Over).
              Model adapts online using streak & pattern features.
            </div>
          )}
        </footer>
      </div>
    </div>
  );
}
