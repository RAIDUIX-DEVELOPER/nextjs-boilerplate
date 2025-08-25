"use client";
// HistoryList replaced by inline chips for binary outcomes
import React, { useState, useMemo } from "react";
import { BinaryButtons } from "./components/BinaryButtons";
import { ControlPanel } from "./components/ControlPanel";
import { useBinaryModel } from "./hooks/useBinaryModel";
import RecommendationPanel from "./components/RecommendationPanel";

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
    probs?: number[] | null;
  }[];
  accuracy: number | null;
  samples: number; // total samples overall
  history: (0 | 1 | 2)[];
}) {
  const [windowSize, setWindowSize] = useState<number | "ALL">("ALL");
  const windowRecords = useMemo(
    () => (windowSize === "ALL" ? records : records.slice(-windowSize)),
    [records, windowSize]
  );
  const windowHistory = useMemo(
    () => (windowSize === "ALL" ? history : history.slice(-windowSize)),
    [history, windowSize]
  );
  // Per-dozen occurrence counts (based on actual history subset)
  const counts = [0, 0, 0];
  windowHistory.forEach((d) => (counts[d] += 1));
  // Per-class accuracy (occurrence accuracy: how often we were correct when that dozen appeared)
  const perClassCorrect = [0, 0, 0];
  const perClassTotal = [0, 0, 0];
  windowRecords.forEach((r) => {
    if (typeof r.label === "number" && r.correct != null) {
      perClassTotal[r.label] += 1;
      if (r.correct) perClassCorrect[r.label] += 1;
    }
  });
  // Prediction-based win% (success rate conditioned on predicting that dozen)
  const predPerClassCorrect = [0, 0, 0];
  const predPerClassTotal = [0, 0, 0];
  windowRecords.forEach((r) => {
    if (typeof r.predLabel === "number" && r.correct != null) {
      predPerClassTotal[r.predLabel] += 1;
      if (r.correct) predPerClassCorrect[r.predLabel] += 1;
    }
  });
  // Window accuracy (overall)
  const windowDecided = windowRecords.filter((r) => r.correct != null);
  const windowWins = windowDecided.filter((r) => r.correct).length;
  const windowWinPct = windowDecided.length
    ? (windowWins / windowDecided.length) * 100
    : 0;
  // Streak/run length stats for windowHistory
  const streakSums = [0, 0, 0];
  const streakCounts = [0, 0, 0];
  if (windowHistory.length) {
    let cur = windowHistory[0];
    let len = 1;
    for (let i = 1; i < windowHistory.length; i++) {
      if (windowHistory[i] === cur) len++;
      else {
        streakSums[cur] += len;
        streakCounts[cur] += 1;
        cur = windowHistory[i];
        len = 1;
      }
    }
    streakSums[cur] += len;
    streakCounts[cur] += 1;
  }
  const avgStreak = [0, 1, 2].map((i) =>
    streakCounts[i] ? streakSums[i] / streakCounts[i] : 0
  );
  // Prediction correctness streak metrics (per dozen) within windowRecords
  const curWinPer = [0, 0, 0];
  const curLosePer = [0, 0, 0];
  const longWinPer = [0, 0, 0];
  const longLosePer = [0, 0, 0];
  let overallCurWin = 0,
    overallCurLose = 0,
    overallLongestWin = 0,
    overallLongestLose = 0;
  let sumWinRuns = 0,
    countWinRuns = 0,
    sumLoseRuns = 0,
    countLoseRuns = 0;
  let runType: "win" | "lose" | null = null;
  let runLen = 0;
  for (let i = 0; i < windowRecords.length; i++) {
    const r = windowRecords[i];
    if (r.correct == null || typeof r.label !== "number") continue;
    const d = r.label as 0 | 1 | 2;
    const isWin = r.correct === true;
    if (isWin) {
      overallCurWin += 1;
      overallCurLose = 0;
      if (overallCurWin > overallLongestWin) overallLongestWin = overallCurWin;
      curWinPer[d] += 1;
      curLosePer[d] = 0;
      if (curWinPer[d] > longWinPer[d]) longWinPer[d] = curWinPer[d];
    } else {
      overallCurLose += 1;
      overallCurWin = 0;
      if (overallCurLose > overallLongestLose)
        overallLongestLose = overallCurLose;
      curLosePer[d] += 1;
      curWinPer[d] = 0;
      if (curLosePer[d] > longLosePer[d]) longLosePer[d] = curLosePer[d];
    }
    const currentType = isWin ? "win" : "lose";
    if (runType == null) {
      runType = currentType;
      runLen = 1;
    } else if (runType === currentType) {
      runLen += 1;
    } else {
      if (runType === "win") {
        sumWinRuns += runLen;
        countWinRuns += 1;
      } else {
        sumLoseRuns += runLen;
        countLoseRuns += 1;
      }
      runType = currentType;
      runLen = 1;
    }
  }
  if (runType === "win" && runLen) {
    sumWinRuns += runLen;
    countWinRuns += 1;
  } else if (runType === "lose" && runLen) {
    sumLoseRuns += runLen;
    countLoseRuns += 1;
  }
  const avgWinRun = countWinRuns ? sumWinRuns / countWinRuns : 0;
  const avgLoseRun = countLoseRuns ? sumLoseRuns / countLoseRuns : 0;
  const freqPct = (i: number) =>
    windowHistory.length
      ? ((counts[i] / windowHistory.length) * 100).toFixed(1) + "%"
      : "—";
  const occAccPct = (i: number) =>
    perClassTotal[i]
      ? ((perClassCorrect[i] / perClassTotal[i]) * 100).toFixed(1) + "%"
      : "—";
  const winPctPerPred = (i: number) =>
    predPerClassTotal[i]
      ? ((predPerClassCorrect[i] / predPerClassTotal[i]) * 100).toFixed(1) + "%"
      : "—";
  return (
    <div className="mt-3 text-[11px] space-y-3">
      <div className="space-y-2">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-2">
            <span className="text-[10px] uppercase tracking-wide text-slate-500 pl-1 border-l border-slate-700/60">
              Window
            </span>
            <div className="flex gap-1">
              {[25, 50, 100, 500, "ALL"].map((w) => (
                <button
                  key={w}
                  onClick={() => setWindowSize(w as any)}
                  className={`h-7 px-3 rounded-md text-[11px] font-medium transition-colors border ${
                    windowSize === w
                      ? "border-teal-500/70 bg-teal-600/20 text-teal-300 shadow-[0_0_0_1px_rgba(45,212,191,0.3)]"
                      : "border-slate-700 bg-slate-800/60 text-slate-300 hover:border-teal-500/60 hover:text-teal-200"
                  }`}
                >
                  {w}
                </button>
              ))}
            </div>
          </div>
        </div>
        <div className="flex flex-wrap gap-2 items-stretch">
          {[
            { label: "Total", value: samples, title: "Total samples overall" },
            {
              label: "Window",
              value: windowHistory.length,
              title: "Samples inside selected window",
            },
            {
              label: "Dec",
              value: windowDecided.length,
              title: "Predictions made (decided spins) in window",
            },
            {
              label: "Win%",
              value: windowWinPct.toFixed(1) + "%",
              title: "Win% (correct / decided) in window",
            },
            {
              label: "Acc",
              value: windowDecided.length
                ? ((windowWins / windowDecided.length) * 100).toFixed(1) + "%"
                : "—",
              title: "Overall accuracy in window",
            },
          ].map((box) => (
            <div
              key={box.label}
              title={box.title}
              className="h-8 px-3 rounded-md bg-slate-800/70 border border-slate-700 flex items-center gap-2"
            >
              <span className="text-[10px] uppercase tracking-wide text-slate-400">
                {box.label}
              </span>
              <span className="tabular-nums text-slate-200 text-[11px]">
                {box.value}
              </span>
            </div>
          ))}
          <div
            title="Average consecutive correct prediction run"
            className="h-8 px-3 rounded-md bg-gradient-to-r from-emerald-900/40 to-emerald-800/20 border border-emerald-700/60 text-emerald-300 flex items-center gap-2"
          >
            <span className="text-[10px] uppercase tracking-wide">Avg W</span>
            <span className="tabular-nums text-[11px]">
              {avgWinRun.toFixed(2)}
            </span>
          </div>
          <div
            title="Average consecutive incorrect prediction run"
            className="h-8 px-3 rounded-md bg-gradient-to-r from-rose-900/40 to-rose-800/20 border border-rose-700/60 text-rose-300 flex items-center gap-2"
          >
            <span className="text-[10px] uppercase tracking-wide">Avg L</span>
            <span className="tabular-nums text-[11px]">
              {avgLoseRun.toFixed(2)}
            </span>
          </div>
        </div>
      </div>
      <table className="w-full text-[11px] border-separate border-spacing-y-1">
        <thead>
          <tr className="text-slate-400">
            <th className="text-left px-2 py-1">Dozen</th>
            <th className="text-right px-2 py-1">Freq</th>
            <th className="text-right px-2 py-1">Occ Acc</th>
            <th className="text-right px-2 py-1">Win%</th>
            <th className="text-right px-2 py-1">Avg Run</th>
            <th className="text-right px-2 py-1">L.Win</th>
            <th className="text-right px-2 py-1">L.Lose</th>
            <th className="text-right px-2 py-1">Cur Win</th>
            <th className="text-right px-2 py-1">Cur Lose</th>
          </tr>
        </thead>
        <tbody>
          <tr className="bg-slate-900/70">
            <td className="px-2 py-1 font-semibold text-slate-200">Overall</td>
            <td className="px-2 py-1 text-right text-slate-300">100%</td>
            <td className="px-2 py-1 text-right text-slate-300">
              {windowDecided.length
                ? ((windowWins / windowDecided.length) * 100).toFixed(1) + "%"
                : "—"}
            </td>
            <td className="px-2 py-1 text-right text-teal-300">
              {windowDecided.length ? windowWinPct.toFixed(1) + "%" : "—"}
            </td>
            <td className="px-2 py-1 text-right text-slate-300">
              {(avgStreak.reduce((a, b) => a + b, 0) / 3).toFixed(2)}
            </td>
            <td className="px-2 py-1 text-right text-emerald-300">
              {Math.max(...longWinPer)}
            </td>
            <td className="px-2 py-1 text-right text-rose-300">
              {Math.max(...longLosePer)}
            </td>
            <td className="px-2 py-1 text-right text-emerald-400/80">
              {overallCurWin}
            </td>
            <td className="px-2 py-1 text-right text-rose-400/80">
              {overallCurLose}
            </td>
          </tr>
          {[0, 1, 2].map((i) => {
            const curWin = curWinPer[i];
            const curLose = curLosePer[i];
            return (
              <tr
                key={i}
                className="bg-slate-900/50 hover:bg-slate-800/50 transition-colors"
              >
                <td className="px-2 py-1 font-medium text-slate-300">
                  {i === 0 ? "1st 12" : i === 1 ? "2nd 12" : "3rd 12"}
                </td>
                <td className="px-2 py-1 text-right text-slate-300">
                  {freqPct(i)}
                </td>
                <td className="px-2 py-1 text-right text-slate-300">
                  {occAccPct(i)}
                </td>
                <td className="px-2 py-1 text-right text-teal-300">
                  {winPctPerPred(i)}
                </td>
                <td className="px-2 py-1 text-right text-slate-300">
                  {avgStreak[i].toFixed(2)}
                </td>
                <td className="px-2 py-1 text-right text-emerald-300">
                  {longWinPer[i]}
                </td>
                <td className="px-2 py-1 text-right text-rose-300">
                  {longLosePer[i]}
                </td>
                <td className="px-2 py-1 text-right text-emerald-400/80">
                  {curWin}
                </td>
                <td className="px-2 py-1 text-right text-rose-400/80">
                  {curLose}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="flex gap-3 flex-wrap text-[10px] text-slate-500">
        <span>Occ Acc = accuracy on spins where that dozen occurred.</span>
        <span>Win% = accuracy when predicting that dozen.</span>
        <span>Window buttons slice recent samples.</span>
        <span>
          Avg W/L Run = mean consecutive correct / incorrect prediction streaks.
        </span>
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
    rose: "border-rose-600/40 bg-rose-700/10 text-rose-200",
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
    rouletteDozenAdaptive,
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
    <div className="min-h-screen w-full bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 text-slate-200 px-6 py-10">
      <div className="max-w-2xl mx-auto">
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
              // Re-ordered so Roulette Dozens appears first / default
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
              {
                key: "binary",
                label: "Binary (Over/Under)",
                mode: "binary",
                variant: "redblack",
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
              dozenProbs={rouletteDozenProbs || undefined}
              dozenPred={rouletteDozenPred ?? null}
            />
          </div>
          {/* Removed legacy large Dozens Prediction panel (superseded by compact buttons + stats) */}
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
            <>
              <DozensHistoryPanel records={rouletteDozenRecords} />
              <RecommendationPanel
                mode="dozens"
                dozen={rouletteDozenRecords.map((r) => ({
                  label: r.label as number,
                  predLabel: r.predLabel as number | null,
                  probs: r.probs as number[] | null,
                  correct: r.correct,
                }))}
                adaptive={rouletteDozenAdaptive}
              />
            </>
          ) : (
            <>
              <BinaryHistoryPanel
                records={records}
                isRoulette={controls.mode === "roulette"}
              />
              <RecommendationPanel
                mode={
                  controls.mode === "roulette" ? "roulette-binary" : "binary"
                }
                binary={records.map((r) => ({
                  label: r.label,
                  predLabel: r.predLabel,
                  prob: r.prob,
                  correct: r.correct,
                }))}
              />
            </>
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

// Inline panels with copy buttons (HistoryList component not used here)
function DozensHistoryPanel({
  records,
}: {
  records: {
    label?: number;
    predLabel?: number | null;
    correct: boolean | null;
    probs?: number[] | null;
  }[];
}) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      // Plain text lines mirroring on-screen order (newest first)
      const lines: string[] = [];
      records
        .slice()
        .reverse()
        .forEach((r, idx) => {
          const labelTxt =
            r.label === 0
              ? "1st12"
              : r.label === 1
              ? "2nd12"
              : r.label === 2
              ? "3rd12"
              : "?";
          const predTxt =
            r.predLabel == null
              ? "n/a"
              : r.predLabel === 0
              ? "1st12"
              : r.predLabel === 1
              ? "2nd12"
              : "3rd12";
          const resTxt = r.correct == null ? "?" : r.correct ? "WIN" : "LOSS";
          const topProb = r.probs
            ? (Math.max(...r.probs) * 100).toFixed(1) + "%"
            : "n/a";
          const dist = r.probs
            ? r.probs.map((p) => (p * 100).toFixed(1) + "%").join("/")
            : "";
          lines.push(
            `#${
              idx + 1
            } actual=${labelTxt} pred=${predTxt} p=${topProb} dist=[${dist}] result=${resTxt}`
          );
        });
      const txt = lines.join("\n");
      await navigator.clipboard.writeText(txt);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (e) {
      console.error("Copy dozens history failed", e);
    }
  };
  return (
    <div className="mt-4 p-3 border border-slate-700 rounded bg-slate-900/40 h-48 text-[10px] font-mono flex flex-col">
      <div className="flex items-center justify-between mb-1">
        <span className="uppercase tracking-wide text-[9px] text-slate-400">
          History
        </span>
        <button
          onClick={copy}
          disabled={!records.length}
          className="px-2 py-0.5 rounded text-[9px] border border-slate-600/70 bg-slate-800/60 hover:border-teal-500/60 hover:text-teal-300 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <div className="flex-1 overflow-y-auto flex flex-col gap-1">
        {records
          .slice()
          .reverse()
          .map((r, i) => {
            const res = r.correct == null ? "—" : r.correct ? "WIN" : "LOSS";
            const color =
              r.correct == null
                ? "bg-slate-600/30 border-slate-600/50 text-slate-300"
                : r.correct
                ? "bg-emerald-600/30 border-emerald-500/40 text-emerald-200"
                : "bg-rose-700/30 border-rose-600/40 text-rose-200";
            const labelTxt =
              r.label === 0 ? "1st 12" : r.label === 1 ? "2nd 12" : "3rd 12";
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
                className={`px-2 py-1 rounded border ${color} flex flex-col gap-0.5`}
                title="Dozens prediction record"
              >
                <div className="flex items-center gap-2">
                  <span
                    className={`text-[9px] px-1.5 py-0.5 rounded bg-teal-600/20 text-teal-200`}
                  >
                    {labelTxt}
                  </span>
                  <span className="text-slate-400">pred: {predTxt}</span>
                  {r.probs && (
                    <span className="text-slate-400">
                      top {(Math.max(...r.probs) * 100).toFixed(1)}%
                    </span>
                  )}
                  <span className="ml-auto font-semibold tracking-wide">
                    {res}
                  </span>
                </div>
                {r.probs && (
                  <div className="text-[9px] text-slate-500 font-mono flex flex-wrap gap-2">
                    {r.probs.map((p, pi) => {
                      const pct = (p * 100).toFixed(1) + "%";
                      const lab = pi === 0 ? "1st" : pi === 1 ? "2nd" : "3rd";
                      const isPred = r.predLabel === pi;
                      return (
                        <span
                          key={pi}
                          className={isPred ? "text-teal-300" : ""}
                        >
                          {lab}:{pct}
                        </span>
                      );
                    })}
                    {(() => {
                      const sorted = [...r.probs].sort((a, b) => b - a);
                      if (sorted.length > 1) {
                        const margin = ((sorted[0] - sorted[1]) * 100).toFixed(
                          1
                        );
                        return (
                          <span className="text-slate-400">Δ{margin}%</span>
                        );
                      }
                      return null;
                    })()}
                  </div>
                )}
              </div>
            );
          })}
        {!records.length && (
          <span className="text-slate-600">No observations yet.</span>
        )}
      </div>
    </div>
  );
}

function BinaryHistoryPanel({
  records,
  isRoulette,
}: {
  records: {
    label?: number;
    predLabel?: number | null;
    correct: boolean | null;
    prob?: number | null;
  }[];
  isRoulette: boolean;
}) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      // Plain text lines mirroring on-screen order (newest first)
      const lines: string[] = [];
      records
        .slice()
        .reverse()
        .forEach((r, idx) => {
          const actual = r.label
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
          const resTxt = r.correct == null ? "?" : r.correct ? "WIN" : "LOSS";
          const pTxt = r.prob == null ? "n/a" : (r.prob * 100).toFixed(1) + "%";
          lines.push(
            `#${
              idx + 1
            } actual=${actual} pred=${predTxt} p=${pTxt} result=${resTxt}`
          );
        });
      const txt = lines.join("\n");
      await navigator.clipboard.writeText(txt);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (e) {
      console.error("Copy binary history failed", e);
    }
  };
  return (
    <div className="mt-4 p-3 border border-slate-700 rounded bg-slate-900/40 h-48 text-[10px] font-mono flex flex-col">
      <div className="flex items-center justify-between mb-1">
        <span className="uppercase tracking-wide text-[9px] text-slate-400">
          History
        </span>
        <button
          onClick={copy}
          disabled={!records.length}
          className="px-2 py-0.5 rounded text-[9px] border border-slate-600/70 bg-slate-800/60 hover:border-teal-500/60 hover:text-teal-300 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <div className="flex-1 overflow-y-auto flex flex-col gap-1">
        {records
          .slice()
          .reverse()
          .map((r, i) => {
            const res = r.correct == null ? "—" : r.correct ? "WIN" : "LOSS";
            const color =
              r.correct == null
                ? "bg-slate-600/30 border-slate-600/50 text-slate-300"
                : r.correct
                ? "bg-emerald-600/30 border-emerald-500/40 text-emerald-200"
                : "bg-rose-700/30 border-rose-600/40 text-rose-200";
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
                className={`px-2 py-1 rounded border ${color} flex flex-col gap-0.5`}
                title="Binary prediction record"
              >
                <div className="flex items-center gap-2">
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
                {typeof r.prob === "number" && (
                  <div className="text-[9px] text-slate-500 font-mono flex gap-3">
                    <span>
                      dist:[{(r.prob * 100).toFixed(1)}%/
                      {((1 - r.prob) * 100).toFixed(1)}%]
                    </span>
                    <span>Δ{(Math.abs(r.prob - 0.5) * 200).toFixed(1)}%</span>
                  </div>
                )}
              </div>
            );
          })}
        {!records.length && (
          <span className="text-slate-600">No observations yet.</span>
        )}
      </div>
    </div>
  );
}
