"use client";
import React, { useMemo } from "react";

// Generic spin entry abstraction used for diagnostics
interface Spin3 {
  actual: number; // class index (0/1/2 for dozens; 0/1 for binary mapped into first two slots)
  pred: number | null;
  probs: [number, number, number]; // for binary: [p1, p0, 0]
  win: boolean | null;
}

export interface RecommendationPanelProps {
  mode: "dozens" | "binary" | "roulette-binary"; // roulette-binary == red/black
  dozen?: {
    label: number;
    predLabel: number | null;
    probs: number[] | null;
    correct: boolean | null;
  }[];
  binary?: {
    label?: number;
    predLabel?: number | null;
    prob?: number | null;
    correct: boolean | null;
  }[];
  adaptive?: {
    cap: number | null;
    floor: number | null;
    shrink: number | null;
  };
  extended?: {
    hiProbMissRate?: number;
    lastHiProbMiss?: boolean;
    capDecayMult?: number;
    shock?: any;
    alternation?: any;
    abstain?: boolean;
  };
  maxLookback?: number;
  className?: string;
}

interface Suggestion {
  id: string;
  title: string;
  reason: string;
  severity: "info" | "warn" | "critical" | "good";
}

function entropy(arr: number[]): number {
  return -arr.reduce((a, p) => (p > 0 ? a + p * Math.log2(p) : a), 0);
}

function pct(v: number): string {
  if (!isFinite(v)) return "--";
  return (100 * v).toFixed(1) + "%";
}

export const RecommendationPanel: React.FC<RecommendationPanelProps> = ({
  mode,
  dozen,
  binary,
  adaptive,
  extended,
  maxLookback = 300,
  className = "",
}) => {
  const spins: Spin3[] = useMemo(() => {
    if (mode === "dozens" && dozen) {
      return dozen.slice(-maxLookback).map((r) => ({
        actual: typeof r.label === "number" ? r.label : 0,
        pred: r.predLabel == null ? null : r.predLabel,
        probs:
          r.probs && r.probs.length === 3
            ? (r.probs as [number, number, number])
            : ([1 / 3, 1 / 3, 1 / 3] as [number, number, number]),
        win: r.correct,
      }));
    }
    if ((mode === "binary" || mode === "roulette-binary") && binary) {
      return binary.slice(-maxLookback).map((r) => {
        const p1 = r.prob == null ? 0.5 : r.prob;
        return {
          actual: r.label ? 1 : 0,
          pred: r.predLabel == null ? null : r.predLabel ? 1 : 0,
          probs: [p1, 1 - p1, 0] as [number, number, number],
          win: r.correct,
        };
      });
    }
    return [];
  }, [mode, dozen, binary, maxLookback]);

  const diag = useMemo(() => {
    const recent = spins;
    const len = recent.length;
    const slice = (n: number) => recent.slice(-n);
    const last50 = slice(50);
    const last25 = slice(25);
    const last5 = slice(5);
    const acc = (arr: Spin3[]) =>
      arr.length ? arr.filter((s) => s.win).length / arr.length : 0;
    const overallAcc = acc(recent);
    const acc50 = acc(last50);
    const acc25 = acc(last25);
    const acc5 = acc(last5);
    const hiThresh = 0.5; // generic
    const hi = recent.filter((s) => Math.max(...s.probs) >= hiThresh);
    const hiAcc = acc(hi);
    let topProbSum = 0;
    let brierSum = 0;
    const predCounts = [0, 0, 0];
    const actualCounts = [0, 0, 0];
    const predWins = [0, 0, 0];
    recent.forEach((s) => {
      const tp = Math.max(...s.probs);
      topProbSum += tp;
      brierSum += s.probs.reduce(
        (a, p, i) =>
          a + (p - (s.actual === i ? 1 : 0)) * (p - (s.actual === i ? 1 : 0)),
        0
      );
      if (s.pred != null) {
        predCounts[s.pred] += 1;
        if (s.win) predWins[s.pred] += 1;
      }
      actualCounts[s.actual] += 1;
    });
    const avgTopProb = len ? topProbSum / len : 0;
    const brier = len ? brierSum / len : 0;
    const entValues = recent.map((s) => entropy(s.probs));
    const entAvg = entValues.length
      ? entValues.reduce((a, b) => a + b, 0) / entValues.length
      : 0;
    const ent5 = last5.length
      ? last5.map((s) => entropy(s.probs)).reduce((a, b) => a + b, 0) /
        last5.length
      : 0;
    const entVar = entValues.length
      ? entValues.reduce((a, e) => a + (e - entAvg) * (e - entAvg), 0) /
        entValues.length
      : 0;
    const entStd = Math.sqrt(entVar);
    // Actual streak length
    let streakLen = 0;
    let streakClass: number | null = null;
    if (recent.length) {
      streakClass = recent[recent.length - 1].actual;
      for (let i = recent.length - 1; i >= 0; i--) {
        if (recent[i].actual === streakClass) streakLen++;
        else break;
      }
    }
    // Miss repetition (same predicted class missed sequentially)
    let missRepeat = 0;
    if (recent.length) {
      const lp = recent[recent.length - 1].pred;
      for (let i = recent.length - 1; i >= 0; i--) {
        if (recent[i].pred === lp) {
          if (recent[i].win === false) missRepeat++;
          else if (recent[i].win === true) break;
        } else break;
      }
    }
    // Short vs long frequency divergence (dozens only)
    let freqShift = 0;
    if (mode === "dozens" && recent.length >= 30) {
      const long = recent.slice(-150);
      const short = last25;
      const f = (arr: Spin3[]) => {
        const c = [0, 0, 0];
        arr.forEach((s) => c[s.actual]++);
        const tot = c[0] + c[1] + c[2] || 1;
        return c.map((x) => x / tot);
      };
      const lf = f(long);
      const sf = f(short);
      // simple L1 distance as shift proxy
      freqShift =
        Math.abs(lf[0] - sf[0]) +
        Math.abs(lf[1] - sf[1]) +
        Math.abs(lf[2] - sf[2]);
    }
    // Prediction vs actual distribution skew (L1)
    const predTot = predCounts[0] + predCounts[1] + predCounts[2] || 1;
    const actTot = actualCounts[0] + actualCounts[1] + actualCounts[2] || 1;
    const predDist = predCounts.map((c) => c / predTot) as [
      number,
      number,
      number
    ];
    const actDist = actualCounts.map((c) => c / actTot) as [
      number,
      number,
      number
    ];
    const distSkew =
      Math.abs(predDist[0] - actDist[0]) +
      Math.abs(predDist[1] - actDist[1]) +
      Math.abs(predDist[2] - actDist[2]);
    const calibGap = avgTopProb - overallAcc; // positive => overconfident
    const perClassPredAcc: [number, number, number] = [0, 1, 2].map((i) =>
      predCounts[i] ? predWins[i] / predCounts[i] : 0
    ) as any;
    return {
      len,
      overallAcc,
      acc50,
      acc25,
      acc5,
      hiAcc,
      hiCount: hi.length,
      avgTopProb,
      brier,
      entAvg,
      ent5,
      entStd,
      streakLen,
      streakClass,
      missRepeat,
      freqShift,
      predDist,
      actDist,
      distSkew,
      calibGap,
      perClassPredAcc,
    };
  }, [spins, mode]);

  const suggestions: Suggestion[] = useMemo(() => {
    const s: Suggestion[] = [];
    if (diag.len < 8) return s;
    // High-confidence miscalibration
    if (diag.hiCount > 12 && diag.hiAcc + 0.03 < diag.overallAcc) {
      s.push({
        id: "hiConfMiscalibration",
        title: "High-confidence miscalibration",
        reason: `Hi-conf acc ${pct(diag.hiAcc)} < overall ${pct(
          diag.overallAcc
        )}; consider lowering confidence clamp or adding calibration.`,
        severity: "warn",
      });
    }
    // Overconfidence calibration gap
    if (diag.len > 30 && diag.calibGap > 0.07) {
      s.push({
        id: "overConfGap",
        title: "Overconfidence gap",
        reason: `Avg top ${(diag.avgTopProb * 100).toFixed(1)}% vs acc ${pct(
          diag.overallAcc
        )} (gap ${(diag.calibGap * 100).toFixed(1)}%). Shrink / clamp sooner.`,
        severity: "warn",
      });
    }
    // Underconfidence gap
    if (diag.len > 30 && diag.calibGap < -0.07) {
      s.push({
        id: "underConfGap",
        title: "Underconfidence gap",
        reason: `Avg top ${(diag.avgTopProb * 100).toFixed(
          1
        )}% below realized acc ${pct(
          diag.overallAcc
        )}. Consider sharpening (lower temperature).`,
        severity: "info",
      });
    }
    // Collapse: low entropy & weak short accuracy
    if (diag.ent5 < 0.9 && diag.acc25 < 0.34) {
      s.push({
        id: "collapse",
        title: "Low-entropy collapse",
        reason: `Entropy(5) ${diag.ent5.toFixed(2)} with Acc(25) ${pct(
          diag.acc25
        )} – flatten distribution (raise temperature).`,
        severity: "critical",
      });
    }
    // Regime / frequency shift (dozens)
    if (mode === "dozens" && diag.freqShift > 0.55) {
      s.push({
        id: "regimeShift",
        title: "Distribution shift",
        reason: `Recent frequency divergence ${diag.freqShift.toFixed(
          2
        )} – increase smoothing or reset model weight ramp.`,
        severity: "warn",
      });
    }
    // Prediction imbalance (skew)
    if (diag.distSkew > 0.8 && diag.len > 40) {
      s.push({
        id: "predSkew",
        title: "Prediction imbalance",
        reason: `L1(pred vs actual) ${diag.distSkew.toFixed(
          2
        )} – diversify or add penalty to dominant class.`,
        severity: "warn",
      });
    }
    // Class neglect (dozens only)
    if (
      mode === "dozens" &&
      diag.actDist.some((a, i) => a > 0.45 && diag.predDist[i] < 0.3)
    ) {
      const i = diag.actDist.findIndex(
        (a, j) => a > 0.45 && diag.predDist[j] < 0.3
      );
      if (i >= 0)
        s.push({
          id: "classNeglect",
          title: "Class neglect",
          reason: `Dozen ${i + 1} actual freq ${(diag.actDist[i] * 100).toFixed(
            1
          )}% but predicted ${(diag.predDist[i] * 100).toFixed(
            1
          )}% – increase exploration toward it.`,
          severity: "warn",
        });
    }
    // Streak exploitation
    if (diag.streakLen >= 4) {
      s.push({
        id: "streak",
        title: "Streak detected",
        reason: `Actual run length ${diag.streakLen}; optionally bias slightly toward continuation but watch for reversal.`,
        severity: "info",
      });
    }
    // Emerging reversal (streak but low confidence on streak class)
    if (diag.streakLen >= 4 && diag.avgTopProb < 0.45) {
      s.push({
        id: "reversalRisk",
        title: "Possible reversal",
        reason: `Streak length ${diag.streakLen} but confidence muted – consider partial hedge.`,
        severity: "info",
      });
    }
    // Repeated misses
    if (diag.missRepeat >= 2) {
      s.push({
        id: "missRepeat",
        title: "Repeat miss risk",
        reason: `${diag.missRepeat} consecutive misses on same prediction – raise diversion / rotate secondary class.`,
        severity: "warn",
      });
    }
    // Brier degradation
    if (diag.len > 50 && diag.brier > 0.68) {
      s.push({
        id: "brier",
        title: "Elevated Brier score",
        reason: `Brier ${diag.brier.toFixed(
          3
        )} > baseline ~0.666 – shrink probs toward uniform.`,
        severity: "warn",
      });
    }
    // Under-confidence
    if (
      diag.avgTopProb < 0.37 &&
      diag.overallAcc > 0.36 &&
      diag.entAvg > 1.05
    ) {
      s.push({
        id: "underConf",
        title: "Under-confidence",
        reason: `Avg top ${(diag.avgTopProb * 100).toFixed(
          1
        )}% with solid accuracy – allow mild sharpening.`,
        severity: "info",
      });
    }
    // Good calibration signal
    if (diag.hiCount > 15 && Math.abs(diag.hiAcc - diag.overallAcc) < 0.015) {
      s.push({
        id: "calibrated",
        title: "Calibration stable",
        reason: `Hi-conf accuracy aligned with overall performance.`,
        severity: "good",
      });
    }
    // Stable balanced regime
    if (
      diag.entAvg > 1.2 &&
      diag.entStd < 0.1 &&
      Math.abs(diag.calibGap) < 0.03 &&
      diag.distSkew < 0.35 &&
      diag.overallAcc >= 0.34 &&
      diag.len > 40
    ) {
      s.push({
        id: "stableBalanced",
        title: "Stable balanced regime",
        reason: `Distribution stable (entropy σ ${diag.entStd.toFixed(
          2
        )}) with small calibration gap; maintain settings.`,
        severity: "good",
      });
    }
    return s;
  }, [diag, mode]);

  return (
    <div
      className={`mt-4 p-3 rounded border border-slate-700 bg-slate-900/40 text-[10px] font-mono ${className}`}
    >
      <h3 className="uppercase tracking-wide text-[9px] text-slate-400 mb-2">
        Recommendations (Informational)
      </h3>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 mb-3 text-[9px] text-slate-300">
        <Metric label="Acc(50)" value={pct(diag.acc50)} />
        <Metric label="Acc(25)" value={pct(diag.acc25)} />
        <Metric label="Acc(5)" value={pct(diag.acc5)} />
        <Metric label="HiConf Acc" value={pct(diag.hiAcc)} />
        <Metric
          label="Avg Top"
          value={(diag.avgTopProb * 100).toFixed(1) + "%"}
        />
        <Metric label="Brier" value={diag.brier.toFixed(3)} />
        <Metric label="Ent Avg" value={diag.entAvg.toFixed(2)} />
        <Metric label="Ent(5)" value={diag.ent5.toFixed(2)} />
        {mode === "dozens" && (
          <Metric label="Freq Δ" value={diag.freqShift.toFixed(2)} />
        )}
        <Metric
          label="Streak"
          value={diag.streakLen ? diag.streakLen.toString() : "-"}
        />
        <Metric label="MissRep" value={diag.missRepeat.toString()} />
        <Metric label="Ent σ" value={diag.entStd.toFixed(2)} />
        <Metric label="CalGap" value={(diag.calibGap * 100).toFixed(1) + "%"} />
        <Metric label="Skew" value={diag.distSkew.toFixed(2)} />
        {mode === "dozens" && (
          <Metric
            label="Pred Dist"
            value={diag.predDist.map((p) => (p * 100).toFixed(0)).join("/")}
          />
        )}
        {mode === "dozens" && (
          <Metric
            label="Act Dist"
            value={diag.actDist.map((p) => (p * 100).toFixed(0)).join("/")}
          />
        )}
        {mode === "dozens" && adaptive && adaptive.cap != null && (
          <Metric label="Cap" value={(adaptive.cap * 100).toFixed(1) + "%"} />
        )}
        {mode === "dozens" && adaptive && adaptive.floor != null && (
          <Metric
            label="Floor"
            value={(adaptive.floor * 100).toFixed(1) + "%"}
          />
        )}
        {mode === "dozens" && adaptive && adaptive.shrink != null && (
          <Metric
            label="Shrink"
            value={(adaptive.shrink * 100).toFixed(0) + "%"}
          />
        )}
        {mode === "dozens" && extended?.hiProbMissRate != null && (
          <Metric
            label="HiMiss%"
            value={((extended.hiProbMissRate || 0) * 100).toFixed(0) + "%"}
          />
        )}
        {mode === "dozens" && extended?.capDecayMult != null && (
          <Metric
            label="CapDec"
            value={(extended.capDecayMult as number).toFixed(2)}
          />
        )}
        {mode === "dozens" && extended?.shock && (
          <Metric
            label="Shock"
            value={
              extended.shock.active ? `ON:${extended.shock.ttl || 0}` : "off"
            }
          />
        )}
        {mode === "dozens" && extended?.alternation && (
          <Metric
            label="Alt"
            value={
              extended.alternation.active ? extended.alternation.spins : "-"
            }
          />
        )}
        {mode === "dozens" && extended?.abstain != null && (
          <Metric label="Abstain" value={extended.abstain ? "Yes" : "No"} />
        )}
      </div>
      <ul className="space-y-2">
        {suggestions.length === 0 && (
          <li className="text-slate-500 italic">
            No notable adjustments suggested.
          </li>
        )}
        {suggestions.map((sg) => (
          <li
            key={sg.id}
            className={`p-2 rounded border text-[10px] leading-snug ${
              sg.severity === "critical"
                ? "border-rose-600/60 bg-rose-900/20 text-rose-200"
                : sg.severity === "warn"
                ? "border-amber-600/60 bg-amber-900/20 text-amber-200"
                : sg.severity === "good"
                ? "border-teal-600/60 bg-teal-900/20 text-teal-200"
                : "border-slate-600/50 bg-slate-800/40 text-slate-300"
            }`}
          >
            <span className="font-semibold mr-2">{sg.title}</span>
            {sg.reason}
          </li>
        ))}
      </ul>
    </div>
  );
};

const Metric: React.FC<{ label: string; value: string }> = ({
  label,
  value,
}) => (
  <div className="flex flex-col px-2 py-1 rounded bg-slate-800/40 border border-slate-700/40">
    <span className="text-[8px] uppercase tracking-wide text-slate-500">
      {label}
    </span>
    <span className="text-[10px] text-slate-200">{value}</span>
  </div>
);

export default RecommendationPanel;
