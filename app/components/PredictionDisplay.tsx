"use client";
import React from "react";

interface Props {
  prediction: number | null;
  variance?: number | null;
  loading: boolean;
  stats: { count: number; mean: number; std: number };
  backend: string | null;
  ready?: boolean;
  windowSize?: number;
  ar1Phi?: number;
  weights?: [number, number, number]; // cnn, ar1, markov
  rewardScore?: number;
  probAbove2?: number | null;
}

export const PredictionDisplay: React.FC<Props> = ({
  prediction,
  variance,
  loading,
  stats,
  backend,
  ready,
  windowSize,
  ar1Phi,
  weights,
  rewardScore,
  probAbove2,
}) => {
  return (
    <div className="mt-6 p-4 bg-slate-800/60 border border-slate-700 rounded-lg shadow-inner">
      <div className="text-sm text-slate-400 uppercase mb-2 tracking-wide">
        Model Output
      </div>
      <div className="text-3xl font-semibold text-teal-400 tabular-nums">
        {prediction !== null ? prediction.toFixed(4) : "—"}
      </div>
      <div className="mt-4 grid grid-cols-2 md:grid-cols-5 gap-3 text-[11px] text-slate-400">
        <div>
          Samples
          <br />
          <span className="text-slate-200 font-mono">{stats.count}</span>
        </div>
        <div>
          Mean
          <br />
          <span className="text-slate-200 font-mono">
            {isFinite(stats.mean) ? stats.mean.toFixed(2) : "—"}
          </span>
        </div>
        <div>
          Std
          <br />
          <span className="text-slate-200 font-mono">
            {isFinite(stats.std) ? stats.std.toFixed(2) : "—"}
          </span>
        </div>
        <div>
          Backend
          <br />
          <span className="text-slate-200 font-mono">{backend || "—"}</span>
        </div>
        <div>
          Status
          <br />
          <span
            className={`font-mono ${
              loading ? "text-amber-400" : "text-emerald-400"
            }`}
          >
            {loading ? "training" : "idle"}
          </span>
        </div>
        <div className="col-span-2 md:col-span-5 space-y-1">
          <div>
            Mode
            <br />
            <span className="font-mono text-xs text-slate-300">
              {ready
                ? "Ensemble (CNN + AR1) prediction"
                : `Baseline (running mean) - need > ${windowSize || 0} samples`}
            </span>
          </div>
          {ready && (
            <div className="text-[10px] text-slate-500 flex flex-wrap gap-4">
              <span>
                AR1 φ:{" "}
                <span className="text-slate-300">{ar1Phi?.toFixed(3)}</span>
              </span>
              {variance != null && (
                <span>
                  σ²≈{" "}
                  <span className="text-slate-300">{variance.toFixed(2)}</span>
                </span>
              )}
              {weights && (
                <span>
                  w=[
                  <span className="text-slate-300">
                    {weights.map((w, i) => w.toFixed(2)).join(",")}
                  </span>
                  ]
                </span>
              )}
              {rewardScore !== undefined && (
                <span>
                  Reward:{" "}
                  <span className="text-slate-300">
                    {rewardScore.toFixed(2)}
                  </span>
                </span>
              )}
              {probAbove2 != null && (
                <span>
                  P(x&gt;2):{" "}
                  <span className="text-slate-300">
                    {probAbove2.toFixed(3)}
                  </span>
                </span>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PredictionDisplay;
