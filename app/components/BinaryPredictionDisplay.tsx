"use client";
import React from "react";

export const BinaryPredictionDisplay: React.FC<{
  probOver: number | null;
  accuracy: number | null;
  loading: boolean;
  backend: string | null;
  count: number;
  threshold: number;
  suggestedThreshold?: number;
  biasInfo?: {
    predRate: number | null;
    actualRate: number | null;
    window: number;
    status: string;
    override: boolean;
  };
}> = ({
  probOver,
  accuracy,
  loading,
  backend,
  count,
  threshold,
  suggestedThreshold,
  biasInfo,
}) => {
  const rec = probOver == null ? "—" : probOver >= 0.5 ? "OVER >2" : "BELOW ≤2";
  return (
    <div className="mt-6 p-4 bg-slate-800/60 border border-slate-700 rounded-lg shadow-inner">
      <div className="text-sm text-slate-400 uppercase mb-2 tracking-wide">
        Prediction
      </div>
      <div className="text-4xl font-semibold tabular-nums bg-gradient-to-r from-teal-300 to-cyan-400 bg-clip-text text-transparent">
        {probOver == null ? "—" : (probOver * 100).toFixed(2) + "%"}
      </div>
      <div className="mt-2 text-xs text-slate-400">
        Likelihood next value is OVER 2
      </div>
      <div className="mt-4 text-sm font-mono">
        Recommendation: <span className="text-slate-200">{rec}</span>
      </div>
      <div className="mt-4 grid grid-cols-3 gap-3 text-[11px] text-slate-400">
        <div>
          Samples
          <br />
          <span className="text-slate-200 font-mono">{count}</span>
        </div>
        <div>
          Accuracy
          <br />
          <span className="text-slate-200 font-mono">
            {accuracy == null ? "—" : (accuracy * 100).toFixed(1) + "%"}
          </span>
        </div>
        <div>
          Backend
          <br />
          <span className="text-slate-200 font-mono">{backend || "—"}</span>
        </div>
        <div className="col-span-3">
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
        {biasInfo && (
          <div className="col-span-3 mt-2 border-t border-slate-700 pt-2 grid grid-cols-2 gap-2">
            <div>
              Bias Window
              <br />
              <span className="font-mono text-slate-200">
                {biasInfo.window}
              </span>
            </div>
            <div>
              Pred/Actual Rate
              <br />
              <span className="font-mono text-slate-200">
                {biasInfo.predRate == null
                  ? "—"
                  : (biasInfo.predRate * 100).toFixed(0)}
                % /{" "}
                {biasInfo.actualRate == null
                  ? "—"
                  : (biasInfo.actualRate * 100).toFixed(0)}
                %
              </span>
            </div>
            <div className="col-span-2">
              Bias Status
              <br />
              <span
                className={`font-mono ${
                  biasInfo.override ? "text-amber-400" : "text-slate-300"
                }`}
              >
                {biasInfo.status}
                {biasInfo.override ? " (override)" : ""}
              </span>
            </div>
          </div>
        )}
      </div>
      <div className="mt-3 text-[10px] text-slate-500 flex gap-3">
        <span>
          Threshold:{" "}
          <span className="text-slate-300">{threshold.toFixed(2)}</span>
        </span>
        {suggestedThreshold !== undefined && (
          <span>
            Suggested:{" "}
            <span className="text-amber-400">
              {suggestedThreshold.toFixed(2)}
            </span>
          </span>
        )}
      </div>
    </div>
  );
};
