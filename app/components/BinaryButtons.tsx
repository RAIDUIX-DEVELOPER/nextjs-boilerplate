"use client";
import React from "react";

export const BinaryButtons: React.FC<{
  onBelow: () => void;
  onOver: () => void;
  disabled?: boolean;
  probOver?: number | null;
  threshold?: number;
}> = ({ onBelow, onOver, disabled, probOver, threshold = 0.5 }) => {
  const isOver = probOver != null && probOver >= threshold;
  const label =
    probOver == null ? "Waiting" : isOver ? "Predict OVER" : "Predict BELOW";
  const pTxt = probOver == null ? "" : (probOver * 100).toFixed(1) + "%";
  return (
    <div className="flex items-stretch gap-3 mt-2">
      <div className="flex gap-3 flex-1">
        <button
          onClick={onOver}
          disabled={disabled}
          className="w-full px-4 py-3 rounded bg-slate-800 border border-slate-600 hover:border-teal-400 hover:text-teal-300 text-sm disabled:opacity-40 transition-colors"
        >
          Over 2
        </button>
        <button
          onClick={onBelow}
          disabled={disabled}
          className="w-full px-4 py-3 rounded bg-slate-800 border border-slate-600 hover:border-rose-400 hover:text-rose-300 text-sm disabled:opacity-40 transition-colors"
        >
          Below 2
        </button>
      </div>
      <div
        className={`min-w-[130px] px-3 py-2 rounded border text-[11px] flex flex-col justify-center ${
          probOver == null
            ? "border-slate-600 bg-slate-800/60 text-slate-400"
            : isOver
            ? "border-teal-600/50 bg-teal-600/10 text-teal-300"
            : "border-rose-600/50 bg-rose-600/10 text-rose-300"
        }`}
      >
        <span className="uppercase tracking-wide font-semibold leading-tight">
          {label}
        </span>
        {probOver != null && (
          <span className="text-[10px] opacity-80">p {pTxt}</span>
        )}
      </div>
    </div>
  );
};
