"use client";
import React from "react";

export const BinaryButtons: React.FC<{
  onBelow: () => void;
  onOver: () => void;
  onRed?: () => void; // roulette mode handlers
  onBlack?: () => void;
  onDozen1?: () => void;
  onDozen2?: () => void;
  onDozen3?: () => void;
  disabled?: boolean;
  probOver?: number | null;
  threshold?: number;
  mode?: "binary" | "roulette";
  rouletteVariant?: "redblack" | "dozens";
}> = ({
  onBelow,
  onOver,
  onRed,
  onBlack,
  onDozen1,
  onDozen2,
  onDozen3,
  disabled,
  probOver,
  threshold = 0.5,
  mode = "binary",
  rouletteVariant = "redblack",
}) => {
  const isOver = probOver != null && probOver >= threshold;
  const label =
    probOver == null ? "Waiting" : isOver ? "Predict OVER" : "Predict UNDER";
  const pTxt = probOver == null ? "" : (probOver * 100).toFixed(1) + "%";
  if (mode === "roulette" && rouletteVariant === "redblack") {
    return (
      <div className="flex items-stretch gap-3 mt-2">
        <div className="flex gap-3 flex-1">
          <button
            onClick={onRed}
            disabled={disabled}
            className="w-full px-4 py-3 rounded bg-red-900/40 border border-red-600 hover:border-red-400 hover:text-red-300 text-sm disabled:opacity-40 transition-colors"
          >
            Red
          </button>
          <button
            onClick={onBlack}
            disabled={disabled}
            className="w-full px-4 py-3 rounded bg-slate-800 border border-slate-600 hover:border-slate-400 hover:text-slate-200 text-sm disabled:opacity-40 transition-colors"
          >
            Black
          </button>
        </div>
        <div
          className={`min-w-[130px] px-3 py-2 rounded border text-[11px] flex flex-col justify-center ${
            probOver == null
              ? "border-slate-600 bg-slate-800/60 text-slate-400"
              : isOver
              ? "border-red-600/60 bg-red-700/10 text-red-300"
              : "border-slate-500/60 bg-slate-700/20 text-slate-200"
          }`}
        >
          <span className="uppercase tracking-wide font-semibold leading-tight">
            {probOver == null
              ? "Waiting"
              : isOver
              ? "Predict RED"
              : "Predict BLACK"}
          </span>
          {probOver != null && (
            <span className="text-[10px] opacity-80">
              p {(probOver * 100).toFixed(1)}%
            </span>
          )}
        </div>
      </div>
    );
  }
  if (mode === "roulette" && rouletteVariant === "dozens") {
    return (
      <div className="flex items-stretch gap-3 mt-2">
        <div className="flex gap-3 flex-1">
          <button
            onClick={onDozen1}
            disabled={disabled}
            className="w-full px-3 py-3 rounded bg-indigo-900/40 border border-indigo-600 hover:border-indigo-400 hover:text-indigo-300 text-[11px] disabled:opacity-40 transition-colors"
          >
            1st 12
          </button>
          <button
            onClick={onDozen2}
            disabled={disabled}
            className="w-full px-3 py-3 rounded bg-violet-900/40 border border-violet-600 hover:border-violet-400 hover:text-violet-300 text-[11px] disabled:opacity-40 transition-colors"
          >
            2nd 12
          </button>
          <button
            onClick={onDozen3}
            disabled={disabled}
            className="w-full px-3 py-3 rounded bg-fuchsia-900/40 border border-fuchsia-600 hover:border-fuchsia-400 hover:text-fuchsia-300 text-[11px] disabled:opacity-40 transition-colors"
          >
            3rd 12
          </button>
        </div>
        {/* Prediction badge not shown for dozens (multi-class) using binary prob currently */}
      </div>
    );
  }
  return (
    <div className="flex items-stretch gap-3 mt-2">
      <div className="flex gap-3 flex-1">
        <button
          onClick={onOver}
          disabled={disabled}
          className="w-full px-4 py-3 rounded bg-slate-800 border border-slate-600 hover:border-teal-400 hover:text-teal-300 text-sm disabled:opacity-40 transition-colors"
        >
          Over
        </button>
        <button
          onClick={onBelow}
          disabled={disabled}
          className="w-full px-4 py-3 rounded bg-slate-800 border border-slate-600 hover:border-rose-400 hover:text-rose-300 text-sm disabled:opacity-40 transition-colors"
        >
          Under
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
