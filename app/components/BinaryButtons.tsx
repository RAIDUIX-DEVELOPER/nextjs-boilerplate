"use client";
import React from "react";

export const BinaryButtons: React.FC<{
  onBelow: () => void;
  onOver: () => void;
  disabled?: boolean;
}> = ({ onBelow, onOver, disabled }) => {
  return (
    <div className="flex gap-3 mt-2">
      <button
        onClick={onOver}
        disabled={disabled}
        className="flex-1 px-4 py-3 rounded bg-slate-800 border border-slate-600 hover:border-teal-400 hover:text-teal-300 text-sm disabled:opacity-40 transition-colors"
      >
        Over 2
      </button>
      <button
        onClick={onBelow}
        disabled={disabled}
        className="flex-1 px-4 py-3 rounded bg-slate-800 border border-slate-600 hover:border-rose-400 hover:text-rose-300 text-sm disabled:opacity-40 transition-colors"
      >
        Below 2
      </button>
    </div>
  );
};
