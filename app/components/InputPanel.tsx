"use client";
import React, { useState, useRef, useEffect } from "react";

interface Props {
  onSubmit: (v: number) => void;
  disabled?: boolean;
}

export const InputPanel: React.FC<Props> = ({ onSubmit, disabled }) => {
  const [val, setVal] = useState("");
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!disabled) {
      inputRef.current?.focus();
    }
  }, [disabled]);
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const f = parseFloat(val);
    if (!isNaN(f)) {
      onSubmit(f);
      setVal("");
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  };
  return (
    <form onSubmit={submit} className="space-y-3">
      <label className="block text-xs uppercase tracking-wide text-slate-400">
        Enter Value
      </label>
      <div className="flex gap-2">
        <input
          ref={inputRef}
          type="number"
          step="0.01"
          value={val}
          onChange={(e) => setVal(e.target.value)}
          disabled={disabled}
          placeholder="0.00"
          className="flex-1 bg-slate-800/60 border border-slate-600 focus:border-teal-400 focus:ring-0 rounded px-3 py-2 text-slate-100 placeholder-slate-500 text-sm outline-none"
        />
        <button
          disabled={disabled || val === ""}
          className="px-4 py-2 rounded bg-gradient-to-r from-teal-500 to-cyan-500 text-slate-900 font-medium text-sm disabled:opacity-40 disabled:cursor-not-allowed shadow-sm hover:from-teal-400 hover:to-cyan-400 transition-colors"
        >
          Train
        </button>
      </div>
    </form>
  );
};

export default InputPanel;
