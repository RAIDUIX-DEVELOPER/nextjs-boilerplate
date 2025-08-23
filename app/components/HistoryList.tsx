"use client";
import React from "react";

export const HistoryList: React.FC<{ history: number[] }> = ({ history }) => {
  return (
    <div className="mt-6">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-slate-300">History</h3>
        <span className="text-[10px] text-slate-500 uppercase tracking-wide">
          {history.length}
        </span>
      </div>
      <div className="h-56 overflow-y-auto rounded border border-slate-700 bg-slate-900/40 text-[11px] font-mono text-slate-300 divide-y divide-slate-800">
        {history.length === 0 && (
          <div className="p-3 text-slate-600">No values yet.</div>
        )}
        {history
          .slice()
          .reverse()
          .map((v, i) => (
            <div key={i} className="px-3 py-1 flex justify-between">
              <span className="text-slate-500">{history.length - i}</span>
              <span>{v}</span>
            </div>
          ))}
      </div>
    </div>
  );
};

export default HistoryList;
