"use client";
import React from "react";

interface ControlPanelProps {
  controls: any;
  setControls: (c: Partial<any>) => void;
  rlWeights: {
    model: number;
    markov: number;
    streak: number;
    pattern: number;
    ewma: number;
    bayes?: number;
    entropyMod?: number;
  };
  resetModel: () => void;
  loading: boolean;
  saveNow?: () => void;
  remoteState?: { status: string; lastPush?: number; lastPull?: number };
}

const sliderStyle = "w-full cursor-pointer accent-teal-400";
const numberBox =
  "w-16 bg-slate-900/70 border border-slate-600 rounded px-1.5 py-1 text-[10px]";
const groupCard =
  "rounded border border-slate-700/70 bg-slate-800/40 p-3 flex flex-col gap-3";
const labelRow = "flex items-center gap-2 text-[11px]";

export function ControlPanel({
  controls,
  setControls,
  rlWeights,
  resetModel,
  loading,
  saveNow,
  remoteState,
}: ControlPanelProps) {
  const Row = ({
    label,
    children,
    hint,
  }: {
    label: string;
    children: React.ReactNode;
    hint: string;
  }) => (
    <div className="flex flex-col gap-0.5">
      <label className={labelRow}>
        <span className="flex-1 text-slate-300 select-none">{label}</span>
        {children}
      </label>
      <p className="text-[10px] text-slate-500 leading-snug">{hint}</p>
    </div>
  );
  return (
    <div className="grid gap-3 md:grid-cols-3 text-[11px]">
      <div className={groupCard}>
        <h3 className="text-xs font-semibold text-slate-200 tracking-wide">
          Core
        </h3>
        <Row
          label="Threshold"
          hint="Decision boundary fixed at 0.5 (locked to avoid accuracy skew)."
        >
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={controls.threshold}
            disabled
            className={sliderStyle}
            name="threshold"
            id="threshold-range"
          />
          <input
            type="number"
            step={0.01}
            min={0}
            max={1}
            value={controls.threshold}
            disabled
            className={numberBox}
            name="thresholdValue"
            id="threshold-number"
          />
        </Row>
        <Row
          label="EWMA α"
          hint="Smoothing factor for long‑term baseline; higher = faster adaptation."
        >
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={controls.ewmaAlpha}
            onChange={(e) =>
              setControls({ ewmaAlpha: parseFloat(e.target.value) })
            }
            className={sliderStyle}
            name="ewmaAlpha"
            id="ewmaAlpha-range"
          />
          <input
            type="number"
            step={0.01}
            min={0}
            max={1}
            value={controls.ewmaAlpha}
            onChange={(e) =>
              setControls({ ewmaAlpha: parseFloat(e.target.value) || 0 })
            }
            className={numberBox}
            name="ewmaAlphaValue"
            id="ewmaAlpha-number"
          />
        </Row>
        <Row
          label="RL η"
          hint="Learning rate for adaptive (reinforcement) weight updates."
        >
          <input
            type="range"
            min={0}
            max={0.5}
            step={0.01}
            value={controls.rlEta}
            onChange={(e) => setControls({ rlEta: parseFloat(e.target.value) })}
            className={sliderStyle}
            name="rlEta"
            id="rlEta-range"
          />
          <input
            type="number"
            step={0.01}
            min={0}
            max={0.5}
            value={controls.rlEta}
            onChange={(e) =>
              setControls({ rlEta: parseFloat(e.target.value) || 0 })
            }
            className={numberBox}
            name="rlEtaValue"
            id="rlEta-number"
          />
        </Row>
        <Row
          label="RL Enabled"
          hint="Toggle automatic reward/punish adjustment of source weights."
        >
          <input
            type="checkbox"
            checked={controls.rlEnabled}
            onChange={(e) => setControls({ rlEnabled: e.target.checked })}
            name="rlEnabled"
            id="rlEnabled-checkbox"
          />
        </Row>
        <Row
          label="Anti-Streak"
          hint="Heuristic dampens continuation bias after long runs (>=5)."
        >
          <input
            type="checkbox"
            checked={controls.antiStreak}
            onChange={(e) => setControls({ antiStreak: e.target.checked })}
            name="antiStreak"
            id="antiStreak-checkbox"
          />
        </Row>
        <Row
          label="Invert"
          hint="Flip final probability (contrarian strategy)."
        >
          <input
            type="checkbox"
            checked={controls.invertStrategy}
            onChange={(e) => setControls({ invertStrategy: e.target.checked })}
            name="invertStrategy"
            id="invertStrategy-checkbox"
          />
        </Row>
        <Row
          label="Roulette Mode"
          hint="Switch UI to Roulette inputs (Red/Black or Dozens modes)."
        >
          <input
            type="checkbox"
            checked={controls.mode === "roulette"}
            onChange={(e) =>
              setControls({ mode: e.target.checked ? "roulette" : "binary" })
            }
            name="rouletteMode"
            id="rouletteMode-checkbox"
          />
        </Row>
        {controls.mode === "roulette" && (
          <Row
            label="Variant"
            hint="Choose Red/Black (binary) or Dozens (1st/2nd/3rd 12)."
          >
            <select
              value={controls.rouletteVariant}
              onChange={(e) => setControls({ rouletteVariant: e.target.value })}
              className="bg-slate-900/70 border border-slate-600 rounded px-2 py-1 text-[11px]"
              name="rouletteVariant"
              id="rouletteVariant-select"
            >
              <option value="redblack">Red/Black</option>
              <option value="dozens">Dozens</option>
            </select>
          </Row>
        )}
        <button
          onClick={resetModel}
          disabled={loading}
          className="mt-1 text-xs px-2 py-1.5 rounded border border-slate-600 text-slate-300 hover:border-teal-400 hover:text-teal-300 disabled:opacity-40"
        >
          Reset
        </button>
        <div className="mt-2 flex items-center gap-2">
          <button
            onClick={() => saveNow && saveNow()}
            disabled={loading || !saveNow}
            className="text-xs px-2 py-1.5 rounded border border-teal-600/60 text-teal-300 hover:border-teal-400 hover:text-teal-200 disabled:opacity-40"
            title="Force immediate persist to local + remote store"
          >
            Save Now
          </button>
          {remoteState && (
            <span className="text-[9px] text-slate-500">
              {remoteState.status === "loading" && "syncing..."}
              {remoteState.status === "ready" && (
                <>
                  last push{" "}
                  {remoteState.lastPush
                    ? Math.round((Date.now() - remoteState.lastPush) / 1000) +
                      "s"
                    : "—"}{" "}
                  ago
                </>
              )}
              {remoteState.status === "error" && "sync error"}
              {remoteState.status === "disabled" && "remote disabled"}
            </span>
          )}
        </div>
      </div>
      <div className={groupCard}>
        <h3 className="text-xs font-semibold text-slate-200 tracking-wide">
          Manual Weights
        </h3>
        <Row
          label="Model"
          hint="Manual emphasis for neural network probability."
        >
          <input
            type="range"
            min={0}
            max={5}
            step={0.1}
            value={controls.wModel}
            onChange={(e) =>
              setControls({ wModel: parseFloat(e.target.value) })
            }
            className={sliderStyle}
            name="wModel"
            id="wModel-range"
          />
          <input
            type="number"
            step={0.1}
            min={0}
            max={5}
            value={controls.wModel}
            onChange={(e) =>
              setControls({ wModel: parseFloat(e.target.value) || 0 })
            }
            className={numberBox}
            name="wModelValue"
            id="wModel-number"
          />
        </Row>
        <Row
          label="Markov"
          hint="Weight for 1-step transition (last state -> next) probability."
        >
          <input
            type="range"
            min={0}
            max={5}
            step={0.1}
            value={controls.wMarkov}
            onChange={(e) =>
              setControls({ wMarkov: parseFloat(e.target.value) })
            }
            className={sliderStyle}
            name="wMarkov"
            id="wMarkov-range"
          />
          <input
            type="number"
            step={0.1}
            min={0}
            max={5}
            value={controls.wMarkov}
            onChange={(e) =>
              setControls({ wMarkov: parseFloat(e.target.value) || 0 })
            }
            className={numberBox}
            name="wMarkovValue"
            id="wMarkov-number"
          />
        </Row>
        <Row
          label="Streak"
          hint="Weight for run-length (continuation vs reversal) posterior."
        >
          <input
            type="range"
            min={0}
            max={5}
            step={0.1}
            value={controls.wStreak}
            onChange={(e) =>
              setControls({ wStreak: parseFloat(e.target.value) })
            }
            className={sliderStyle}
            name="wStreak"
            id="wStreak-range"
          />
          <input
            type="number"
            step={0.1}
            min={0}
            max={5}
            value={controls.wStreak}
            onChange={(e) =>
              setControls({ wStreak: parseFloat(e.target.value) || 0 })
            }
            className={numberBox}
            name="wStreakValue"
            id="wStreak-number"
          />
        </Row>
        <Row
          label="Pattern"
          hint="Weight for run-length distribution continuation probability."
        >
          <input
            type="range"
            min={0}
            max={5}
            step={0.1}
            value={controls.wPattern}
            onChange={(e) =>
              setControls({ wPattern: parseFloat(e.target.value) })
            }
            className={sliderStyle}
            name="wPattern"
            id="wPattern-range"
          />
          <input
            type="number"
            step={0.1}
            min={0}
            max={5}
            value={controls.wPattern}
            onChange={(e) =>
              setControls({ wPattern: parseFloat(e.target.value) || 0 })
            }
            className={numberBox}
            name="wPatternValue"
            id="wPattern-number"
          />
        </Row>
        <Row
          label="EWMA"
          hint="Weight for baseline exponential moving average probability."
        >
          <input
            type="range"
            min={0}
            max={5}
            step={0.1}
            value={controls.wEwma}
            onChange={(e) => setControls({ wEwma: parseFloat(e.target.value) })}
            className={sliderStyle}
            name="wEwma"
            id="wEwma-range"
          />
          <input
            type="number"
            step={0.1}
            min={0}
            max={5}
            value={controls.wEwma}
            onChange={(e) =>
              setControls({ wEwma: parseFloat(e.target.value) || 0 })
            }
            className={numberBox}
            name="wEwmaValue"
            id="wEwma-number"
          />
        </Row>
        <Row
          label="Bayes"
          hint="Weight for global Beta posterior (overall Over rate with prior)."
        >
          <input
            type="range"
            min={0}
            max={5}
            step={0.1}
            value={controls.wBayes}
            onChange={(e) =>
              setControls({ wBayes: parseFloat(e.target.value) })
            }
            className={sliderStyle}
            name="wBayes"
            id="wBayes-range"
          />
          <input
            type="number"
            step={0.1}
            min={0}
            max={5}
            value={controls.wBayes}
            onChange={(e) =>
              setControls({ wBayes: parseFloat(e.target.value) || 0 })
            }
            className={numberBox}
            name="wBayesValue"
            id="wBayes-number"
          />
        </Row>
        <Row
          label="Entropy"
          hint="Weight for entropy-based modulation (recent stability tilt)."
        >
          <input
            type="range"
            min={0}
            max={5}
            step={0.1}
            value={controls.wEntropy}
            onChange={(e) =>
              setControls({ wEntropy: parseFloat(e.target.value) })
            }
            className={sliderStyle}
            name="wEntropy"
            id="wEntropy-range"
          />
          <input
            type="number"
            step={0.1}
            min={0}
            max={5}
            value={controls.wEntropy}
            onChange={(e) =>
              setControls({ wEntropy: parseFloat(e.target.value) || 0 })
            }
            className={numberBox}
            name="wEntropyValue"
            id="wEntropy-number"
          />
        </Row>
        <Row
          label="Beta Prior a,b"
          hint="Pseudo-counts for Bayesian Over rate (a=prior overs, b=prior unders)."
        >
          <div className="flex items-center gap-1">
            <input
              type="number"
              min={0}
              step={0.5}
              value={controls.bayesPriorA}
              onChange={(e) =>
                setControls({ bayesPriorA: parseFloat(e.target.value) || 0 })
              }
              className={numberBox}
              name="bayesPriorA"
              id="bayesPriorA"
            />
            <span className="text-slate-500">/</span>
            <input
              type="number"
              min={0}
              step={0.5}
              value={controls.bayesPriorB}
              onChange={(e) =>
                setControls({ bayesPriorB: parseFloat(e.target.value) || 0 })
              }
              className={numberBox}
              name="bayesPriorB"
              id="bayesPriorB"
            />
          </div>
        </Row>
        <Row
          label="Entropy Win"
          hint="Window size for recent entropy (stability) measurement."
        >
          <input
            type="range"
            min={5}
            max={100}
            step={1}
            value={controls.entropyWindow}
            onChange={(e) =>
              setControls({ entropyWindow: parseInt(e.target.value, 10) })
            }
            className={sliderStyle}
            name="entropyWindow"
            id="entropyWindow-range"
          />
          <input
            type="number"
            min={5}
            max={100}
            step={1}
            value={controls.entropyWindow}
            onChange={(e) =>
              setControls({ entropyWindow: parseInt(e.target.value, 10) || 5 })
            }
            className={numberBox}
            name="entropyWindowValue"
            id="entropyWindow-number"
          />
        </Row>
        <p className="text-[10px] text-slate-500 leading-snug">
          Manual weights are multiplied by adaptive RL weights (if enabled) then
          normalized.
        </p>
      </div>
      <div className={groupCard}>
        <h3 className="text-xs font-semibold text-slate-200 tracking-wide">
          Adaptive (RL) Weights
        </h3>
        <div className="grid grid-cols-2 gap-2 text-[10px]">
          <span className="text-slate-400">Model:</span>
          <span className="text-teal-300 tabular-nums">
            {rlWeights.model.toFixed(2)}
          </span>
          <span className="text-slate-400">Markov:</span>
          <span className="text-teal-300 tabular-nums">
            {rlWeights.markov.toFixed(2)}
          </span>
          <span className="text-slate-400">Streak:</span>
          <span className="text-teal-300 tabular-nums">
            {rlWeights.streak.toFixed(2)}
          </span>
          <span className="text-slate-400">Pattern:</span>
          <span className="text-teal-300 tabular-nums">
            {rlWeights.pattern.toFixed(2)}
          </span>
          <span className="text-slate-400">EWMA:</span>
          <span className="text-teal-300 tabular-nums">
            {rlWeights.ewma.toFixed(2)}
          </span>
          {rlWeights.bayes != null && (
            <>
              <span className="text-slate-400">Bayes:</span>
              <span className="text-teal-300 tabular-nums">
                {rlWeights.bayes.toFixed(2)}
              </span>
            </>
          )}
          {rlWeights.entropyMod != null && (
            <>
              <span className="text-slate-400">Entropy:</span>
              <span className="text-teal-300 tabular-nums">
                {rlWeights.entropyMod.toFixed(2)}
              </span>
            </>
          )}
        </div>
        <div className="text-[10px] text-slate-500 leading-snug mt-1">
          Updated after each round using multiplicative rewards; centered around
          1.
        </div>
      </div>
    </div>
  );
}
