"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgpu";
import "@tensorflow/tfjs-backend-webgl";

interface ProbParts {
  model: number;
  markov: number;
  streak: number;
  pattern: number;
  ewma: number;
  bayes: number;
  entropyMod: number;
}

export function useBinaryModel() {
  // === Persistence Keys / Version ===
  const PERSIST_VERSION = 1;
  const PKEY = (k: string) => `rbm_${PERSIST_VERSION}_${k}`; // roulette/binary model namespace
  const PERSIST_LIMIT = 5000; // max spins to persist per history (raised per requirement)
  const persistThrottleMs = 2500;
  const lastPersistRef = useRef<number>(0);
  const windowSize = 24;
  // Expanded feature set: original 8 + avg OVER run length + avg BELOW run length (normalized)
  const featureCount = 10;
  const epochs = 1; // reduced for perf
  const batchSize = 32;
  const TRAIN_START = 10; // lowered (was 50) to begin CNN training earlier
  const TRAIN_EVERY = 3; // train every 3rd eligible sample
  const [history, setHistory] = useState<number[]>([]);
  interface OutcomeRecord {
    label: 0 | 1;
    predLabel: 0 | 1 | null;
    prob: number | null;
    rawProb?: number | null;
    thresholdUsed?: number;
    invertUsed?: boolean;
    correct: boolean | null;
  }

  const [records, setRecords] = useState<OutcomeRecord[]>([]);

  // === Base binary model refs (restored shapes) ===
  const modelRef = useRef<tf.LayersModel | null>(null);
  const lastPredictionRef = useRef<any>(null); // heterogeneous object across lifecycle
  const markovCountsRef = useRef<number[][]>([
    [1, 1],
    [1, 1],
  ]); // transition counts prev -> current (below/over)
  const streakPosteriorsRef = useRef<
    Record<string, { cont: number; rev: number }>
  >({});
  const ewmaRef = useRef<number | null>(null);
  const probPartsRef = useRef<ProbParts | null>(null);
  const rlWeightsRef = useRef<{
    markov: number;
    streak: number;
    pattern: number;
    ewma: number;
    bayes: number;
    entropyMod: number;
    model: number;
  }>({
    markov: 1,
    streak: 1,
    pattern: 1,
    ewma: 1,
    bayes: 1,
    entropyMod: 1,
    model: 1,
  });
  const accRef = useRef<{ correct: number; total: number }>({
    correct: 0,
    total: 0,
  });
  // Dozens tracking state we reference later
  const [dozenRecords, setDozenRecords] = useState<
    {
      label: number;
      predLabel: number | null;
      probs: [number, number, number];
      correct: boolean | null;
    }[]
  >([]);
  const [dozenVersion, setDozenVersion] = useState(0);
  const dozenHiProbMissRateRef = useRef(0);
  const dozenLastHiProbMissRef = useRef(false);
  const dozenAbstainRef = useRef(false);
  const [probOver, setProbOver] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [backend, setBackend] = useState<string | null>(null);
  // Remote persistence (Edge Config via /api/state) control refs & helpers (placed after state declarations)
  const remoteStateRef = useRef<{
    status: "idle" | "loading" | "ready" | "error";
    lastPull?: number;
    lastPush?: number;
    error?: string;
  }>({ status: "idle" });
  // Backend readiness control to prevent model building before backend initialized
  const backendReadyRef = useRef(false);
  const backendInitPromiseRef = useRef<Promise<void> | null>(null);
  const ensureBackend = useCallback(async () => {
    if (backendReadyRef.current && backend) return;
    if (backendInitPromiseRef.current) {
      await backendInitPromiseRef.current;
      return;
    }
    backendInitPromiseRef.current = (async () => {
      await tf.ready();
      let selected = tf.getBackend();
      if (!selected || selected === "cpu") {
        try {
          const hasNavigatorGPU =
            typeof navigator !== "undefined" && (navigator as any).gpu;
          if (
            hasNavigatorGPU &&
            (tf as any).engine()?.registryFactory?.["webgpu"]
          ) {
            try {
              const adapter = await (navigator as any).gpu.requestAdapter?.();
              if (
                adapter &&
                typeof (adapter as any).requestAdapterInfo === "function"
              ) {
                await tf.setBackend("webgpu");
                selected = tf.getBackend();
              } else {
                await tf.setBackend("webgl");
                selected = tf.getBackend();
              }
            } catch (e) {
              if (console && console.warn)
                console.warn("WebGPU init failed, fallback to webgl", e);
              await tf.setBackend("webgl");
              selected = tf.getBackend();
            }
          } else {
            await tf.setBackend("webgl");
            selected = tf.getBackend();
          }
        } catch {
          // fallback remains whatever tf chose
          selected = tf.getBackend();
        }
      }
      setBackend(selected);
      backendReadyRef.current = true;
      if (console && console.log) console.log("TF backend ready:", selected);
    })();
    await backendInitPromiseRef.current;
  }, [backend]);
  // Roulette dozens mode state (separate simple frequency tracker)
  const [dozenHistory, setDozenHistory] = useState<(0 | 1 | 2)[]>([]);
  const dozenProbsRef = useRef<[number, number, number] | null>(null);
  // Multi-class (dozens) model & tracking
  const dozenModelRef = useRef<tf.LayersModel | null>(null);
  const dozenPredictionRef = useRef<{
    probs: [number, number, number] | null;
    predLabel: 0 | 1 | 2 | null;
  }>({ probs: null, predLabel: null });
  // Adaptive metrics (exposed to UI for diagnostics)
  const dozenAdaptiveRef = useRef<{
    cap: number | null;
    floor: number | null;
    shrink: number | null;
  }>({
    cap: null,
    floor: null,
    shrink: null,
  });
  // Brier auto-shrink escalation state (tracks severity 0 none, 1 mild, 2 pronounced)
  const dozenBrierAutoRef = useRef<{ level: 0 | 1 | 2 }>({ level: 0 });
  // Progressive cap decay & shock handling
  const dozenCapDecayRef = useRef<{ mult: number }>({ mult: 1 });
  // Track hi-prob prediction outcomes (window) & shock state scaffold
  const dozenHiProbEventsRef = useRef<number[]>([]); // 1 = miss, 0 = hit for qualifying hi-prob predictions
  // === New adaptive refs (recency reliability, confusion, opportunity, etc.) ===
  const dozenReliabilityEwmaRef = useRef<{
    ema: [number, number, number];
    total: number;
  }>({ ema: [1 / 3, 1 / 3, 1 / 3], total: 0 });
  const dozenReliabilityVarRef = useRef<[number, number, number]>([0, 0, 0]);
  const dozenConfusionWindowRef = useRef<{ a: number; p: number | null }[]>([]); // last 60 predictions: actual a, predicted p
  const dozenOpportunityRef = useRef<{
    topPickCounts: number[];
    history: number[];
  }>({ topPickCounts: [0, 0, 0], history: [] });
  const dozenProbHistoryRef = useRef<[number[], number[], number[]]>([
    [],
    [],
    [],
  ]); // per-class predicted prob history (last 120)
  const dozenFalseNegStreakRef = useRef<[number, number, number]>([0, 0, 0]);
  const dozenPerSourceClassStatsRef = useRef<
    Record<string, { perClass: { n: number; reward: number }[] }>
  >({});
  const dozenGapHazardRef = useRef<{
    gapSince: [number, number, number];
    gapAppear: Record<string, number>;
    gapTotals: Record<string, number>;
  }>({ gapSince: [0, 0, 0], gapAppear: {}, gapTotals: {} });
  const dozenAbstainStatsRef = useRef<{
    qHist: number[];
    abstains: number;
    decided: number;
    wins: number;
  }>({ qHist: [], abstains: 0, decided: 0, wins: 0 });
  const dozenPlateauRef = useRef<{
    last: [number, number, number][];
    jitterEvents: number;
  }>({ last: [], jitterEvents: 0 });
  const dozenShockLevelRef = useRef<number>(0); // escalates 0-3
  const dozenHiProbMissWindowRef = useRef<boolean[]>([]); // hi-prob misses
  const dozenRunPostRef = useRef<Record<string, { cont: number; rev: number }>>(
    {}
  );
  const dozenShockRef = useRef<
    | { active: number }
    | {
        active: boolean;
        ttl: number; // remaining spins in shock
        hits: number; // hi-prob hits while in shock
        consec: number; // consecutive hi-prob misses (outside/in shock)
        severity: number; // last trigger severity 0..1
      }
  >({ active: 0 });
  // Normalize legacy shape to extended structure lazily
  if (
    "active" in dozenShockRef.current &&
    typeof (dozenShockRef.current as any).active === "number"
  ) {
    const legacy = dozenShockRef.current as { active: number };
    if (legacy.active > 0) {
      dozenShockRef.current = {
        active: true,
        ttl: legacy.active,
        hits: 0,
        consec: 0,
        severity: 0,
      };
    } else {
      dozenShockRef.current = {
        active: false,
        ttl: 0,
        hits: 0,
        consec: 0,
        severity: 0,
      };
    }
  }
  // Alternation detector
  const dozenAlternationRef = useRef<{ active: boolean }>({ active: false });
  const biasOverrideRef = useRef(false);
  const biasStableNeutralRef = useRef(0);
  const biasConfigRef = useRef({ window: 20, threshold: 0.3 });
  interface Metrics {
    longestWin: number;
    longestLose: number;
    currentWin: number;
    currentLose: number;
    overPredSuccess: number;
    overPredTotal: number;
    sumWin: number; // total length of completed winning streaks
    countWin: number; // number of completed winning streaks
    sumLose: number; // total length of completed losing streaks
    countLose: number; // number of completed losing streaks
    sumActualRuns: number; // total length of completed actual outcome runs
    countActualRuns: number; // number of completed actual outcome runs
    sumOverRuns: number; // total length of completed OVER (1) runs
    countOverRuns: number; // number of completed OVER runs
    sumBelowRuns: number; // total length of completed BELOW (0) runs
    countBelowRuns: number; // number of completed BELOW runs
  }
  const [metrics, setMetrics] = useState<Metrics>({
    longestWin: 0,
    longestLose: 0,
    currentWin: 0,
    currentLose: 0,
    overPredSuccess: 0,
    overPredTotal: 0,
    sumWin: 0,
    countWin: 0,
    sumLose: 0,
    countLose: 0,
    sumActualRuns: 0,
    countActualRuns: 0,
    sumOverRuns: 0,
    countOverRuns: 0,
    sumBelowRuns: 0,
    countBelowRuns: 0,
  });

  // === Bias / Regime Detection Config ===
  const BIAS_WINDOW = 40; // window for bias & run-length regime detection
  const RUNLEN_EMA_ALPHA = 0.03;
  const REGIME_DAMPEN_MAX = 0.75; // max shrink of (patternP-0.5)
  const BIAS_GAP_LOGIT_GAIN = 0.85; // scale for bias gap corrective logit shift

  // Regime & bias tracking (separate from existing bias override mechanism)
  const biasRegimeRef = useRef({
    recent: [] as { predOver: number; actualOver: number }[],
    predOverCount: 0,
    actualOverCount: 0,
    avgRunLenEma: 0,
    lastAvgRunLen: 0,
    regimeShift: 0,
    biasGap: 0,
    patternDampen: 1,
  });

  const computeRecentAverageRunLength = (sequence: number[]): number => {
    if (sequence.length === 0) return 0;
    const slice = sequence.slice(-BIAS_WINDOW);
    const runs: number[] = [];
    let cur = 1;
    for (let i = 1; i < slice.length; i++) {
      if (slice[i] === slice[i - 1]) cur++;
      else {
        runs.push(cur);
        cur = 1;
      }
    }
    runs.push(cur);
    return runs.reduce((a, c) => a + c, 0) / runs.length;
  };

  const applyBiasGapCorrection = (p: number): number => {
    const b = biasRegimeRef.current;
    const gap = b.biasGap;
    if (Math.abs(gap) < 0.05) return p; // small gaps ignored
    const clip = (x: number) => Math.min(1 - 1e-6, Math.max(1e-6, x));
    const pc = clip(p);
    const logit = Math.log(pc / (1 - pc));
    const corrected = logit - BIAS_GAP_LOGIT_GAIN * gap;
    return 1 / (1 + Math.exp(-corrected));
  };

  const buildModel = () => {
    // Leaner model to reduce per-fit latency & overfitting risk on tiny online batches
    const m = tf.sequential();
    m.add(tf.layers.inputLayer({ inputShape: [windowSize, featureCount] }));
    m.add(
      tf.layers.conv1d({
        filters: 24,
        kernelSize: 3,
        activation: "relu",
        padding: "same",
      })
    );
    m.add(
      tf.layers.conv1d({
        filters: 16,
        kernelSize: 5,
        activation: "relu",
        padding: "same",
      })
    );
    m.add(tf.layers.dropout({ rate: 0.15 }));
    m.add(tf.layers.globalAveragePooling1d());
    m.add(
      tf.layers.dense({
        units: 24,
        activation: "relu",
        kernelRegularizer: tf.regularizers.l2({ l2: 1e-4 }),
      })
    );
    m.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
    m.compile({ optimizer: tf.train.adam(0.001), loss: "binaryCrossentropy" });
    return m;
  };
  const ensureModel = useCallback(async () => {
    await ensureBackend();
    if (modelRef.current) {
      // Guard against hot-reload mismatch when featureCount changes
      const curShape = modelRef.current.inputs?.[0]?.shape;
      if (curShape && curShape[2] !== featureCount) {
        modelRef.current.dispose();
        modelRef.current = null;
      }
    }
    if (!modelRef.current) modelRef.current = buildModel();
    return modelRef.current;
  }, [featureCount, ensureBackend]);

  // === Dozens (multi-class) model helpers ===
  const dozenFeatureCount = 11; // one-hot(3) + runLen + transition + freq(3) + timeSince(3)
  const buildDozenModel = () => {
    const m = tf.sequential();
    m.add(
      tf.layers.inputLayer({ inputShape: [windowSize, dozenFeatureCount] })
    );
    m.add(
      tf.layers.conv1d({
        filters: 24,
        kernelSize: 3,
        activation: "relu",
        padding: "same",
        kernelRegularizer: tf.regularizers.l2({ l2: 1e-4 }),
      })
    );
    m.add(
      tf.layers.conv1d({
        filters: 16,
        kernelSize: 5,
        activation: "relu",
        padding: "same",
        kernelRegularizer: tf.regularizers.l2({ l2: 1e-4 }),
      })
    );
    m.add(tf.layers.dropout({ rate: 0.15 }));
    m.add(tf.layers.globalAveragePooling1d());
    m.add(
      tf.layers.dense({
        units: 24,
        activation: "relu",
        kernelRegularizer: tf.regularizers.l2({ l2: 1e-4 }),
      })
    );
    m.add(
      tf.layers.dense({
        units: 3,
        activation: "softmax",
        kernelRegularizer: tf.regularizers.l2({ l2: 1e-4 }),
      })
    );
    m.compile({
      optimizer: tf.train.adam(0.0007),
      loss: "categoricalCrossentropy",
    });
    return m;
  };
  const ensureDozenModel = useCallback(async () => {
    await ensureBackend();
    if (!dozenModelRef.current) dozenModelRef.current = buildDozenModel();
    return dozenModelRef.current;
  }, [ensureBackend]);

  // Restore persisted reliability & gap windows
  useEffect(() => {
    try {
      const persisted = localStorage.getItem("dz_rel_calib_v1");
      if (persisted) {
        const obj = JSON.parse(persisted);
        if (obj && Array.isArray(obj.mult) && obj.mult.length === 3) {
          dozenCalibAdjRef.current.mult = obj.mult.map((m: number) =>
            Math.min(1.6, Math.max(0.6, m))
          ) as [number, number, number];
        }
        if (typeof obj.temp === "number") {
          dozenCalibAdjRef.current.temp = Math.min(
            1.6,
            Math.max(0.7, obj.temp)
          );
        }
      }
      const topWin = localStorage.getItem("dz_topprob_win_v1");
      if (topWin) {
        const arr = JSON.parse(topWin);
        if (Array.isArray(arr)) {
          dozenTopProbWindowRef.current = arr
            .filter(
              (r: any) =>
                r && typeof r.p === "number" && typeof r.correct === "boolean"
            )
            .slice(-90);
        }
      }
    } catch {}
    const handleBeforeUnload = () => {
      try {
        localStorage.setItem(
          "dz_rel_calib_v1",
          JSON.stringify({
            mult: dozenCalibAdjRef.current.mult,
            temp: dozenCalibAdjRef.current.temp,
          })
        );
        localStorage.setItem(
          "dz_topprob_win_v1",
          JSON.stringify(dozenTopProbWindowRef.current.slice(-90))
        );
      } catch {}
    };
    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => window.removeEventListener("beforeunload", handleBeforeUnload);
  }, []);

  const buildDozenFeatureWindow = useCallback(
    (seq: (0 | 1 | 2)[]): number[][] => {
      const feats: number[][] = [];
      if (!seq.length) return feats;
      let runLen = 1;
      const timeSince = [0, 0, 0];
      let last = seq[0];
      const freqCounts = [0, 0, 0];
      for (let i = 0; i < seq.length; i++) {
        const v = seq[i];
        freqCounts[v] += 1;
        if (i > 0) {
          if (v === last) runLen += 1;
          else {
            runLen = 1;
            last = v;
          }
        }
        for (let k = 0; k < 3; k++) {
          if (k === v) timeSince[k] = 0;
          else timeSince[k] += 1;
        }
        const window10 = seq.slice(Math.max(0, i - 9), i + 1);
        const freq10 = [0, 0, 0];
        window10.forEach((x) => (freq10[x] += 1));
        const f10 = freq10.map((c) => c / window10.length);
        const oneHot = [v === 0 ? 1 : 0, v === 1 ? 1 : 0, v === 2 ? 1 : 0];
        feats.push([
          ...oneHot, // 3
          Math.min(runLen / 20, 1), // 1
          i > 0 && seq[i - 1] !== v ? 1 : 0, // 1
          ...f10, // 3
          Math.min(timeSince[0] / 20, 1),
          Math.min(timeSince[1] / 20, 1),
          Math.min(timeSince[2] / 20, 1), // 3
        ]);
      }
      return feats;
    },
    []
  );

  // Stable memoized feature builder (prevents effect dependency churn -> retrain loop)
  const buildFeatureWindow = useCallback((seq: number[]): number[][] => {
    const feats: number[][] = [];
    if (!seq.length) return feats;
    let streak = 0;
    let lastVal = seq[0];
    let timeSinceOver = 0;
    let timeSinceUnder = 0;
    // Running sums/counts of COMPLETED runs (separate for over/below)
    let sumOverRuns = 0;
    let countOverRuns = 0;
    let sumBelowRuns = 0;
    let countBelowRuns = 0;
    // Track previous index value to detect run boundaries for completed runs BEFORE current index i
    let runVal = seq[0];
    let runLen = 1;
    const recordCompletedRun = (val: number, len: number) => {
      if (val === 1) {
        sumOverRuns += len;
        countOverRuns += 1;
      } else {
        sumBelowRuns += len;
        countBelowRuns += 1;
      }
    };
    for (let i = 0; i < seq.length; i++) {
      const v = seq[i];
      if (i > 0) {
        if (v === runVal) runLen += 1;
        else {
          // previous run completed at i-1, record it (excluding current ongoing run from averages at this point)
          recordCompletedRun(runVal, runLen);
          runVal = v;
          runLen = 1;
        }
      }
      if (v === lastVal) streak += 1;
      else {
        streak = 1;
        lastVal = v;
      }
      if (v === 1) {
        timeSinceOver = 0;
        timeSinceUnder += 1;
      } else {
        timeSinceUnder = 0;
        timeSinceOver += 1;
      }
      const slice = (n: number) => seq.slice(Math.max(0, i - (n - 1)), i + 1);
      const roll = (arr: number[]) =>
        arr.reduce((a, b) => a + b, 0) / arr.length;
      const avgOver = countOverRuns
        ? sumOverRuns / countOverRuns
        : runVal === 1
        ? runLen
        : 0;
      const avgBelow = countBelowRuns
        ? sumBelowRuns / countBelowRuns
        : runVal === 0
        ? runLen
        : 0;
      feats.push([
        v, // current value
        Math.min(streak / 20, 1), // normalized current streak length
        i > 0 && seq[i - 1] !== v ? 1 : 0, // transition flag
        roll(slice(5)),
        roll(slice(10)),
        roll(slice(20)),
        Math.min(timeSinceOver / 20, 1),
        Math.min(timeSinceUnder / 20, 1),
        Math.min(avgOver / 20, 1), // NEW avg OVER run length normalized
        Math.min(avgBelow / 20, 1), // NEW avg BELOW run length normalized
      ]);
    }
    return feats;
  }, []);

  const controlsRef = useRef({
    threshold: 0.5,
    wModel: 1,
    wMarkov: 1,
    wStreak: 1,
    wPattern: 1,
    wEwma: 1,
    wBayes: 1,
    wEntropy: 1,
    ewmaAlpha: 0.1,
    rlEnabled: true,
    rlEta: 0.15,
    antiStreak: true,
    suggestedThreshold: 0.5,
    invertStrategy: false,
    bayesPriorA: 1,
    bayesPriorB: 1,
    entropyWindow: 20,
    mode: "roulette" as "binary" | "roulette", // default start mode switched to roulette
    rouletteVariant: "dozens" as "redblack" | "dozens", // default variant set to dozens
  });
  const [controlsVersion, setControlsVersion] = useState(0);

  const computeBayes = (a0: number, b0: number, seq: number[]) => {
    const sum = seq.reduce((a, b) => a + b, 0);
    return (a0 + sum) / (a0 + b0 + seq.length);
  };
  const computeEntropyMod = (seq: number[], last: number): number => {
    const entW = Math.max(5, controlsRef.current.entropyWindow);
    const recent = seq.slice(-entW);
    const p = recent.reduce((a, b) => a + b, 0) / recent.length;
    let entropy = 0;
    if (p > 0 && p < 1)
      entropy = -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p));
    const decisiveness = 1 - entropy; // 0..1
    let mod = 0.5 + (last === 1 ? 1 : -1) * decisiveness * 0.25;
    return Math.min(0.999999, Math.max(0.000001, mod));
  };

  const assemblePatternProb = (
    historyArr: number[],
    last: number,
    rl: number
  ) => {
    let runs: number[] = [];
    let cur = 1;
    for (let i = historyArr.length - 2; i >= 0; i--) {
      if (historyArr[i] === historyArr[i + 1]) cur++;
      else {
        runs.push(cur);
        cur = 1;
      }
    }
    runs.push(cur);
    if (!runs.length) return 0.5;
    const longer = runs.filter((r) => r > rl).length + 1;
    const totalRuns = runs.length + 2;
    const pCont = longer / totalRuns;
    return last === 1 ? pCont : 1 - pCont;
  };

  const blend = (probs: ProbParts, manual: any, dyn: any): number => {
    const clamp = (p: number) => Math.min(1 - 1e-6, Math.max(1e-6, p));
    const weightsRaw = [
      manual.wModel * dyn.model,
      manual.wMarkov * dyn.markov,
      manual.wStreak * dyn.streak,
      manual.wPattern * dyn.pattern,
      manual.wEwma * dyn.ewma,
      manual.wBayes * dyn.bayes,
      manual.wEntropy * dyn.entropyMod,
    ];
    const sumW = weightsRaw.reduce((a, b) => a + b, 0) || 1;
    const normW = weightsRaw.map((w) => w / sumW);
    const srcs = [
      probs.model,
      probs.markov,
      probs.streak,
      probs.pattern,
      probs.ewma,
      probs.bayes,
      probs.entropyMod,
    ].map(clamp);
    const logits = srcs.map((p) => Math.log(p / (1 - p)));
    let blendedLogit = 0;
    for (let i = 0; i < logits.length; i++)
      blendedLogit += logits[i] * normW[i];
    return 1 / (1 + Math.exp(-blendedLogit));
  };

  const recomputePendingRef = useRef(false);
  const lastFrameRef = useRef<number | null>(null);
  const recomputeCurrentProb = useCallback(async () => {
    // Throttle to one recompute per animation frame
    const nowFrame = typeof window !== "undefined" ? performance.now() : 0;
    if (lastFrameRef.current && nowFrame - lastFrameRef.current < 12) {
      if (!recomputePendingRef.current) {
        recomputePendingRef.current = true;
        requestAnimationFrame(() => {
          recomputePendingRef.current = false;
          recomputeCurrentProb();
        });
      }
      return;
    }
    lastFrameRef.current = nowFrame;
    if (recomputePendingRef.current) return; // simple debounce
    recomputePendingRef.current = true;
    if (history.length < 5) {
      lastPredictionRef.current = {
        rawProb: null,
        invertUsed: controlsRef.current.invertStrategy,
        threshold: controlsRef.current.threshold,
      };
      setProbOver(null);
      recomputePendingRef.current = false;
      return;
    }
    if (history.length <= windowSize) {
      const last = history[history.length - 1];
      const mc = markovCountsRef.current;
      const row = mc[last];
      const markovP = row[1] / (row[0] + row[1]);
      let rl = 1;
      for (let i = history.length - 2; i >= 0; i--) {
        if (history[i] === last) rl++;
        else break;
      }
      const key = `${last}:${rl}`;
      const rec = streakPosteriorsRef.current[key];
      let contP = 0.5;
      if (rec) contP = (rec.cont + 1) / (rec.cont + rec.rev + 2);
      const streakP = last === 1 ? contP : 1 - contP;
      const patternP = assemblePatternProb(history, last, rl);
      if (ewmaRef.current == null)
        ewmaRef.current = history.reduce((a, b) => a + b, 0) / history.length;
      const ewmaP = ewmaRef.current;
      const bayesP = computeBayes(
        controlsRef.current.bayesPriorA,
        controlsRef.current.bayesPriorB,
        history
      );
      const entropyMod = computeEntropyMod(history, last);
      // Regime update (uses previous probability before inversion)
      // Adjust patternP via regime detection (use history to compute averages)
      const recentAvg = computeRecentAverageRunLength(history);
      const br = biasRegimeRef.current;
      if (br.avgRunLenEma === 0) br.avgRunLenEma = recentAvg;
      else
        br.avgRunLenEma =
          br.avgRunLenEma + RUNLEN_EMA_ALPHA * (recentAvg - br.avgRunLenEma);
      br.lastAvgRunLen = recentAvg;
      let regimeShift = 0;
      if (br.avgRunLenEma > 0 && recentAvg < br.avgRunLenEma) {
        regimeShift = Math.min(
          1,
          (br.avgRunLenEma - recentAvg) / br.avgRunLenEma
        );
      }
      br.regimeShift = regimeShift;
      br.patternDampen = 1 - regimeShift * REGIME_DAMPEN_MAX;
      const dampPattern = 0.5 + (patternP - 0.5) * br.patternDampen;
      const probs: ProbParts = {
        model: 0.5,
        markov: markovP,
        streak: streakP,
        pattern: dampPattern,
        ewma: ewmaP,
        bayes: bayesP,
        entropyMod,
      };
      probPartsRef.current = probs;
      const dyn = controlsRef.current.rlEnabled
        ? rlWeightsRef.current
        : {
            model: 1,
            markov: 1,
            streak: 1,
            pattern: 1,
            ewma: 1,
            bayes: 1,
            entropyMod: 1,
          };
      let blended = blend(probs, controlsRef.current, dyn);
      // Update bias gap counters
      const thisPredLabel = blended >= 0.5 ? 1 : 0;
      const brr = biasRegimeRef.current;
      brr.recent.push({ predOver: thisPredLabel, actualOver: last });
      if (brr.recent.length > BIAS_WINDOW) brr.recent.shift();
      brr.predOverCount = brr.recent.reduce((a, r) => a + r.predOver, 0);
      brr.actualOverCount = brr.recent.reduce((a, r) => a + r.actualOver, 0);
      const len = brr.recent.length || 1;
      const predRate = brr.predOverCount / len;
      const actualRate = brr.actualOverCount / len;
      brr.biasGap = predRate - actualRate;
      blended = applyBiasGapCorrection(blended);
      if (controlsRef.current.antiStreak && rl >= 5) {
        const f = Math.min(0.4, (rl - 4) * 0.05);
        const lastIsOver = last === 1;
        const target = lastIsOver ? 0.3 : 0.7;
        if ((lastIsOver && blended > 0.55) || (!lastIsOver && blended < 0.45))
          blended = blended * (1 - f) + target * f;
      }
      if (biasOverrideRef.current) {
        const adjusted = [
          probs.model,
          probs.markov,
          probs.streak,
          probs.pattern,
          probs.ewma,
          probs.bayes,
          probs.entropyMod,
        ].map((p) => 0.75 * p + 0.25 * 0.5);
        const clamp = (p: number) => Math.min(1 - 1e-6, Math.max(1e-6, p));
        const logits = adjusted.map((p) => Math.log(clamp(p) / (1 - clamp(p))));
        const mean = logits.reduce((a, b) => a + b, 0) / logits.length;
        blended = 1 / (1 + Math.exp(-mean));
      }
      const rawProb = blended;
      const invertUsed = controlsRef.current.invertStrategy;
      const effective = invertUsed ? 1 - rawProb : rawProb;
      lastPredictionRef.current = {
        rawProb,
        effectiveProb: effective,
        invertUsed,
        threshold: controlsRef.current.threshold,
      };
      setProbOver(effective);
      recomputePendingRef.current = false;
      return;
    }
    if (!backendReadyRef.current) await ensureBackend();
    if (!backendReadyRef.current) {
      recomputePendingRef.current = false;
      return;
    }
    const m = await ensureModel();
    const { modelProb } = tf.tidy(() => {
      const lastWinRaw = history.slice(-windowSize);
      const feats = buildFeatureWindow(lastWinRaw);
      const lastTensor = tf.tensor3d([feats]);
      const modelProbT = m.predict(lastTensor) as tf.Tensor;
      const data = modelProbT.dataSync(); // sync small tensor read
      lastTensor.dispose();
      modelProbT.dispose();
      return { modelProb: data[0] };
    });
    const last = history[history.length - 1];
    const mc = markovCountsRef.current;
    const row = mc[last];
    const markovP = row[1] / (row[0] + row[1]);
    let rl = 1;
    for (let i = history.length - 2; i >= 0; i--) {
      if (history[i] === last) rl++;
      else break;
    }
    const key = `${last}:${rl}`;
    const rec = streakPosteriorsRef.current[key];
    let contP = 0.5;
    if (rec) contP = (rec.cont + 1) / (rec.cont + rec.rev + 2);
    const streakP = last === 1 ? contP : 1 - contP;
    const patternP = assemblePatternProb(history, last, rl);
    if (ewmaRef.current == null)
      ewmaRef.current = history.reduce((a, b) => a + b, 0) / history.length;
    const ewmaP = ewmaRef.current;
    const bayesP = computeBayes(
      controlsRef.current.bayesPriorA,
      controlsRef.current.bayesPriorB,
      history
    );
    const entropyMod = computeEntropyMod(history, last);
    const probs: ProbParts = {
      model: modelProb,
      markov: markovP,
      streak: streakP,
      pattern: (() => {
        const recentAvg = computeRecentAverageRunLength(history);
        const br = biasRegimeRef.current;
        if (br.avgRunLenEma === 0) br.avgRunLenEma = recentAvg;
        else
          br.avgRunLenEma =
            br.avgRunLenEma + RUNLEN_EMA_ALPHA * (recentAvg - br.avgRunLenEma);
        br.lastAvgRunLen = recentAvg;
        let regimeShift = 0;
        if (br.avgRunLenEma > 0 && recentAvg < br.avgRunLenEma) {
          regimeShift = Math.min(
            1,
            (br.avgRunLenEma - recentAvg) / br.avgRunLenEma
          );
        }
        br.regimeShift = regimeShift;
        br.patternDampen = 1 - regimeShift * REGIME_DAMPEN_MAX;
        return 0.5 + (patternP - 0.5) * br.patternDampen;
      })(),
      ewma: ewmaP,
      bayes: bayesP,
      entropyMod,
    };
    probPartsRef.current = probs;
    const dyn = controlsRef.current.rlEnabled
      ? rlWeightsRef.current
      : {
          model: 1,
          markov: 1,
          streak: 1,
          pattern: 1,
          ewma: 1,
          bayes: 1,
          entropyMod: 1,
        };
    let blended = blend(probs, controlsRef.current, dyn);
    // Update bias gap counters
    const brr = biasRegimeRef.current;
    const thisPredLabel = blended >= 0.5 ? 1 : 0;
    brr.recent.push({ predOver: thisPredLabel, actualOver: last });
    if (brr.recent.length > BIAS_WINDOW) brr.recent.shift();
    brr.predOverCount = brr.recent.reduce((a, r) => a + r.predOver, 0);
    brr.actualOverCount = brr.recent.reduce((a, r) => a + r.actualOver, 0);
    const len = brr.recent.length || 1;
    const predRate = brr.predOverCount / len;
    const actualRate = brr.actualOverCount / len;
    brr.biasGap = predRate - actualRate;
    blended = applyBiasGapCorrection(blended);
    if (controlsRef.current.antiStreak && rl >= 5) {
      const f = Math.min(0.4, (rl - 4) * 0.05);
      const lastIsOver = last === 1;
      const target = lastIsOver ? 0.3 : 0.7;
      if ((lastIsOver && blended > 0.55) || (!lastIsOver && blended < 0.45))
        blended = blended * (1 - f) + target * f;
    }
    if (biasOverrideRef.current) {
      const adjusted = [
        probs.model,
        probs.markov,
        probs.streak,
        probs.pattern,
        probs.ewma,
        probs.bayes,
        probs.entropyMod,
      ].map((p) => 0.75 * p + 0.25 * 0.5);
      const clamp = (p: number) => Math.min(1 - 1e-6, Math.max(1e-6, p));
      const logits = adjusted.map((p) => Math.log(clamp(p) / (1 - clamp(p))));
      const mean = logits.reduce((a, b) => a + b, 0) / logits.length;
      blended = 1 / (1 + Math.exp(-mean));
    }
    const rawProb = blended;
    const invertUsed = controlsRef.current.invertStrategy;
    const effective = invertUsed ? 1 - rawProb : rawProb;
    lastPredictionRef.current = {
      rawProb,
      effectiveProb: effective,
      invertUsed,
      threshold: controlsRef.current.threshold,
    };
    setProbOver(effective);
    recomputePendingRef.current = false;
  }, [history, windowSize, buildFeatureWindow, ensureModel]);

  const setControls = (partial: Partial<typeof controlsRef.current>) => {
    const keys = Object.keys(partial);
    Object.assign(controlsRef.current, partial);
    if (
      keys.some((k) =>
        [
          "invertStrategy",
          "wModel",
          "wMarkov",
          "wStreak",
          "wPattern",
          "wEwma",
          "wBayes",
          "wEntropy",
          "rlEnabled",
          "antiStreak",
          "threshold",
          "bayesPriorA",
          "bayesPriorB",
          "entropyWindow",
          "mode",
          "rouletteVariant",
        ].includes(k)
      )
    ) {
      recomputeCurrentProb();
    }
    setControlsVersion((v) => v + 1); // force re-render for ref consumers
  };

  const lastTrainCountRef = useRef(0);
  const trainAndPredict = useCallback(async () => {
    const m = await ensureModel();
    if (history.length < 5) {
      setProbOver(null);
      return;
    }
    if (history.length <= windowSize) {
      await recomputeCurrentProb();
      return;
    }
    // Gate early training to allow heuristics stabilization
    if (history.length < TRAIN_START || history.length % TRAIN_EVERY !== 0) {
      await recomputeCurrentProb();
      return;
    }
    // Throttle: only retrain if we have at least 1 new sample since last train
    if (history.length === lastTrainCountRef.current) {
      // Only need to recompute blend (e.g. weights changed externally)
      await recomputeCurrentProb();
      return;
    }
    const xsArr: number[][][] = [];
    const ysArr: number[] = [];
    for (let i = 0; i < history.length - windowSize; i++) {
      const win = history.slice(i, i + windowSize);
      const y = history[i + windowSize];
      xsArr.push(buildFeatureWindow(win));
      ysArr.push(y);
    }
    if (!xsArr.length) {
      setProbOver(null);
      return;
    }
    const t0 = performance.now();
    const xs = tf.tensor3d(xsArr);
    const ys = tf.tensor2d(ysArr.map((v) => [v]));
    await m.fit(xs, ys, {
      epochs,
      batchSize: Math.min(16, xsArr.length),
      shuffle: true,
      verbose: 0,
    });
    xs.dispose();
    ys.dispose();
    if (process.env.NODE_ENV !== "production") {
      const t1 = performance.now();
      const mem = tf.memory();
      // eslint-disable-next-line no-console
      console.debug(
        "[train] samples",
        history.length,
        "fit(ms)",
        (t1 - t0).toFixed(1),
        "tensors",
        mem.numTensors
      );
    }
    lastTrainCountRef.current = history.length;
    await recomputeCurrentProb();
  }, [
    history,
    windowSize,
    buildFeatureWindow,
    ensureModel,
    recomputeCurrentProb,
  ]);

  // Dozens training loop (multi-class)
  const dozenLastTrainRef = useRef(0);
  // === Aggressive Dozens Tunables ===
  const DOZEN_TRAIN_START = 25; // start training after 25 spins
  const DOZEN_TRAIN_EVERY = 2; // train every 2nd spin thereafter
  const dozenTrainLockRef = useRef(false);
  const DOZEN_MODEL_RAMP_SPINS = 12; // ramp model weight over 12 spins (was 30)
  const DOZEN_INV_FREQ_GAMMA = 0.25; // softer inverse-frequency debias (was 0.4)
  const DOZEN_ENTROPY_FLOOR = 0.7; // relaxed entropy floor (was 0.85)
  const DOZEN_ENTROPY_STOP = 40; // stop entropy blending earlier (was 60)
  const DOZEN_STREAK_DAMP_START = 7; // later dampening (was 5)
  const DOZEN_STREAK_DAMP_MAX = 0.28; // slightly lower max damp (was 0.35)
  const DOZEN_TEMP_START = 15; // start temperature sharpening
  const DOZEN_TEMP_FULL = 60; // fully sharpen by this spin
  const DOZEN_TEMP_MIN = 0.8; // final lower temperature (sharper)
  const DOZEN_DIRICHLET_ALPHA = 0.6; // reduced pseudo-count (was implicit 1)
  // Anti-tunnel / error-streak mitigation constants
  const DOZEN_ERR_STREAK_PENALTY_START = 4; // start penalizing after 4 consecutive incorrect predictions of same dozen
  const DOZEN_ERR_STREAK_PENALTY_STEP = 0.12; // incremental penalty fraction per extra miss beyond start
  const DOZEN_ERR_STREAK_PENALTY_MAX = 0.55; // cap total penalty fraction removed from that dozen
  const DOZEN_RECENT_PRED_WINDOW = 30; // window for recent per-class prediction accuracy gating
  // Prediction imbalance mitigation
  const DOZEN_PRED_IMBAL_WINDOW = 40; // window length for assessing prediction imbalance
  const DOZEN_PRED_IMBAL_FACTOR = 1.35; // allowed multiple of actual share before penalizing
  const DOZEN_PRED_IMBAL_MAX_SHARE = 0.55; // hard cap share guideline
  const DOZEN_PRED_IMBAL_PENALTY_MAX = 0.4; // max portion to redistribute from dominant prediction
  // ==================================
  // track prediction streak to damp persistent bias
  const dozenPredStreakRef = useRef<{ last: number | null; len: number }>({
    last: null,
    len: 0,
  });
  // track consecutive incorrect predictions for currently repeating predicted dozen
  const dozenPredErrorStreakRef = useRef<{
    last: number | null;
    incorrect: number;
  }>({ last: null, incorrect: 0 });
  // Auto-rotation cooldown (spins remaining before another forced rotation allowed)
  const dozenRotateCooldownRef = useRef<number>(0);
  const dozenPredFreqRef = useRef<[number, number, number]>([0, 0, 0]);
  // Track per-class prediction accuracy and model-only accuracy for gating
  const dozenClassStatsRef = useRef<
    Record<number, { correct: number; total: number }>
  >({
    0: { correct: 0, total: 0 },
    1: { correct: 0, total: 0 },
    2: { correct: 0, total: 0 },
  });
  const dozenModelPerfRef = useRef<{ correct: number; total: number }>({
    correct: 0,
    total: 0,
  });
  const dozenRecentPredsRef = useRef<{ label: number; correct: boolean }[]>([]);
  // Extended decision dynamics
  const dozenPerClassMissRef = useRef<[number, number, number]>([0, 0, 0]);
  const dozenRecentResultsRef = useRef<{ correct: boolean; prob: number }[]>(
    []
  );
  // Regime tracking (streak vs choppy vs mixed)
  const dozenRegimeRef = useRef<{
    regime: "STREAK" | "CHOPPY" | "MIXED";
    avgRun: number;
    changeRate: number;
    runVar: number;
  }>({ regime: "MIXED", avgRun: 0, changeRate: 0, runVar: 0 });
  // Per-class prediction performance & pattern / tunnel tracking
  interface ClassPredStat {
    attempts: number;
    wins: number;
    sumTopProb: number;
    consecMisses: number;
    recent: { win: boolean; p: number }[];
  }
  const dozenClassPerfRef = useRef<ClassPredStat[]>([
    { attempts: 0, wins: 0, sumTopProb: 0, consecMisses: 0, recent: [] },
    { attempts: 0, wins: 0, sumTopProb: 0, consecMisses: 0, recent: [] },
    { attempts: 0, wins: 0, sumTopProb: 0, consecMisses: 0, recent: [] },
  ]);
  const dozenTunnelWindowRef = useRef<{ cls: number; win: boolean }[]>([]);
  const dozenPatternPerfRef = useRef<
    Record<string, { tries: number; wins: number }>
  >({});
  const dozenPatternWindowRef = useRef<string[]>([]);
  // Markov & run-length model refs for dozens
  const dozenMarkovRef = useRef<number[][]>([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
  ]);
  const dozenEwmaRef = useRef<[number, number, number] | null>(null);
  const dozenMarkov2Ref = useRef<number[][][]>([
    [
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1],
    ],
    [
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1],
    ],
    [
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1],
    ],
  ]);
  const dozenRunLengthBreakRef = useRef<Record<number, Record<number, number>>>(
    { 0: {}, 1: {}, 2: {} }
  );
  const dozenRunLengthOngoingRef = useRef<Record<number, number>>({});
  const dozenLastSourcesRef = useRef<Record<string, [number, number, number]>>(
    {}
  );
  const dozenSourceStatsRef = useRef<
    Record<string, { correct: number; total: number }>
  >({});
  const dozenSourceCountsRef = useRef<Record<string, number>>({});
  const dozenShapePerfRef = useRef<
    Record<string, { tries: number; wins: number }>
  >({});
  const dozenCalibBinsRef = useRef<
    {
      low: number;
      high: number;
      acc: number;
      count: number;
      preds?: number;
      correct?: number;
    }[]
  >([
    { low: 0, high: 0.2, acc: 0.33, count: 0 },
    { low: 0.2, high: 0.3, acc: 0.33, count: 0 },
    { low: 0.3, high: 0.4, acc: 0.33, count: 0 },
    { low: 0.4, high: 0.5, acc: 0.33, count: 0 },
    { low: 0.5, high: 0.6, acc: 0.33, count: 0 },
    { low: 0.6, high: 0.7, acc: 0.33, count: 0 },
    { low: 0.7, high: 1.01, acc: 0.33, count: 0 },
  ]);
  const [biasInfo, setBiasInfo] = useState<{
    status: "neutral" | "overBias" | "underBias";
    gap: number;
  }>({ status: "neutral", gap: 0 });
  function dozenShapeHash(dist: number[]) {
    const bins = dist
      .map((p) => Math.min(19, Math.floor((p * 100) / 5)))
      .join("-");
    const order = dist
      .map((p, i) => [p, i] as [number, number])
      .sort((a, b) => b[0] - a[0])
      .map((x) => x[1])
      .join("");
    return bins + "|" + order;
  }
  // Last decision details for diagnostics
  // (brierLevel appended later after final safety application)
  const dozenDecisionInfoRef = useRef<{
    method: string;
    gap: number;
    top: number;
  } | null>(null);
  // Early accuracy guard diagnostics
  const dozenGuardRef = useRef<{ triggers: number; lastActive: boolean }>({
    triggers: 0,
    lastActive: false,
  });
  // Regret & calibration tracking
  const dozenRegretRef = useRef<[number, number, number]>([0, 0, 0]);
  const dozenHiConfWrongRef = useRef<number[]>([]); // 1 = high confidence wrong
  const dozenRecentWindowRef = useRef<number[]>([]); // recent actual outcomes
  // Calibration & monitoring additions
  const dozenCalibBufferRef = useRef<
    {
      probs: [number, number, number];
      label: number;
    }[]
  >([]);
  const dozenCalibAdjRef = useRef<{
    mult: [number, number, number];
    temp: number;
  }>({ mult: [1, 1, 1], temp: 1 });
  const dozenBrierWindowRef = useRef<number[]>([]); // per-spin multi-class Brier
  const dozenConfusionRef = useRef<number[][]>([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ]);
  // Overall accuracy for dozens mode
  const dozenAccRef = useRef<{ correct: number; total: number }>({
    correct: 0,
    total: 0,
  });
  // Rolling top-probability vs accuracy window (for overconfidence gap control)
  const dozenTopProbWindowRef = useRef<{ p: number; correct: boolean }[]>([]);
  // Cooldown after catastrophic high-confidence miss
  const dozenHiProbCooldownRef = useRef<{ spins: number; factor: number }>({
    spins: 0,
    factor: 1,
  });
  // High-confidence outcome window (top prob >= threshold)
  const dozenHiConfWindowRef = useRef<{ p: number; correct: boolean }[]>([]);
  // Reliability running sums to compute multiplicative calibration factors
  const dozenReliabilityRef = useRef<{
    predSum: [number, number, number];
    actual: [number, number, number];
  }>({ predSum: [0, 0, 0], actual: [0, 0, 0] });
  // Miss-streak diversion tracking (per predicted class)
  const dozenPredMissStreakRef = useRef<[number, number, number]>([0, 0, 0]);
  // Regret / calibration constants
  const DZ_LOWCONF_GAP = 0.05;
  const DZ_LOWCONF_TOP = 0.45;
  const DZ_REGRET_DECAY = 0.92;
  const DZ_REGRET_LIMIT = 2.2;
  const DZ_REGRET_HARD = 3.5;
  const DZ_HICONF_PROB = 0.5;
  const DZ_HICONF_CLAMP = 0.56;
  const DZ_HICONF_WINDOW = 40;
  const DZ_HICONF_ERRRATE = 0.62;
  const DZ_MAJ_WINDOW = 12;
  const DZ_CALIB_WINDOW = 60; // rolling window for calibration
  const DZ_CALIB_MIN = 15; // minimum samples before applying adjustments
  const DZ_CALIB_LR = 0.25; // learning rate for multiplicative reliability
  const DZ_BRIER_WINDOW = 50;
  const DZ_BRIER_BASELINE = 2 * (2 / 3) ** 2 + (1 / 3) ** 2; // baseline uniform multi-class Brier (~0.666...)
  const DZ_BRIER_SHRINK_MAX = 0.35; // max pull toward uniform
  const DZ_ANTICOLL_ENTROPY = 0.9; // entropy threshold (bits) to trigger temperature raising
  const DZ_ANTICOLL_LASTN = 5; // accuracy window for anti-collapse
  const DZ_ANTICOLL_MIN_ACC = 0.33; // baseline accuracy
  const DZ_MISS_DIVERSION_GAP = 0.06; // gap threshold for miss-streak diversion
  const DZ_EARLY_HICONF_CLAMP_BASE = 0.52; // initial clamp start
  const DZ_HICONF_CLAMP_MIN = 0.48; // adaptive lower bound

  // --- Remote persistence helpers (Edge Config via /api/state) ---
  const remoteAttemptRef = useRef(false);
  const pendingRemotePushRef = useRef(false);
  const pullRemoteOnce = useCallback(async () => {
    if (remoteAttemptRef.current) return;
    remoteAttemptRef.current = true;
    remoteStateRef.current.status = "loading";
    try {
      const res = await fetch("/api/state", { cache: "no-store" });
      if (!res.ok) throw new Error("remote fetch failed " + res.status);
      const data = await res.json();
      const safeArr = (v: any, pred: (x: any) => boolean) =>
        Array.isArray(v) && v.every(pred) ? v : null;
      if (data.bin_history) {
        const arr = safeArr(data.bin_history, (x) => x === 0 || x === 1);
        if (arr && arr.length && arr.length > history.length)
          setHistory(arr.slice(-PERSIST_LIMIT));
      }
      if (data.bin_records) {
        const arr = safeArr(data.bin_records, (r) => typeof r === "object");
        if (arr && arr.length && arr.length > records.length)
          setRecords(arr.slice(-PERSIST_LIMIT));
      }
      if (data.dz_history) {
        const arr = safeArr(
          data.dz_history,
          (x) => x === 0 || x === 1 || x === 2
        );
        if (arr && arr.length && arr.length > dozenHistory.length)
          setDozenHistory(arr.slice(-PERSIST_LIMIT));
      }
      if (data.dz_records) {
        const arr = safeArr(data.dz_records, (r) => typeof r === "object");
        if (arr && arr.length && arr.length > dozenRecords.length)
          setDozenRecords(arr.slice(-PERSIST_LIMIT));
      }
      if (data.dz_state) {
        const ds = data.dz_state;
        if (ds?.rel?.predSum && Array.isArray(ds.rel.predSum)) {
          dozenReliabilityRef.current.predSum = ds.rel.predSum;
          dozenReliabilityRef.current.actual = ds.rel.actual || [0, 0, 0];
        }
        if (ds?.calib?.mult && Array.isArray(ds.calib.mult)) {
          dozenCalibAdjRef.current.mult = ds.calib.mult;
          if (typeof ds.calib.temp === "number")
            dozenCalibAdjRef.current.temp = ds.calib.temp;
        }
      }
      remoteStateRef.current.status = "ready";
      remoteStateRef.current.lastPull = Date.now();
    } catch (err: any) {
      remoteStateRef.current.status = "error";
      remoteStateRef.current.error = err?.message || "remote pull failed";
    }
  }, [
    history.length,
    records.length,
    dozenHistory.length,
    dozenRecords.length,
  ]);
  const pushRemote = useCallback(async () => {
    if (pendingRemotePushRef.current) return;
    pendingRemotePushRef.current = true;
    try {
      const payload = {
        bin_history: history.slice(-PERSIST_LIMIT),
        bin_records: records.slice(-PERSIST_LIMIT),
        dz_history: dozenHistory.slice(-PERSIST_LIMIT),
        dz_records: dozenRecords.slice(-PERSIST_LIMIT),
        dz_state: {
          rel: dozenReliabilityRef.current,
          calib: dozenCalibAdjRef.current,
        },
      };
      await fetch("/api/state", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        keepalive: true,
      });
      remoteStateRef.current.lastPush = Date.now();
      if (remoteStateRef.current.status === "idle")
        remoteStateRef.current.status = "ready";
    } catch {
    } finally {
      pendingRemotePushRef.current = false;
    }
  }, [history, records, dozenHistory, dozenRecords]);

  // === DOZENS DECISION HELPERS ===
  const sampleDirichlet = (
    v: [number, number, number]
  ): [number, number, number] => {
    // Simplified gamma sampling (shape alpha, scale 1)
    const gamma = (alpha: number): number => {
      const a = Math.max(1e-3, alpha);
      if (a < 1) {
        // Johnk / Ahrens-Dieter style
        const u = Math.random();
        return gamma(a + 1) * Math.pow(u, 1 / a);
      }
      const d = a - 1 / 3;
      const c = 1 / Math.sqrt(9 * d);
      while (true) {
        let x: number;
        let v2: number;
        do {
          const u = Math.random();
          const u2 = Math.random();
          const n = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * u2);
          x = n;
          v2 = 1 + c * x;
        } while (v2 <= 0);
        v2 = v2 * v2 * v2;
        const u3 = Math.random();
        if (
          u3 < 1 - 0.0331 * (x * x) * (x * x) ||
          Math.log(u3) < 0.5 * x * x + d * (1 - v2 + Math.log(v2))
        ) {
          return d * v2;
        }
      }
    };
    const g = v.map(gamma) as [number, number, number];
    const s = g[0] + g[1] + g[2];
    return [g[0] / s, g[1] / s, g[2] / s];
  };

  interface DecisionInfo {
    method: string;
    gap: number;
    top: number;
  }
  const chooseDozen = (
    baseDist: [number, number, number],
    counts: [number, number, number],
    regime: "STREAK" | "CHOPPY" | "MIXED"
  ): {
    dist: [number, number, number];
    pred: 0 | 1 | 2;
    info: DecisionInfo;
  } => {
    let dist: [number, number, number] = [...baseDist] as any;
    // Dynamic temperature based on recent realized accuracy
    const rec = dozenRecentResultsRef.current.slice(-25);
    if (rec.length >= 12) {
      const acc = rec.filter((r) => r.correct).length / rec.length;
      let T = 1.0;
      // Escalated flattening when recent accuracy is materially below random baseline (~0.33)
      if (acc < 0.26) T = 1.35; // heavy flatten – severe slump
      else if (acc < 0.3) T = 1.2; // moderate slump
      else if (acc < 0.33) T = 1.1; // slight underperformance
      else if (acc > 0.42) T = 0.85; // sharpen (exploit) only on clear strength
      if (T !== 1) {
        const adj = dist.map((p) => Math.pow(Math.max(p, 1e-8), 1 / T));
        const sT = adj.reduce((a, b) => a + b, 0);
        if (sT > 0) dist = [adj[0] / sT, adj[1] / sT, adj[2] / sT] as any;
      }
    }
    // Miss streak capping (if repeatedly wrong on class with low prob advantage)
    const secondProb = [...dist].sort((a, b) => b - a)[1];
    const missCapThreshold = regime === "CHOPPY" ? 1 : 2; // tighter in choppy regime
    for (let i = 0; i < 3; i++) {
      if (
        dozenPerClassMissRef.current[i] >= missCapThreshold &&
        dist[i] < 0.5
      ) {
        const cap = Math.min(dist[i], secondProb + 0.04);
        if (cap < dist[i]) {
          const diff = dist[i] - cap;
          dist[i] = cap;
          const others = [0, 1, 2].filter((c) => c !== i);
          const add = diff / 2;
          dist[others[0]] += add;
          dist[others[1]] += diff - add;
        }
      }
    }
    // Low-confidence diversification
    const sorted = [...dist].sort((a, b) => b - a);
    const top = sorted[0];
    const gap = top - sorted[1];
    let method = "argmax";
    // Regime-aware thresholds
    const gapThresh =
      regime === "STREAK" ? 0.05 : regime === "CHOPPY" ? 0.07 : 0.06;
    const topThresh =
      regime === "STREAK" ? 0.45 : regime === "CHOPPY" ? 0.42 : 0.43;
    if (gap < gapThresh || top < topThresh) {
      // Thompson sampling
      const alpha: [number, number, number] = [
        counts[0] + 0.6,
        counts[1] + 0.6,
        counts[2] + 0.6,
      ];
      const dir = sampleDirichlet(alpha);
      const BLEND =
        regime === "CHOPPY" ? 0.6 : regime === "STREAK" ? 0.5 : 0.55; // slightly more exploration in choppy
      dist = [
        (1 - BLEND) * dist[0] + BLEND * dir[0],
        (1 - BLEND) * dist[1] + BLEND * dir[1],
        (1 - BLEND) * dist[2] + BLEND * dir[2],
      ];
      const sDir = dist[0] + dist[1] + dist[2];
      dist = [dist[0] / sDir, dist[1] / sDir, dist[2] / sDir] as any;
      method = "thompson";
      // Epsilon small exploration if still near uniform
      const dev =
        Math.abs(dist[0] - 1 / 3) +
        Math.abs(dist[1] - 1 / 3) +
        Math.abs(dist[2] - 1 / 3);
      const epsP =
        regime === "CHOPPY" ? 0.1 : regime === "STREAK" ? 0.05 : 0.07;
      if (dev < 0.12 && Math.random() < epsP) {
        let r = Math.random();
        for (let i = 0; i < 3; i++) {
          if (r < dist[i]) {
            return {
              dist,
              pred: i as 0 | 1 | 2,
              info: { method: "epsilon", gap, top },
            };
          }
          r -= dist[i];
        }
      }
    }
    const pred =
      dist[0] >= dist[1] && dist[0] >= dist[2] ? 0 : dist[1] >= dist[2] ? 1 : 2;
    return { dist, pred: pred as 0 | 1 | 2, info: { method, gap, top } };
  };
  // === END DOZENS DECISION HELPERS ===
  const trainAndPredictDozens = useCallback(async () => {
    if (!backendReadyRef.current) await ensureBackend();
    if (!backendReadyRef.current) return; // still not ready
    // decrement rotation cooldown
    if (dozenRotateCooldownRef.current > 0) dozenRotateCooldownRef.current -= 1;
    const m = await ensureDozenModel();
    // If insufficient history for a window, still output a uniform distribution so UI has stable values.
    if (dozenHistory.length === 0) {
      const uniform: [number, number, number] = [1 / 3, 1 / 3, 1 / 3];
      dozenProbsRef.current = uniform;
      dozenPredictionRef.current = { probs: uniform, predLabel: 0 };
      setDozenVersion((v) => v + 1);
      return;
    }
    if (
      !dozenTrainLockRef.current &&
      dozenHistory.length >= DOZEN_TRAIN_START &&
      dozenHistory.length % DOZEN_TRAIN_EVERY === 0 &&
      dozenHistory.length !== dozenLastTrainRef.current &&
      dozenHistory.length > windowSize
    ) {
      const xsArr: number[][][] = [];
      const ysArr: number[][] = [];
      for (let i = 0; i < dozenHistory.length - windowSize; i++) {
        const win = dozenHistory.slice(i, i + windowSize) as (0 | 1 | 2)[];
        const y = dozenHistory[i + windowSize];
        xsArr.push(buildDozenFeatureWindow(win));
        const oneHot: [number, number, number] = [0, 0, 0];
        oneHot[y] = 1;
        ysArr.push(oneHot);
      }
      if (xsArr.length) {
        try {
          dozenTrainLockRef.current = true;
          const t0 = performance.now();
          const xs = tf.tensor3d(xsArr);
          const ys = tf.tensor2d(ysArr);
          await m
            .fit(xs, ys, {
              epochs: 1,
              verbose: 0,
              batchSize: Math.min(16, xsArr.length),
            })
            .then(() => {
              const dt = performance.now() - t0;
              (dozenAdaptiveRef.current as any).lastTrainMs = dt;
              (dozenAdaptiveRef.current as any).lastTrainSize = xsArr.length;
              if (console && console.log)
                console.log("Dozens model trained", {
                  spins: dozenHistory.length,
                  samples: xsArr.length,
                  ms: +dt.toFixed(1),
                });
            })
            .catch((e) => {
              (dozenAdaptiveRef.current as any).lastTrainError = String(e);
              if (console && console.error)
                console.error("Dozens training error", e);
            });
          xs.dispose();
          ys.dispose();
          dozenLastTrainRef.current = dozenHistory.length;
        } finally {
          dozenTrainLockRef.current = false;
        }
      }
    }
    const counts = [0, 0, 0];
    dozenHistory.forEach((v) => (counts[v] += 1));
    const dirichlet = counts.map(
      (c) =>
        (c + DOZEN_DIRICHLET_ALPHA) /
        (dozenHistory.length + 3 * DOZEN_DIRICHLET_ALPHA)
    ) as [number, number, number];
    // --- Regime Detection (recent segment) ---
    const REG_WINDOW = 40;
    const seg = dozenHistory.slice(-REG_WINDOW);
    if (seg.length >= 8) {
      // compute run lengths
      const runs: number[] = [];
      let cur = 1;
      for (let i = 1; i < seg.length; i++) {
        if (seg[i] === seg[i - 1]) cur++;
        else {
          runs.push(cur);
          cur = 1;
        }
      }
      runs.push(cur);
      const avgRun = runs.reduce((a, b) => a + b, 0) / runs.length;
      const changeRate =
        seg.length - 1 === 0
          ? 0
          : runs.length - 1 === 0
          ? 0
          : (seg.length - runs.reduce((a, b) => a + b, 0) + runs.length) /
            (seg.length - 1); // fallback formula
      // simpler: transitions / (len-1)
      let transitions = 0;
      for (let i = 1; i < seg.length; i++)
        if (seg[i] !== seg[i - 1]) transitions++;
      const transRate = transitions / (seg.length - 1);
      // variance of runs
      const mean = avgRun;
      const runVar =
        runs.reduce((a, r) => a + (r - mean) * (r - mean), 0) / runs.length;
      // classify
      let regime: "STREAK" | "CHOPPY" | "MIXED" = "MIXED";
      if (avgRun >= 2.4 && transRate < 0.55) regime = "STREAK";
      else if (avgRun < 1.8 && transRate >= 0.65) regime = "CHOPPY";
      dozenRegimeRef.current = {
        regime,
        avgRun,
        changeRate: transRate,
        runVar,
      };
    } else {
      dozenRegimeRef.current = {
        regime: "MIXED",
        avgRun: 0,
        changeRate: 0,
        runVar: 0,
      };
    }
    let modelProbs: [number, number, number] | null = null;
    if (dozenHistory.length >= windowSize) {
      modelProbs = tf.tidy(() => {
        const lastWin = dozenHistory.slice(-windowSize) as (0 | 1 | 2)[];
        const feats = buildDozenFeatureWindow(lastWin);
        const t = tf.tensor3d([feats]);
        const out = m.predict(t) as tf.Tensor;
        const arr = Array.from(out.dataSync()) as [number, number, number];
        t.dispose();
        out.dispose();
        return arr;
      });
    }
    // Apply rolling reliability calibration (simple multiplicative per-class + temperature) BEFORE other mixing
    if (modelProbs) {
      const calib = dozenCalibAdjRef.current;
      const adj = modelProbs.map((p, i) => p * calib.mult[i]) as [
        number,
        number,
        number
      ];
      let sAdj = adj[0] + adj[1] + adj[2];
      if (sAdj > 0) {
        let temp = calib.temp;
        if (temp !== 1) {
          const tAdj = adj.map((p) => Math.pow(Math.max(p, 1e-8), 1 / temp));
          const st = tAdj[0] + tAdj[1] + tAdj[2];
          modelProbs = [tAdj[0] / st, tAdj[1] / st, tAdj[2] / st] as [
            number,
            number,
            number
          ];
        } else {
          modelProbs = [adj[0] / sAdj, adj[1] / sAdj, adj[2] / sAdj];
        }
      }
    }
    // Markov distribution
    let markovDist: [number, number, number] = [1 / 3, 1 / 3, 1 / 3];
    if (dozenHistory.length >= 2) {
      const last = dozenHistory[dozenHistory.length - 1];
      const row = dozenMarkovRef.current[last];
      const s = row[0] + row[1] + row[2];
      markovDist = [row[0] / s, row[1] / s, row[2] / s] as [
        number,
        number,
        number
      ];
    }
    // Run-length continuation vs reversal (streak) + pattern similar
    let streakDist: [number, number, number] = [1 / 3, 1 / 3, 1 / 3];
    let patternDist: [number, number, number] = [1 / 3, 1 / 3, 1 / 3];
    if (dozenHistory.length >= 2) {
      const last = dozenHistory[dozenHistory.length - 1];
      let rl = 1;
      for (let i = dozenHistory.length - 2; i >= 0; i--) {
        if (dozenHistory[i] === last) rl++;
        else break;
      }
      const key = `${last}:${rl}`;
      const rec = dozenRunPostRef.current[key];
      let contP = 0.5;
      if (rec) contP = (rec.cont + 1) / (rec.cont + rec.rev + 2);
      const otherShare = (1 - contP) / 2;
      streakDist = [0, 1, 2].map((c) => (c === last ? contP : otherShare)) as [
        number,
        number,
        number
      ];
      // Pattern: look at distribution of past run lengths for this class; compute probability continuation probability similar to binary longer run heuristic
      const runs: number[] = [];
      let cur = 1;
      for (let i = dozenHistory.length - 2; i >= 0; i--) {
        if (dozenHistory[i] === dozenHistory[i + 1]) cur++;
        else {
          runs.push(cur);
          cur = 1;
        }
      }
      runs.push(cur);
      const matchingRuns = runs.filter((r) => r > rl).length + 1;
      const totalRuns = runs.length + 2;
      const pCont = matchingRuns / totalRuns;
      const pOther = (1 - pCont) / 2;
      patternDist = [0, 1, 2].map((c) => (c === last ? pCont : pOther)) as [
        number,
        number,
        number
      ];
    }
    // EWMA per class
    if (!dozenEwmaRef.current) {
      const s = counts[0] + counts[1] + counts[2];
      if (s === 0) {
        dozenEwmaRef.current = [1 / 3, 1 / 3, 1 / 3];
      } else {
        dozenEwmaRef.current = [
          counts[0] / s,
          counts[1] / s,
          counts[2] / s,
        ] as [number, number, number];
      }
    }
    const alpha = controlsRef.current.ewmaAlpha;
    const total = counts[0] + counts[1] + counts[2];
    const inst: [number, number, number] = total
      ? [counts[0] / total, counts[1] / total, counts[2] / total]
      : [1 / 3, 1 / 3, 1 / 3];
    // Adaptive EWMA alpha: emphasize recency <120 spins
    const spinCount = dozenHistory.length;
    const dynAlpha =
      spinCount < 60
        ? Math.min(0.22, alpha + 0.12)
        : spinCount < 120
        ? Math.min(0.16, alpha + 0.06)
        : alpha;
    dozenEwmaRef.current = [0, 1, 2].map(
      (i) =>
        (1 - dynAlpha) * (dozenEwmaRef.current as any)[i] + dynAlpha * inst[i]
    ) as [number, number, number];
    const ewmaDist = dozenEwmaRef.current;
    // Short vs Long frequency distributions
    const SHORT_FREQ_WIN = 60;
    const shortSlice = dozenHistory.slice(-SHORT_FREQ_WIN);
    const shortCounts = [0, 0, 0];
    shortSlice.forEach((v) => (shortCounts[v] += 1));
    const shortTot = shortCounts[0] + shortCounts[1] + shortCounts[2];
    const shortFreq: [number, number, number] = shortTot
      ? [
          shortCounts[0] / shortTot,
          shortCounts[1] / shortTot,
          shortCounts[2] / shortTot,
        ]
      : [1 / 3, 1 / 3, 1 / 3];
    const longFreq: [number, number, number] = inst; // overall inst already normalized
    // Lightweight recurrent (GRU-lite) hidden state capturing recent predictive context
    const recurRef = (dozenGuardRef as any).current?.recurHidden || {
      h: [1 / 3, 1 / 3, 1 / 3],
    };
    const recurAlpha = spinCount < 80 ? 0.18 : spinCount < 200 ? 0.12 : 0.08;
    for (let i = 0; i < 3; i++)
      recurRef.h[i] =
        (1 - recurAlpha) * recurRef.h[i] + recurAlpha * ewmaDist[i];
    const hSum = recurRef.h[0] + recurRef.h[1] + recurRef.h[2];
    if (hSum > 0) {
      for (let i = 0; i < 3; i++) recurRef.h[i] /= hSum;
    }
    (dozenGuardRef as any).current = {
      ...(dozenGuardRef as any).current,
      recurHidden: recurRef,
    };
    const recurDist: [number, number, number] = [
      recurRef.h[0],
      recurRef.h[1],
      recurRef.h[2],
    ];
    // Entropy modulation: push distribution toward last value when low entropy
    let entropyDist: [number, number, number] = [1 / 3, 1 / 3, 1 / 3];
    if (dozenHistory.length) {
      const last = dozenHistory[dozenHistory.length - 1];
      const recent = dozenHistory.slice(
        -Math.max(5, controlsRef.current.entropyWindow)
      );
      const rc = [0, 0, 0];
      recent.forEach((v) => (rc[v] += 1));
      const denom = recent.length || 1; // avoid 0
      const rp = rc.map((c) => c / denom);
      const ent = -rp.reduce((a, c) => (c > 0 ? a + c * Math.log2(c) : a), 0);
      const maxEnt = Math.log2(3);
      const decisiveness = 1 - ent / maxEnt; // 0..1
      const boost = 0.25 * decisiveness;
      const base = 1 / 3 - boost / 2; // distribute removal across others
      entropyDist = [0, 1, 2].map((i) =>
        i === last ? 1 / 3 + boost : base
      ) as [number, number, number];
    }
    // Collect sources
    const sources: Record<string, [number, number, number]> = {
      model: modelProbs || dirichlet,
      markov: markovDist,
      streak: streakDist,
      pattern: patternDist,
      ewma: ewmaDist,
      bayes: dirichlet,
      entropyMod: entropyDist,
      shortFreq,
      longFreq,
      recur: recurDist,
    };
    // Second-order Markov distribution
    if (dozenHistory.length >= 2) {
      const prev1 = dozenHistory[dozenHistory.length - 1];
      const prev2 = dozenHistory[dozenHistory.length - 2];
      const row = dozenMarkov2Ref.current[prev2][prev1];
      const sum = row[0] + row[1] + row[2];
      sources.markov2 = [row[0] / sum, row[1] / sum, row[2] / sum];
    }
    // Hazard source (run-length break probability)
    if (dozenHistory.length) {
      const last = dozenHistory[dozenHistory.length - 1];
      let run = 1;
      for (let i = dozenHistory.length - 2; i >= 0; i--) {
        if (dozenHistory[i] === last) run++;
        else break;
      }
      const breaks = dozenRunLengthBreakRef.current[last][run] || 1;
      const ongoing = dozenRunLengthOngoingRef.current[last] || run;
      const hazard = breaks / (breaks + ongoing); // probability run breaks now
      const dist: [number, number, number] = [
        hazard / 2,
        hazard / 2,
        hazard / 2,
      ];
      dist[last] = 1 - hazard;
      sources.hazard = dist;
    }
    dozenLastSourcesRef.current = sources; // snapshot for RL update
    // Weighting (manual * RL if enabled)
    // UCB weighting across sources
    const dyn: Record<string, number> = {};
    const totalSpins = Math.max(1, dozenHistory.length);
    const c = totalSpins < 150 ? 0.9 : 0.5;
    Object.keys(sources).forEach((k) => {
      const stat = dozenSourceStatsRef.current[k];
      const cnt = dozenSourceCountsRef.current[k] || 1;
      const mean = stat ? stat.correct / Math.max(1, stat.total) : 0.33;
      dyn[k] = mean + c * Math.sqrt(Math.log(totalSpins + 1) / cnt);
    });
    const manual = controlsRef.current;
    // Ramp model weight in gradually to avoid early overfitting dominance
    const n = dozenHistory.length;
    const modelRamp =
      n <= DOZEN_TRAIN_START
        ? 0
        : Math.min(1, (n - DOZEN_TRAIN_START) / DOZEN_MODEL_RAMP_SPINS);
    const weightMap: Record<string, number> = {
      model: manual.wModel * modelRamp * dyn.model,
      markov: manual.wMarkov * dyn.markov,
      streak: manual.wStreak * dyn.streak,
      pattern: manual.wPattern * dyn.pattern,
      ewma: manual.wEwma * dyn.ewma,
      bayes: manual.wBayes * dyn.bayes,
      entropyMod: manual.wEntropy * dyn.entropyMod,
    };
    // Regime-based dynamic scaling
    const regimeNow = dozenRegimeRef.current.regime;
    const adj: Record<string, number> = {
      model: 1,
      markov: 1,
      streak: 1,
      pattern: 1,
      ewma: 1,
      bayes: 1,
      entropyMod: 1,
    };
    if (regimeNow === "STREAK") {
      adj.streak *= 1.25;
      adj.pattern *= 1.2;
      adj.markov *= 0.85;
      adj.entropyMod *= 0.9;
    } else if (regimeNow === "CHOPPY") {
      adj.markov *= 1.25;
      adj.entropyMod *= 1.15;
      adj.streak *= 0.8;
      adj.pattern *= 0.85;
    }
    Object.keys(weightMap).forEach((k) => {
      weightMap[k] *= adj[k];
    });
    // Performance gating: If model's recent pure accuracy (over last 30 spins) underperforms simple baselines, down-weight it dynamically.
    if (dozenModelPerfRef.current.total >= 15) {
      const recent = dozenRecentPredsRef.current.slice(-30);
      if (recent.length) {
        const acc =
          recent.filter((r) => r.correct).length / Math.max(1, recent.length);
        // baseline = max of (uniform 1/3, best class empirical freq)
        const countsCopy = [...counts];
        const freqMax = Math.max(countsCopy[0], countsCopy[1], countsCopy[2]);
        const freqBaseline =
          freqMax / Math.max(1, countsCopy[0] + countsCopy[1] + countsCopy[2]);
        const baseline = Math.max(1 / 3, freqBaseline * 0.9); // small discount
        if (acc < baseline) {
          const penalty = Math.min(0.6, baseline - acc); // cap penalty
          weightMap.model *= 1 - penalty; // shrink model influence
        } else if (acc > baseline + 0.05) {
          const bonus = Math.min(0.4, acc - baseline - 0.05);
          weightMap.model *= 1 + bonus; // modest boost
        }
      }
    }
    // Chi-squared bias detection on last 100 spins to tilt shortFreq/recur early
    (function biasTilt() {
      const win = dozenHistory.slice(-100);
      if (win.length >= 40) {
        const c = [0, 0, 0];
        win.forEach((v) => c[v]++);
        const n = win.length;
        const exp = n / 3;
        let chi = 0;
        for (let i = 0; i < 3; i++) chi += ((c[i] - exp) * (c[i] - exp)) / exp;
        // df=2 approx p-value (1 - CDF) using series expansion
        const p = Math.exp(-0.5 * chi) * (1 + chi / 2);
        const threshold = spinCount < 200 ? 0.05 : 0.01;
        if (p < threshold) {
          const biasIdx = c.indexOf(Math.max(...c));
          weightMap.shortFreq *= 1.15;
          weightMap.recur *= 1.1;
          // slight de-emphasis long sources to exploit temporary bias
          weightMap.longFreq *= 0.95;
          weightMap.markov *= 0.96;
          weightMap.bayes *= 0.96;
        }
      }
    })();
    // RL decay (forget older adjustments very slowly)
    Object.keys(rlWeightsRef.current).forEach((k) => {
      rlWeightsRef.current[k as keyof typeof rlWeightsRef.current] *= 0.998; // ~0.2 decay per 100 spins
    });
    const sumW = Object.values(weightMap).reduce((a, b) => a + b, 0) || 1;
    Object.keys(weightMap).forEach((k) => (weightMap[k] /= sumW));
    // Blend via weighted log probs
    const logBlend = [0, 0, 0];
    Object.entries(sources).forEach(([k, dist]) => {
      const w = weightMap[k];
      dist.forEach((p, i) => {
        const c = Math.max(p, 1e-6);
        logBlend[i] += w * Math.log(c);
      });
    });
    const expVals = logBlend.map((l) =>
      Math.exp(Math.max(-50, Math.min(50, l)))
    );
    let sumExp = expVals.reduce((a, b) => a + b, 0);
    if (!isFinite(sumExp) || sumExp <= 0) {
      sumExp = expVals
        .filter((v) => isFinite(v) && v > 0)
        .reduce((a, b) => a + b, 0);
    }
    let finalDist = expVals.map((v) => (sumExp > 0 ? v / sumExp : 1 / 3)) as [
      number,
      number,
      number
    ];
    // sanitize
    if (finalDist.some((p) => !isFinite(p) || p < 0)) {
      finalDist = [1 / 3, 1 / 3, 1 / 3];
    } else {
      const sfix = finalDist[0] + finalDist[1] + finalDist[2];
      if (Math.abs(sfix - 1) > 1e-6) {
        finalDist = [
          finalDist[0] / sfix,
          finalDist[1] / sfix,
          finalDist[2] / sfix,
        ];
      }
    }
    // --- Early-phase diversification (prevent pure persistence before learning) ---
    if (n < 25) {
      const total = counts[0] + counts[1] + counts[2];
      let freqDist: [number, number, number] = total
        ? [counts[0] / total, counts[1] / total, counts[2] / total]
        : [1 / 3, 1 / 3, 1 / 3];
      // Light Laplace smoothing toward uniform
      const SMOOTH_A = 0.8;
      freqDist = [
        (counts[0] + SMOOTH_A) / (total + 3 * SMOOTH_A),
        (counts[1] + SMOOTH_A) / (total + 3 * SMOOTH_A),
        (counts[2] + SMOOTH_A) / (total + 3 * SMOOTH_A),
      ];
      const uniform: [number, number, number] = [1 / 3, 1 / 3, 1 / 3];
      // Dynamic diversity strength stronger very early, fades to 0 by 25
      const diversityMix = 0.35 * (1 - n / 25);
      if (diversityMix > 0) {
        const blendedBase = [0, 1, 2].map(
          (i) => 0.5 * freqDist[i] + 0.5 * uniform[i]
        ) as [number, number, number];
        finalDist = [0, 1, 2].map(
          (i) =>
            (1 - diversityMix) * finalDist[i] + diversityMix * blendedBase[i]
        ) as [number, number, number];
        const sE = finalDist[0] + finalDist[1] + finalDist[2];
        if (sE > 0) {
          finalDist = [finalDist[0] / sE, finalDist[1] / sE, finalDist[2] / sE];
        }
        // If still near-persistent (top equals last actual with tiny gap) inject a mild Thompson sample blend
        if (dozenHistory.length) {
          const lastVal = dozenHistory[dozenHistory.length - 1];
          const sortedGap = [...finalDist].sort((a, b) => b - a);
          const gap = sortedGap[0] - sortedGap[1];
          if (gap < 0.12 && finalDist[lastVal] === Math.max(...finalDist)) {
            const alphaDir: [number, number, number] = [
              counts[0] + 0.8,
              counts[1] + 0.8,
              counts[2] + 0.8,
            ];
            // simple Dirichlet sample (reuse helper if desired, inline light version)
            const gamma = (a: number): number => {
              const aa = Math.max(1e-3, a);
              if (aa < 1) {
                const u = Math.random();
                return gamma(aa + 1) * Math.pow(u, 1 / aa);
              }
              const d = aa - 1 / 3;
              const c = 1 / Math.sqrt(9 * d);
              while (true) {
                let x: number, v: number;
                do {
                  const u = Math.random();
                  const u2 = Math.random();
                  const nrm =
                    Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * u2);
                  x = nrm;
                  v = 1 + c * x;
                } while (v <= 0);
                v = v * v * v;
                const u3 = Math.random();
                if (
                  u3 < 1 - 0.0331 * (x * x) * (x * x) ||
                  Math.log(u3) < 0.5 * x * x + d * (1 - v + Math.log(v))
                ) {
                  return d * v;
                }
              }
            };
            const g0 = gamma(alphaDir[0]);
            const g1 = gamma(alphaDir[1]);
            const g2 = gamma(alphaDir[2]);
            const gs = g0 + g1 + g2;
            const dirSample: [number, number, number] = [
              g0 / gs,
              g1 / gs,
              g2 / gs,
            ];
            const exploreBlend = 0.5 * diversityMix; // modest injection
            finalDist = [0, 1, 2].map(
              (i) =>
                (1 - exploreBlend) * finalDist[i] + exploreBlend * dirSample[i]
            ) as [number, number, number];
            const sEx = finalDist[0] + finalDist[1] + finalDist[2];
            finalDist = [
              finalDist[0] / sEx,
              finalDist[1] / sEx,
              finalDist[2] / sEx,
            ];
          }
        }
      }
    }
    // Miss-streak diversion pre-choice: if predicted class recently missed 2+ times and advantage small, rotate
    const sortedPre = [...finalDist]
      .map((p, i) => ({ i, p }))
      .sort((a, b) => b.p - a.p);
    if (sortedPre.length >= 2) {
      const topIdx = sortedPre[0].i;
      const secondIdx = sortedPre[1].i;
      const gap = sortedPre[0].p - sortedPre[1].p;
      if (
        dozenPredMissStreakRef.current[topIdx] >= 2 &&
        gap < DZ_MISS_DIVERSION_GAP
      ) {
        // divert probability mass to second
        const transfer = Math.min(0.15, (DZ_MISS_DIVERSION_GAP - gap) * 1.2);
        const amt = Math.min(finalDist[topIdx] * 0.4, transfer);
        finalDist[topIdx] -= amt;
        finalDist[secondIdx] += amt;
        const ss = finalDist[0] + finalDist[1] + finalDist[2];
        finalDist = [finalDist[0] / ss, finalDist[1] / ss, finalDist[2] / ss];
      }
    }
    // Anti-collapse temperature raising (flatten) if low entropy & short-term accuracy under baseline
    (function antiCollapse() {
      const recent = dozenRecentPredsRef.current.slice(-DZ_ANTICOLL_LASTN);
      if (recent.length >= DZ_ANTICOLL_LASTN) {
        const acc = recent.filter((r) => r.correct).length / recent.length;
        const ent = -finalDist.reduce(
          (a, p) => (p > 0 ? a + p * Math.log2(p) : a),
          0
        );
        if (ent < DZ_ANTICOLL_ENTROPY && acc < DZ_ANTICOLL_MIN_ACC) {
          const deficit = Math.max(0, DZ_ANTICOLL_MIN_ACC - acc);
          const raise = 1 + Math.min(0.6, deficit * 1.2); // temperature >1 flattens
          const adj = finalDist.map((p) =>
            Math.pow(Math.max(p, 1e-8), 1 / raise)
          );
          const sa = adj[0] + adj[1] + adj[2];
          finalDist = [adj[0] / sa, adj[1] / sa, adj[2] / sa];
        }
      }
    })();
    // Prediction imbalance mitigation: avoid over-predicting one dozen compared to its actual frequency.
    const recentPredsForImbal = dozenRecentPredsRef.current.slice(
      -DOZEN_PRED_IMBAL_WINDOW
    );
    if (recentPredsForImbal.length >= 12) {
      const predCounts = [0, 0, 0];
      recentPredsForImbal.forEach((p) => {
        if (p.label >= 0 && p.label <= 2) predCounts[p.label] += 1;
      });
      const predTot = predCounts[0] + predCounts[1] + predCounts[2];
      const predShare = predTot
        ? predCounts.map((c) => c / predTot)
        : [1 / 3, 1 / 3, 1 / 3];
      const recentActualSlice = dozenHistory.slice(-DOZEN_PRED_IMBAL_WINDOW);
      const actCounts = [0, 0, 0];
      recentActualSlice.forEach((v) => (actCounts[v] += 1));
      const actTot = actCounts[0] + actCounts[1] + actCounts[2];
      const actShare = actTot
        ? actCounts.map((c) => c / actTot)
        : [1 / 3, 1 / 3, 1 / 3];
      dozenPredFreqRef.current = predShare as [number, number, number];
      const curTop =
        finalDist[0] >= finalDist[1] && finalDist[0] >= finalDist[2]
          ? 0
          : finalDist[1] >= finalDist[2]
          ? 1
          : 2;
      const cap = Math.min(
        DOZEN_PRED_IMBAL_MAX_SHARE,
        actShare[curTop] * DOZEN_PRED_IMBAL_FACTOR + 0.05
      );
      if (predShare[curTop] > cap) {
        const excess = predShare[curTop] - cap;
        const penaltyFrac = Math.min(
          DOZEN_PRED_IMBAL_PENALTY_MAX,
          excess * 0.85
        );
        const removed = finalDist[curTop] * penaltyFrac;
        finalDist[curTop] -= removed;
        const others = [0, 1, 2].filter((i) => i !== curTop);
        const shortfalls = others.map((i) =>
          Math.max(0.001, cap - predShare[i])
        );
        const shortSum = shortfalls.reduce((a, b) => a + b, 0) || 1;
        others.forEach((cls, idx) => {
          finalDist[cls] += removed * (shortfalls[idx] / shortSum);
        });
        const s3 = finalDist[0] + finalDist[1] + finalDist[2];
        finalDist = [finalDist[0] / s3, finalDist[1] / s3, finalDist[2] / s3];
      }
    } else {
      dozenPredFreqRef.current = [0, 0, 0];
    }
    // === Recurring frequency & pattern strategy shaping ===
    // 1. Long-window empirical frequency bias (favor a class that persistently appears more, unless contradicted by strong alternation pattern).
    const FREQ_WINDOW = 60;
    const freqSlice = dozenHistory.slice(-FREQ_WINDOW);
    if (freqSlice.length >= 15) {
      const fc = [0, 0, 0];
      freqSlice.forEach((v) => (fc[v] += 1));
      const ftot = fc[0] + fc[1] + fc[2];
      if (ftot > 0) {
        const fshare = fc.map((c) => c / ftot);
        // Centered advantage over uniform
        const adv = fshare.map((f) => f - 1 / 3);
        // Soft boost: multiply distribution by exp(scale * adv)
        const FREQ_BOOST_SCALE = 0.9; // aggressive but capped by renorm
        const mult = adv.map((a) =>
          Math.exp(Math.min(0.6, Math.max(-0.6, FREQ_BOOST_SCALE * a)))
        );
        for (let i = 0; i < 3; i++) finalDist[i] *= mult[i];
        const sf = finalDist[0] + finalDist[1] + finalDist[2];
        if (sf > 0) {
          finalDist = [finalDist[0] / sf, finalDist[1] / sf, finalDist[2] / sf];
        }
        // Due-frequency (undersampled) gentle encouragement: if a class is significantly below uniform (by gap), add a small additive mass proportional to gap and time since last occurrence.
        const timeSince = [0, 0, 0];
        // compute time since last occurrence for each dozen
        for (let i = dozenHistory.length - 1; i >= 0; i--) {
          const v = dozenHistory[i];
          if (timeSince[v] === 0) timeSince[v] = dozenHistory.length - i - 1;
          if (timeSince.every((t) => t > 0)) break;
        }
        const DUE_GAP_MIN = 0.1; // minimum gap below uniform to consider due
        const DUE_BASE_ADD = 0.04; // base additive probability mass
        const DUE_TIME_SCALE = 0.015; // incremental per spin absent
        let added = 0;
        for (let i = 0; i < 3; i++) {
          const gap = 1 / 3 - fshare[i];
          if (gap > DUE_GAP_MIN) {
            const add =
              DUE_BASE_ADD +
              Math.min(0.12, gap * 0.5) +
              Math.min(0.08, timeSince[i] * DUE_TIME_SCALE);
            finalDist[i] += add;
            added += add;
          }
        }
        if (added > 0) {
          const sDue = finalDist[0] + finalDist[1] + finalDist[2];
          finalDist = [
            finalDist[0] / sDue,
            finalDist[1] / sDue,
            finalDist[2] / sDue,
          ];
        }
      }
    }
    // 2. Alternation detection (simple ABAB or ABC rotation) to encourage switching when pattern strength high.
    const ALT_LOOKBACK = 8;
    const altSeq = dozenHistory.slice(-ALT_LOOKBACK);
    let alternationExpected: number | null = null;
    let alternationConfidence = 0;
    if (altSeq.length >= 4) {
      const uniq = Array.from(new Set(altSeq));
      if (uniq.length === 2) {
        // Test ABAB...
        const a = altSeq[0];
        const b = uniq.find((x) => x !== a)!;
        let matches = 0;
        for (let i = 0; i < altSeq.length; i++) {
          const expect = i % 2 === 0 ? a : b;
          if (altSeq[i] === expect) matches++;
        }
        const conf = matches / altSeq.length;
        if (conf >= 0.65) {
          const lastVal = altSeq[altSeq.length - 1];
          alternationExpected = lastVal === a ? b : a;
          alternationConfidence = conf;
        }
      } else if (uniq.length === 3) {
        // Check simple cyclic pattern (e.g., abcabc)
        const cycle = uniq; // approximate order by first occurrences
        let cycMatches = 0;
        for (let i = 0; i < altSeq.length; i++) {
          const expect = cycle[i % 3];
          if (altSeq[i] === expect) cycMatches++;
        }
        const cycConf = cycMatches / altSeq.length;
        if (cycConf >= 0.7) {
          const lastIdx = cycle.indexOf(altSeq[altSeq.length - 1]);
          if (lastIdx >= 0) {
            alternationExpected = cycle[(lastIdx + 1) % 3];
            alternationConfidence = cycConf;
          }
        }
      }
    }
    if (alternationExpected != null && alternationConfidence > 0) {
      // Apply alternation encouragement: boost expected and soften others proportionally.
      const ALT_BOOST_BASE = 0.25;
      const boost = ALT_BOOST_BASE * alternationConfidence; // scaled 0..ALT_BOOST_BASE
      const removed = boost * (1 - finalDist[alternationExpected]);
      finalDist[alternationExpected] += removed;
      const distribute = removed;
      const others = [0, 1, 2].filter((i) => i !== alternationExpected);
      // remove proportionally from others
      let otherSum = others.reduce((a, i) => a + finalDist[i], 0);
      if (otherSum > 0) {
        others.forEach((i) => {
          finalDist[i] -= (distribute * finalDist[i]) / otherSum;
        });
      }
      const sAlt = finalDist[0] + finalDist[1] + finalDist[2];
      finalDist = [
        finalDist[0] / sAlt,
        finalDist[1] / sAlt,
        finalDist[2] / sAlt,
      ];
    }
    // 3. Streak continuation vs break decision refinement: if current run length greater than 1 and alternation weak, reinforce continuation modestly.
    if (alternationConfidence < 0.55 && dozenHistory.length >= 2) {
      const lastVal = dozenHistory[dozenHistory.length - 1];
      let rl2 = 1;
      for (let i = dozenHistory.length - 2; i >= 0; i--) {
        if (dozenHistory[i] === lastVal) rl2++;
        else break;
      }
      if (rl2 >= 2) {
        const STREAK_BOOST = Math.min(0.18, (rl2 - 1) * 0.04);
        const gain = STREAK_BOOST * (1 - finalDist[lastVal]);
        finalDist[lastVal] += gain;
        const others = [0, 1, 2].filter((i) => i !== lastVal);
        const oSum = others.reduce((a, i) => a + finalDist[i], 0) || 1;
        others.forEach((i) => {
          finalDist[i] -= gain * (finalDist[i] / oSum);
        });
        const sSt = finalDist[0] + finalDist[1] + finalDist[2];
        finalDist = [
          finalDist[0] / sSt,
          finalDist[1] / sSt,
          finalDist[2] / sSt,
        ];
      }
    }
    // --- Debiasing & confidence shaping ---
    // 1. Inverse-frequency debias (mild) to prevent early lock-in
    const totalCounts = counts[0] + counts[1] + counts[2];
    if (totalCounts > 0) {
      const gamma = DOZEN_INV_FREQ_GAMMA; // softer debias
      const invFreqWeights = counts.map((c) =>
        Math.pow((totalCounts + 3) / (c + 1), gamma)
      );
      let wSum = invFreqWeights.reduce((a, b) => a + b, 0);
      const debiased = [0, 1, 2].map(
        (i) => (finalDist[i] * invFreqWeights[i]) / wSum
      ) as [number, number, number];
      finalDist = debiased;
    }
    // 2. Entropy floor (blend with uniform if over-confident too early)
    const entropy = -finalDist.reduce(
      (a, p) => (p > 0 ? a + p * Math.log2(p) : a),
      0
    );
    const maxEnt = Math.log2(3);
    const minAllowed = DOZEN_ENTROPY_FLOOR * maxEnt; // relaxed floor
    if (entropy < minAllowed && dozenHistory.length < DOZEN_ENTROPY_STOP) {
      const deficit = (minAllowed - entropy) / minAllowed;
      const scarcity =
        1 - Math.min(dozenHistory.length / DOZEN_ENTROPY_STOP, 1);
      const blend = Math.min(0.4, deficit * 0.6 * scarcity);
      finalDist = [0, 1, 2].map(
        (i) => (1 - blend) * finalDist[i] + blend * (1 / 3)
      ) as [number, number, number];
    }
    // 3. Hi-prob miss rate exploration boost: if recent hi-prob miss rate high, inject controlled flatten (pre-calibration stage)
    if (dozenHiProbMissRateRef.current > 0.65) {
      // Apply a soft temperature (power<1) then mild uniform mix
      let tDist = finalDist.map((p) => Math.pow(p, 0.9)) as [
        number,
        number,
        number
      ];
      const sT = tDist[0] + tDist[1] + tDist[2];
      tDist = [tDist[0] / sT, tDist[1] / sT, tDist[2] / sT];
      const uBlend =
        0.1 + Math.min(0.08, (dozenHiProbMissRateRef.current - 0.65) * 0.25);
      finalDist = tDist.map((p) => p * (1 - uBlend) + uBlend / 3) as [
        number,
        number,
        number
      ];
    }
    // 3. Prediction streak dampening: if same prediction repeats with mediocre realized performance, soften
    const ps = dozenPredStreakRef.current;
    const topIdx =
      finalDist[0] >= finalDist[1] && finalDist[0] >= finalDist[2]
        ? 0
        : finalDist[1] >= finalDist[2]
        ? 1
        : 2;
    // Anti-tunnel: penalize if we've repeatedly predicted this topIdx incorrectly many times in a row
    const es = dozenPredErrorStreakRef.current;
    if (
      es.last === topIdx &&
      es.incorrect >= DOZEN_ERR_STREAK_PENALTY_START &&
      dozenHistory.length > 10
    ) {
      const over = es.incorrect - (DOZEN_ERR_STREAK_PENALTY_START - 1);
      const frac = Math.min(
        DOZEN_ERR_STREAK_PENALTY_MAX,
        over * DOZEN_ERR_STREAK_PENALTY_STEP
      );
      const removed = finalDist[topIdx] * frac;
      finalDist[topIdx] -= removed;
      const addEach = removed / 2;
      for (let i = 0; i < 3; i++) if (i !== topIdx) finalDist[i] += addEach;
      // renormalize safety
      const sAnti = finalDist[0] + finalDist[1] + finalDist[2];
      finalDist = [
        finalDist[0] / sAnti,
        finalDist[1] / sAnti,
        finalDist[2] / sAnti,
      ];
    }
    if (ps.last === topIdx) {
      ps.len += 1;
    } else {
      ps.last = topIdx;
      ps.len = 1;
    }
    if (ps.len >= DOZEN_STREAK_DAMP_START && n < 120) {
      const damp = Math.min(
        0.12 + (ps.len - (DOZEN_STREAK_DAMP_START - 1)) * 0.025,
        DOZEN_STREAK_DAMP_MAX
      );
      const removed = finalDist[topIdx] * damp;
      const remain = finalDist[topIdx] - removed;
      const addEach = removed / 2;
      finalDist = [0, 1, 2].map((i) =>
        i === topIdx ? remain : finalDist[i] + addEach
      ) as [number, number, number];
    }
    // Temperature sharpening (softmax^(1/T)) after dampening
    if (n >= DOZEN_TEMP_START) {
      const prog = Math.min(
        1,
        (n - DOZEN_TEMP_START) / Math.max(1, DOZEN_TEMP_FULL - DOZEN_TEMP_START)
      );
      const T = 0.95 - prog * (0.95 - DOZEN_TEMP_MIN);
      const adj = finalDist.map((p) => Math.pow(Math.max(p, 1e-8), 1 / T));
      const sAdj = adj.reduce((a, b) => a + b, 0);
      if (sAdj > 0 && isFinite(sAdj)) {
        finalDist = adj.map((v) => v / sAdj) as [number, number, number];
      }
    }
    // Decision (confidence-aware & stochastic when low confidence)
    // --- Early-phase accuracy guard ---
    let guardActive = false;
    if (dozenRecentPredsRef.current.length >= 12) {
      const recentEval = dozenRecentPredsRef.current.slice(-18);
      const rAcc =
        recentEval.filter((r) => r.correct).length /
        Math.max(1, recentEval.length);
      if (rAcc < 0.32 && Math.max(...finalDist) < 0.55) {
        // Blend with recent empirical frequency to stabilize
        const FREQ_WIN = 30;
        const seg = dozenHistory.slice(-FREQ_WIN);
        if (seg.length >= 6) {
          const fc = [0, 0, 0];
          seg.forEach((v) => (fc[v] += 1));
          const sumF = fc[0] + fc[1] + fc[2];
          let freqDist: [number, number, number] = sumF
            ? [fc[0] / sumF, fc[1] / sumF, fc[2] / sumF]
            : [1 / 3, 1 / 3, 1 / 3];
          // Light smoothing toward uniform to avoid extreme early swings
          const SMOOTH = 0.25;
          freqDist = [0, 1, 2].map(
            (i) => (1 - SMOOTH) * freqDist[i] + SMOOTH * (1 / 3)
          ) as [number, number, number];
          const BLEND = 0.5; // strong correction when struggling
          finalDist = [0, 1, 2].map(
            (i) => (1 - BLEND) * finalDist[i] + BLEND * freqDist[i]
          ) as [number, number, number];
          guardActive = true;
        }
        // Per-class recent accuracy penalty (avoid tunneling on low performing class)
        const rc = [0, 0, 0];
        const rcCorrect = [0, 0, 0];
        recentEval.forEach((r) => {
          rc[r.label] = (rc[r.label] || 0) + 1;
          if (r.correct) rcCorrect[r.label] = (rcCorrect[r.label] || 0) + 1;
        });
        for (let i = 0; i < 3; i++) {
          const trials = rc[i];
          if (trials >= 3) {
            const acc = rcCorrect[i] / trials;
            if (acc < 0.25 && finalDist[i] > 0.34) {
              // below baseline & over uniform
              finalDist[i] *= 0.78; // shrink
            }
          }
        }
        const sG = finalDist[0] + finalDist[1] + finalDist[2];
        if (sG > 0)
          finalDist = [finalDist[0] / sG, finalDist[1] / sG, finalDist[2] / sG];
      }
    }
    if (guardActive) {
      dozenGuardRef.current.triggers += 1;
    }
    dozenGuardRef.current.lastActive = guardActive;
    // --- Regret suppression & calibration adjustments BEFORE chooseDozen ---
    (function applyRegretAndCalibration() {
      // 1. Regret-based suppression
      const reg = dozenRegretRef.current;
      const maxReg = Math.max(reg[0], reg[1], reg[2]);
      if (maxReg > DZ_REGRET_LIMIT) {
        const scaled = [0, 1, 2].map((i) => {
          const r = reg[i];
          if (r <= DZ_REGRET_LIMIT) return finalDist[i];
          const over = Math.min(r, DZ_REGRET_HARD) - DZ_REGRET_LIMIT;
          const factor = Math.max(0.35, Math.exp(-0.7 * over));
          return finalDist[i] * factor;
        }) as [number, number, number];
        const s = scaled[0] + scaled[1] + scaled[2];
        if (s > 0) finalDist = [scaled[0] / s, scaled[1] / s, scaled[2] / s];
      }
      // 2. High-confidence miscalibration clamp
      if (dozenHiConfWrongRef.current.length >= 12) {
        const errRate =
          dozenHiConfWrongRef.current.reduce((a, b) => a + b, 0) /
          dozenHiConfWrongRef.current.length;
        if (errRate > DZ_HICONF_ERRRATE) {
          const topIdx =
            finalDist[0] >= finalDist[1] && finalDist[0] >= finalDist[2]
              ? 0
              : finalDist[1] >= finalDist[2]
              ? 1
              : 2;
          if (finalDist[topIdx] > DZ_HICONF_CLAMP) {
            const cap = DZ_HICONF_CLAMP;
            const excess = finalDist[topIdx] - cap;
            const others = [0, 1, 2].filter((i) => i !== topIdx);
            finalDist[topIdx] = cap;
            finalDist[others[0]] += excess / 2;
            finalDist[others[1]] += excess / 2;
          }
        }
      }
      // 3. Majority fallback for low confidence
      const sortedLC = [...finalDist].sort((a, b) => b - a);
      const topLC = sortedLC[0],
        secondLC = sortedLC[1];
      if (
        topLC - secondLC < DZ_LOWCONF_GAP &&
        topLC < DZ_LOWCONF_TOP &&
        dozenRecentWindowRef.current.length >= 6
      ) {
        const freq: [number, number, number] = [0, 0, 0];
        dozenRecentWindowRef.current.forEach((c) => (freq[c] += 1));
        const majIdx =
          freq[0] >= freq[1] && freq[0] >= freq[2]
            ? 0
            : freq[1] >= freq[2]
            ? 1
            : 2;
        const currentTopIdx =
          finalDist[0] >= finalDist[1] && finalDist[0] >= finalDist[2]
            ? 0
            : finalDist[1] >= finalDist[2]
            ? 1
            : 2;
        if (majIdx !== currentTopIdx) {
          const blend = 0.3;
          finalDist = [0, 1, 2].map(
            (i) =>
              (1 - blend) * finalDist[i] + blend * (i === majIdx ? 0.6 : 0.2)
          ) as [number, number, number];
          const s2 = finalDist[0] + finalDist[1] + finalDist[2];
          finalDist = [finalDist[0] / s2, finalDist[1] / s2, finalDist[2] / s2];
        }
      }
      // 4. Earlier per-class miss streak cap enforcement
      const miss = dozenPerClassMissRef.current;
      const MISS_CAP_START = 2;
      for (let i = 0; i < 3; i++) {
        if (miss[i] >= MISS_CAP_START) {
          const topIdx =
            finalDist[0] >= finalDist[1] && finalDist[0] >= finalDist[2]
              ? 0
              : finalDist[1] >= finalDist[2]
              ? 1
              : 2;
          if (i === topIdx && finalDist[i] < 0.55) {
            const reduce = finalDist[i] * 0.25;
            finalDist[i] -= reduce;
            const others = [0, 1, 2].filter((c) => c !== i);
            finalDist[others[0]] += reduce / 2;
            finalDist[others[1]] += reduce / 2;
            const s3 = finalDist[0] + finalDist[1] + finalDist[2];
            finalDist = [
              finalDist[0] / s3,
              finalDist[1] / s3,
              finalDist[2] / s3,
            ];
          }
        }
      }
    })();
    // Adaptive hi-conf clamp earlier: compute dynamic clamp based on hi-conf wrong rate
    (function adaptiveHiConfClamp() {
      if (dozenHiConfWrongRef.current.length >= 8) {
        const errRate =
          dozenHiConfWrongRef.current.reduce((a, b) => a + b, 0) /
          dozenHiConfWrongRef.current.length;
        const dynamicClamp = Math.max(
          DZ_HICONF_CLAMP_MIN,
          DZ_EARLY_HICONF_CLAMP_BASE - Math.min(0.04, (errRate - 0.5) * 0.1)
        );
        const topIdx =
          finalDist[0] >= finalDist[1] && finalDist[0] >= finalDist[2]
            ? 0
            : finalDist[1] >= finalDist[2]
            ? 1
            : 2;
        if (finalDist[topIdx] > dynamicClamp) {
          const excess = finalDist[topIdx] - dynamicClamp;
          finalDist[topIdx] = dynamicClamp;
          const others = [0, 1, 2].filter((i) => i !== topIdx);
          finalDist[others[0]] += excess / 2;
          finalDist[others[1]] += excess / 2;
        }
      }
    })();
    // Brier score watchdog shrink
    (function brierShrink() {
      if (dozenBrierWindowRef.current.length >= DZ_BRIER_WINDOW / 2) {
        const recent = dozenBrierWindowRef.current.slice(-DZ_BRIER_WINDOW);
        const avg = recent.reduce((a, b) => a + b, 0) / recent.length;
        if (avg > DZ_BRIER_BASELINE) {
          const excess = avg - DZ_BRIER_BASELINE;
          // Dynamic shrink proportional to relative degradation
          const rel = excess / DZ_BRIER_BASELINE; // ~0.05 => mild
          const k = Math.min(
            DZ_BRIER_SHRINK_MAX,
            // base 0.15 plus scaled component
            0.15 + rel * 0.9
          );
          finalDist = [0, 1, 2].map(
            (i) => (1 - k) * finalDist[i] + k * (1 / 3)
          ) as [number, number, number];
          // Escalation: track sustained elevation and apply additional uniform blend + flattening
          let targetLevel: 0 | 1 | 2 = 0;
          if (avg > DZ_BRIER_BASELINE + 0.015)
            targetLevel = 2; // pronounced miscalibration
          else if (avg > DZ_BRIER_BASELINE + 0.008) targetLevel = 1; // mild
          const cur = dozenBrierAutoRef.current.level;
          if (targetLevel > cur) {
            dozenBrierAutoRef.current.level = (cur + 1) as 0 | 1 | 2; // step up one level only
          } else if (targetLevel < cur) {
            // step down only when comfortably recovered
            if (avg < DZ_BRIER_BASELINE + 0.006) {
              dozenBrierAutoRef.current.level = (cur - 1) as 0 | 1 | 2;
            }
          }
          const eff = dozenBrierAutoRef.current.level;
          if (eff > 0) {
            const autoBlend = eff === 1 ? 0.1 : 0.18;
            for (let i = 0; i < 3; i++) {
              finalDist[i] =
                (1 - autoBlend) * finalDist[i] + autoBlend * (1 / 3);
            }
            const T = eff === 1 ? 1.1 : 1.25;
            const adj = finalDist.map((p) =>
              Math.pow(Math.max(p, 1e-8), 1 / T)
            );
            const sT = adj[0] + adj[1] + adj[2];
            finalDist = [adj[0] / sT, adj[1] / sT, adj[2] / sT];
          }
        }
      }
    })();
    // Reliability & anti-tunnel shaping (pre overconfidence adaptive)
    (function reliabilityAndExploration() {
      const stats = dozenClassPerfRef.current;
      if (!finalDist) return;
      // Compute reliability ratios acc / meanProb (clamped) after minimum samples
      const reliability = [0, 1, 2].map((i) => {
        const s = stats[i];
        if (s.attempts < 8) return 1;
        const acc = s.wins / Math.max(1, s.attempts);
        const meanP = s.sumTopProb / Math.max(1e-6, s.attempts);
        return Math.max(0.4, Math.min(1.25, acc / Math.max(0.01, meanP)));
      });
      // Overconfidence gap proxy from top prob window for alpha
      const gapWindow = dozenTopProbWindowRef.current.slice(-50);
      let gap = 0;
      if (gapWindow.length >= 15) {
        const avgTop =
          gapWindow.reduce((a, r) => a + r.p, 0) / gapWindow.length;
        const accTop =
          gapWindow.filter((r) => r.correct).length / gapWindow.length;
        gap = avgTop - accTop;
      }
      const avgBrier = (() => {
        const arr = dozenBrierWindowRef.current.slice(-40);
        if (!arr.length) return DZ_BRIER_BASELINE;
        return arr.reduce((a, b) => a + b, 0) / arr.length;
      })();
      const alpha =
        0.9 +
        (gap > 0.03 ? Math.min(0.8, (gap - 0.03) * 5) : 0) +
        (avgBrier > DZ_BRIER_BASELINE + 0.012 ? 0.25 : 0);
      let adj = [...finalDist];
      for (let i = 0; i < 3; i++) {
        const r = reliability[i];
        const missStreak = stats[i].consecMisses;
        let logit = Math.log(Math.max(1e-9, adj[i]));
        logit += Math.log(r) * alpha;
        if (missStreak >= 2) {
          const penalty = Math.max(0.55, 1 - 0.15 * (missStreak - 1));
          logit += Math.log(penalty);
        }
        adj[i] = Math.exp(logit);
      }
      // Normalize
      let sum = adj[0] + adj[1] + adj[2];
      if (!(sum > 0)) adj = [1 / 3, 1 / 3, 1 / 3];
      else adj = adj.map((p) => p / sum) as typeof adj;
      // Tunnel detection: last 6 predictions identical & low acc
      const tw = dozenTunnelWindowRef.current;
      const k = 6;
      if (tw.length >= k) {
        const last = tw.slice(-k);
        const same = last.every((x) => x.cls === last[0].cls);
        if (same) {
          const acc = last.filter((x) => x.win).length / k;
          if (acc < 0.34) {
            const blend = Math.min(0.35, (0.34 - acc) * 1.2);
            for (let i = 0; i < 3; i++)
              adj[i] = adj[i] * (1 - blend) + blend / 3;
          }
        }
      }
      // Pattern repetition damp: repeated low-performing shape
      const h = dozenShapeHash(adj);
      dozenPatternWindowRef.current.push(h);
      if (dozenPatternWindowRef.current.length > 30)
        dozenPatternWindowRef.current.splice(
          0,
          dozenPatternWindowRef.current.length - 30
        );
      if (!dozenPatternPerfRef.current[h])
        dozenPatternPerfRef.current[h] = { tries: 0, wins: 0 };
      // (wins updated after outcome)
      const counts: Record<string, number> = {};
      dozenPatternWindowRef.current.forEach(
        (ph) => (counts[ph] = (counts[ph] || 0) + 1)
      );
      let damp = 0;
      for (const key in counts) {
        if (counts[key] >= 3) {
          const perf = dozenPatternPerfRef.current[key];
          const acc = perf.tries ? perf.wins / perf.tries : 1;
          if (perf.tries >= 3 && acc < 0.34) {
            damp = Math.max(damp, 0.2);
          }
        }
      }
      if (damp > 0) {
        for (let i = 0; i < 3; i++) adj[i] = adj[i] * (1 - damp) + damp / 3;
      }
      // Dynamic floor raise to prevent collapse
      const baseFloor = 0.055;
      const minFloor = Math.min(
        0.18,
        Math.max(
          baseFloor,
          baseFloor +
            gap * 0.5 +
            (avgBrier > DZ_BRIER_BASELINE + 0.015 ? 0.02 : 0)
        )
      );
      let add = 0;
      for (let i = 0; i < 3; i++)
        if (adj[i] < minFloor) {
          add += minFloor - adj[i];
          adj[i] = minFloor;
        }
      if (add > 0) {
        let surplus = 0;
        const over: number[] = [];
        for (let i = 0; i < 3; i++) {
          const s = adj[i] - minFloor;
          if (s > 1e-6) {
            surplus += s;
            over.push(i);
          }
        }
        if (surplus > 0) {
          over.forEach((i) => {
            const share = (adj[i] - minFloor) / surplus;
            adj[i] -= share * add;
          });
        }
      }
      // Exploration: if top-second gap small & reliability of top <0.9 soften
      const sorted = adj
        .map((p, i) => [p, i] as [number, number])
        .sort((a, b) => b[0] - a[0]);
      if (
        sorted[0][0] - sorted[1][0] < 0.04 &&
        reliability[sorted[0][1]] < 0.9
      ) {
        const soften = 0.5 * (0.04 - (sorted[0][0] - sorted[1][0]));
        adj[sorted[0][1]] -= soften;
        adj[sorted[1][1]] += soften;
      }
      // Final normalize after shaping
      const s2 = adj[0] + adj[1] + adj[2];
      finalDist = [adj[0] / s2, adj[1] / s2, adj[2] / s2];
      // expose debug metrics
      (dozenAdaptiveRef.current as any).classReliability = reliability;
      (dozenAdaptiveRef.current as any).minFloorDyn = Math.min.apply(
        null,
        finalDist
      );
    })();
    // Opportunity / bias rebalancing & confusion correction (post reliability shaping, pre alternation detection)
    (function opportunityBiasConfusion() {
      if (!finalDist) return;
      const spins = dozenHistory.length;
      if (spins < 60) return; // wait for sufficient evidence
      // Medium window actual share
      const ACT_WINDOW = 70;
      const actSlice = dozenHistory.slice(-ACT_WINDOW);
      if (actSlice.length < 30) return;
      const actCounts = [0, 0, 0];
      actSlice.forEach((v) => actCounts[v]++);
      const actTot = actCounts[0] + actCounts[1] + actCounts[2] || 1;
      const actShare = actCounts.map((c) => c / actTot) as [
        number,
        number,
        number
      ];
      // Predicted share (recent)
      const PRED_WINDOW = 70;
      const predSlice = dozenRecentPredsRef.current.slice(-PRED_WINDOW);
      const predCounts = [0, 0, 0];
      predSlice.forEach((r) => {
        if (r.label >= 0 && r.label <= 2) predCounts[r.label] += 1;
      });
      const predTot = predCounts[0] + predCounts[1] + predCounts[2] || 1;
      const predShare = predCounts.map((c) => c / predTot) as [
        number,
        number,
        number
      ];
      // Reliability proxy (already computed earlier & exposed) fallback to 1
      const classPerf = dozenClassPerfRef.current;
      const rel = [0, 1, 2].map((i) => {
        const s = classPerf[i];
        if (s.attempts < 10) return 1;
        const acc = s.wins / Math.max(1, s.attempts);
        const meanP = s.sumTopProb / Math.max(1e-6, s.attempts);
        return Math.max(0.4, Math.min(1.3, acc / Math.max(0.02, meanP)));
      }) as [number, number, number];
      // Opportunity uplift: if actual share > predicted share and reliability >=0.95 add small boost
      let added = [0, 0, 0];
      for (let i = 0; i < 3; i++) {
        const gap = actShare[i] - predShare[i];
        if (gap > 0.04 && rel[i] >= 0.95) {
          // scale boost by gap & reliability
          const boost = Math.min(0.04, gap * 0.35 * rel[i]);
          added[i] = boost;
          finalDist[i] += boost;
        }
      }
      // Collect total added to subtract proportionally from others above baseline
      const totalAdded = added[0] + added[1] + added[2];
      if (totalAdded > 0) {
        // remove from classes with highest probs
        const order = [0, 1, 2].sort((a, b) => finalDist[b] - finalDist[a]);
        let remaining = totalAdded;
        for (const i of order) {
          if (remaining <= 1e-6) break;
          const surplus = Math.max(0, finalDist[i] - 0.3333); // above uniform surrogate
          if (surplus > 0) {
            const take = Math.min(surplus, remaining);
            finalDist[i] -= take;
            remaining -= take;
          }
        }
      }
      // High-prob reliability gating: if top prob > 0.5 but reliability <1 clamp
      const topIdx =
        finalDist[0] >= finalDist[1] && finalDist[0] >= finalDist[2]
          ? 0
          : finalDist[1] >= finalDist[2]
          ? 1
          : 2;
      if (finalDist[topIdx] > 0.5 && rel[topIdx] < 1) {
        const target = 0.46 + (rel[topIdx] - 0.7) * 0.1; // rel 0.7->0.76, rel 1->0.49
        const clampTarget = Math.min(finalDist[topIdx], Math.max(0.42, target));
        if (clampTarget < finalDist[topIdx]) {
          const diff = finalDist[topIdx] - clampTarget;
          finalDist[topIdx] = clampTarget;
          const others = [0, 1, 2].filter((i) => i !== topIdx);
          finalDist[others[0]] += diff / 2;
          finalDist[others[1]] += diff / 2;
        }
      }
      // Confusion correction: persistent misclassification from i->j, shift tiny mass when predicting i in favor of j
      const conf = dozenConfusionRef.current;
      let confTotal = 0;
      for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++) confTotal += conf[i][j];
      if (confTotal > 150) {
        // need data
        for (let i = 0; i < 3; i++) {
          const rowSum = conf[i][0] + conf[i][1] + conf[i][2] || 1;
          for (let j = 0; j < 3; j++) {
            if (i === j) continue;
            const rate = conf[i][j] / rowSum; // how often actual=j when predicted=i
            if (rate > 0.5) {
              // shift small portion of prob(i) toward j
              const shift = Math.min(0.03, (rate - 0.5) * 0.04);
              const amt = finalDist[i] * shift;
              finalDist[i] -= amt;
              finalDist[j] += amt;
            }
          }
        }
      }
      // Renormalize
      const s = finalDist[0] + finalDist[1] + finalDist[2];
      if (s > 0) {
        finalDist = [finalDist[0] / s, finalDist[1] / s, finalDist[2] / s];
      }
      // Expose debug
      (dozenAdaptiveRef.current as any).oppGap = actShare.map(
        (a, i) => +(a - predShare[i]).toFixed(3)
      );
      (dozenAdaptiveRef.current as any).rel = rel;
    })();
    // Alternation regime detection (last 6 actual spins include all three & recent win rate low)
    (function alternationDetect() {
      if (dozenHistory.length >= 6) {
        const last6 = dozenHistory.slice(-6);
        if (new Set(last6).size === 3) {
          const recs = dozenRecords.slice(-6);
          const acc =
            recs.filter((r) => r.correct).length / Math.max(1, recs.length);
          dozenAlternationRef.current.active = acc < 2 / 6;
        } else {
          dozenAlternationRef.current.active = false;
        }
      }
    })();
    if (dozenAlternationRef.current.active) {
      finalDist = [0, 1, 2].map((i) => finalDist[i] * 0.6 + 0.4 / 3) as [
        number,
        number,
        number
      ];
    }
    // Pattern ordinal shape damp (low performing shape)
    (function shapeDamp() {
      const key = finalDist
        .map((p, i) => [p, i])
        .sort((a, b) => b[0] - a[0])
        .map((x) => x[1])
        .join("");
      const perf = dozenShapePerfRef.current[key];
      if (perf && perf.tries >= 4) {
        const acc = perf.wins / Math.max(1, perf.tries);
        if (acc < 0.3) {
          finalDist = [0, 1, 2].map((i) => finalDist[i] * 0.85 + 0.15 / 3) as [
            number,
            number,
            number
          ];
          (dozenAdaptiveRef.current as any).patternDampActive = true;
        } else (dozenAdaptiveRef.current as any).patternDampActive = false;
      }
    })();
    // Calibration bins remap (pull top prob toward observed)
    (function calibRemap() {
      const topIdx =
        finalDist[0] >= finalDist[1] && finalDist[0] >= finalDist[2]
          ? 0
          : finalDist[1] >= finalDist[2]
          ? 1
          : 2;
      const topP = finalDist[topIdx];
      const bin = dozenCalibBinsRef.current.find(
        (b) => topP >= b.low && topP < b.high
      );
      // Placeholder: advanced bin remap removed until new drift logic integrated
      (dozenAdaptiveRef.current as any).calibBins =
        dozenCalibBinsRef.current.map((b) => ({ r: [b.low, b.high] }));
    })();
    // Adaptive overconfidence clamp based on realized gap history BEFORE decision helper
    (function overconfidenceAdaptive() {
      const w = dozenTopProbWindowRef.current.slice(-50);
      if (w.length >= 15) {
        const avgTop = w.reduce((a, r) => a + r.p, 0) / w.length;
        const accTop = w.filter((r) => r.correct).length / w.length;
        const gap = avgTop - accTop; // >0 => overconfident
        if (gap > 0.04) {
          // compute target cap relative to accuracy so probabilities cannot exceed plausible calibration
          const dynamicCap = Math.min(
            0.6,
            Math.max(
              0.38,
              accTop + 0.07 + Math.min(0.06, (0.08 - Math.min(gap, 0.08)) * 0.4)
            )
          );
          const topIdx =
            finalDist[0] >= finalDist[1] && finalDist[0] >= finalDist[2]
              ? 0
              : finalDist[1] >= finalDist[2]
              ? 1
              : 2;
          if (finalDist[topIdx] > dynamicCap) {
            const excess = finalDist[topIdx] - dynamicCap;
            finalDist[topIdx] = dynamicCap;
            const others = [0, 1, 2].filter((i) => i !== topIdx);
            finalDist[others[0]] += excess / 2;
            finalDist[others[1]] += excess / 2;
            const s = finalDist[0] + finalDist[1] + finalDist[2];
            finalDist = [finalDist[0] / s, finalDist[1] / s, finalDist[2] / s];
          }
          // Flatten further via temperature raise depending on gap magnitude
          const T = 1 + Math.min(0.7, (gap - 0.04) * 4); // up to 1.7
          if (T > 1.01) {
            const adj = finalDist.map((p) =>
              Math.pow(Math.max(p, 1e-8), 1 / T)
            );
            const sT = adj[0] + adj[1] + adj[2];
            finalDist = [adj[0] / sT, adj[1] / sT, adj[2] / sT];
          }
        } else if (gap < -0.05) {
          // Underconfident: mild sharpening (temperature <1)
          const T = Math.max(0.85, 1 + gap * 0.8); // gap negative
          if (T < 0.995) {
            const adj = finalDist.map((p) =>
              Math.pow(Math.max(p, 1e-8), 1 / T)
            );
            const sT = adj[0] + adj[1] + adj[2];
            finalDist = [adj[0] / sT, adj[1] / sT, adj[2] / sT];
          }
        }
      }
    })();
    // High-probability miss cooldown flatten (applied if active)
    if (dozenHiProbCooldownRef.current.spins > 0) {
      const factor = dozenHiProbCooldownRef.current.factor; // >1 => flatten
      const adj = finalDist.map((p) => Math.pow(Math.max(p, 1e-8), 1 / factor));
      const sC = adj[0] + adj[1] + adj[2];
      finalDist = [adj[0] / sC, adj[1] / sC, adj[2] / sC];
      dozenHiProbCooldownRef.current.spins -= 1;
    }
    // Scheduled early confidence cap (progressively relaxes)
    (function earlyConfidenceSchedule() {
      if (n >= 3) {
        const capStart = 0.5; // very early
        const capEnd = 0.64; // later allowable cap
        const prog = Math.min(1, n / 160);
        const cap = capStart + (capEnd - capStart) * prog;
        const topIdx =
          finalDist[0] >= finalDist[1] && finalDist[0] >= finalDist[2]
            ? 0
            : finalDist[1] >= finalDist[2]
            ? 1
            : 2;
        if (finalDist[topIdx] > cap) {
          const excess = finalDist[topIdx] - cap;
          finalDist[topIdx] = cap;
          const others = [0, 1, 2].filter((i) => i !== topIdx);
          finalDist[others[0]] += excess / 2;
          finalDist[others[1]] += excess / 2;
          const s = finalDist[0] + finalDist[1] + finalDist[2];
          finalDist = [finalDist[0] / s, finalDist[1] / s, finalDist[2] / s];
        }
      }
    })();
    const decide = chooseDozen(
      finalDist,
      counts as [number, number, number],
      dozenRegimeRef.current.regime
    );
    finalDist = decide.dist;
    const pred = decide.pred;
    dozenDecisionInfoRef.current = decide.info;
    // Early anti-tunnel: if last two incorrect on same dozen with low prob, force alternate best
    if (
      dozenPredErrorStreakRef.current.last === pred &&
      dozenPredErrorStreakRef.current.incorrect >= 2 &&
      Math.max(...finalDist) < 0.5
    ) {
      const alts = [0, 1, 2]
        .filter((i) => i !== pred)
        .map((i) => ({ i, p: finalDist[i] }))
        .sort((a, b) => b.p - a.p);
      if (alts.length) {
        const forced = alts[0].i;
        // small transfer
        const tr = Math.min(0.12, finalDist[pred] * 0.4);
        finalDist[pred] -= tr;
        finalDist[forced] += tr;
        const sf2 = finalDist[0] + finalDist[1] + finalDist[2];
        finalDist = [
          finalDist[0] / sf2,
          finalDist[1] / sf2,
          finalDist[2] / sf2,
        ];
        ps.last = forced;
        ps.len = 1;
        dozenPredictionRef.current = {
          probs: finalDist,
          predLabel: forced as 0 | 1 | 2,
        };
      } else {
        dozenPredictionRef.current = { probs: finalDist, predLabel: pred };
      }
    } else {
      dozenPredictionRef.current = { probs: finalDist, predLabel: pred };
    }
    // --- AUTO SECONDARY ROTATION (pre final safety clamp) ---
    (function autoSecondaryRotate() {
      if (!finalDist) return;
      // Only consider if cooldown expired
      if (dozenRotateCooldownRef.current > 0) return;
      const es = dozenPredErrorStreakRef.current; // streak up to previous outcome
      const currentPred = dozenPredictionRef.current.predLabel;
      if (currentPred == null) return;
      // Require at least 2 prior consecutive misses on this predicted class
      if (es.last === currentPred && es.incorrect >= 2) {
        // Evaluate margin
        const ordered = [0, 1, 2]
          .map((i) => ({ i, p: finalDist[i] }))
          .sort((a, b) => b.p - a.p);
        const top = ordered[0];
        const second = ordered[1];
        const margin = top.p - second.p;
        const MARGIN_MAX = 0.09; // only rotate if advantage small
        if (margin < MARGIN_MAX) {
          // Simple reliability heuristic for each class (wins/attempts)
          const perfTop = dozenClassPerfRef.current[top.i];
          const perfSecond = dozenClassPerfRef.current[second.i];
          const accTop = perfTop.attempts
            ? perfTop.wins / perfTop.attempts
            : 0.34;
          const accSecond = perfSecond.attempts
            ? perfSecond.wins / perfSecond.attempts
            : 0.34;
          // Rotate if second not worse and either (a) second accuracy >= topAcc - 0.03 OR (b) second accuracy substantially better
          if (accSecond >= accTop - 0.03 || accSecond > accTop + 0.04) {
            // Transfer enough mass to invert ranking plus small epsilon
            const need = margin + 0.002;
            const transfer = Math.min(need, top.p * 0.6); // don't over-shift
            finalDist[top.i] -= transfer;
            finalDist[second.i] += transfer;
            // renormalize
            const s = finalDist[0] + finalDist[1] + finalDist[2];
            finalDist = [finalDist[0] / s, finalDist[1] / s, finalDist[2] / s];
            dozenPredictionRef.current = {
              probs: finalDist as [number, number, number],
              predLabel: second.i as 0 | 1 | 2,
            };
            (dozenAdaptiveRef.current as any).rotated = {
              from: top.i,
              to: second.i,
              margin: +margin.toFixed(3),
              accTop: +accTop.toFixed(3),
              accSecond: +accSecond.toFixed(3),
            };
            dozenRotateCooldownRef.current = 3; // cooldown several spins
          }
        }
      }
    })();
    // --- FINAL POST-CHOICE SAFETY CLAMP & BRIER SYNERGY FLATTEN ---
    (function finalSafety() {
      // Stronger global cap applied AFTER chooseDozen and all shaping, so later steps cannot re-sharpen unchecked.
      const gapWindow = dozenTopProbWindowRef.current.slice(-60);
      let accTop = 0;
      let avgTop = 0;
      if (gapWindow.length >= 15) {
        avgTop = gapWindow.reduce((a, r) => a + r.p, 0) / gapWindow.length;
        accTop = gapWindow.filter((r) => r.correct).length / gapWindow.length;
      }
      const gap = avgTop - accTop;
      // Derive dynamic cap referencing realized accuracy; tighter if gap large or recent slump.
      let baseCap = accTop > 0 ? accTop + 0.07 : 0.4;
      if (gap > 0.08) baseCap = accTop + 0.05;
      else if (gap > 0.05) baseCap = accTop + 0.06;
      // Apply progressive decay multiplier
      baseCap *= dozenCapDecayRef.current.mult;
      // Shock override forces lower cap
      // Shock cap clamp: support both legacy numeric active and extended object
      const shock = dozenShockRef.current as any;
      const shockActive =
        typeof shock.active === "number" ? shock.active > 0 : !!shock.active;
      if (shockActive) baseCap = Math.min(baseCap, 0.4);
      // Guardrails
      const nSpins = dozenHistory.length;
      const maxCap = nSpins < 160 ? 0.58 : 0.6; // relax slightly later
      const minCap = 0.36;
      let cap = Math.min(maxCap, Math.max(minCap, baseCap));
      // Hi-confidence accuracy gating: if hi-conf subset underperforms, clip further
      const hi = dozenHiConfWindowRef.current.slice(-70);
      if (hi.length >= 12) {
        const hiAcc = hi.filter((r) => r.correct).length / hi.length;
        if (hiAcc < 0.4) cap -= 0.02;
        if (hiAcc < 0.34) cap -= 0.02; // deeper penalty
      }
      cap = Math.min(maxCap, Math.max(minCap, cap));
      // Apply cap if needed
      const topIdx =
        finalDist[0] >= finalDist[1] && finalDist[0] >= finalDist[2]
          ? 0
          : finalDist[1] >= finalDist[2]
          ? 1
          : 2;
      if (finalDist[topIdx] > cap) {
        const excess = finalDist[topIdx] - cap;
        finalDist[topIdx] = cap;
        const others = [0, 1, 2].filter((i) => i !== topIdx);
        finalDist[others[0]] += excess / 2;
        finalDist[others[1]] += excess / 2;
      }
      // Micro-shrink on marginal evidence after previous hi-prob miss
      const sorted = [0, 1, 2]
        .map((i) => [finalDist[i], i] as [number, number])
        .sort((a, b) => b[0] - a[0]);
      const margin = sorted[0][0] - sorted[1][0];
      if (dozenLastHiProbMissRef.current && margin < 0.035) {
        for (let i = 0; i < 3; i++) finalDist[i] = finalDist[i] * 0.7 + 0.3 / 3;
        const sM = finalDist[0] + finalDist[1] + finalDist[2];
        finalDist = [finalDist[0] / sM, finalDist[1] / sM, finalDist[2] / sM];
      }
      // Shock entropy enforcement (~1.05 bits minimum) if shock active
      const shockObj = dozenShockRef.current as any;
      if (shockObj && shockObj.active) {
        const entropy = -finalDist.reduce(
          (a, p) => a + (p > 0 ? p * Math.log2(p) : 0),
          0
        );
        if (entropy < 1.05) {
          const need = 1.05 - entropy;
          const alpha = Math.min(0.35, need * 0.55 + 0.12); // 12%+ depending on deficit
          for (let i = 0; i < 3; i++)
            finalDist[i] = finalDist[i] * (1 - alpha) + alpha / 3;
          const sE = finalDist[0] + finalDist[1] + finalDist[2];
          finalDist = [finalDist[0] / sE, finalDist[1] / sE, finalDist[2] / sE];
        }
      }
      // Abstain logic (experimental): abstain if margin very low AND hi-prob miss regime active OR shock active
      const entropyNow = -finalDist.reduce(
        (a, p) => a + (p > 0 ? p * Math.log2(p) : 0),
        0
      );
      const maxProb = Math.max(...finalDist);
      const sortedA = [...finalDist].sort((a, b) => b - a);
      const curMargin = sortedA[0] - sortedA[1];
      const abstainTrigger =
        (curMargin < 0.022 &&
          (dozenHiProbMissRateRef.current > 0.7 ||
            (shockObj && shockObj.active))) ||
        (entropyNow > 1.45 && curMargin < 0.015);
      dozenAbstainRef.current = abstainTrigger;
      // Strengthened Brier-based flatten (synergy with gap): if average Brier over window still > baseline, blend further toward uniform.
      let appliedShrink = 0;
      if (dozenBrierWindowRef.current.length >= DZ_BRIER_WINDOW / 2) {
        const recentB = dozenBrierWindowRef.current.slice(-DZ_BRIER_WINDOW);
        const bAvg = recentB.reduce((a, b) => a + b, 0) / recentB.length;
        if (bAvg > DZ_BRIER_BASELINE + 0.01) {
          const rel = (bAvg - DZ_BRIER_BASELINE) / DZ_BRIER_BASELINE; // relative degradation
          // Increase shrink ceiling when both gap & Brier poor
          const gapBoost = gap > 0.06 ? 0.15 : gap > 0.04 ? 0.07 : 0;
          const k = Math.min(0.45, 0.18 + rel * 1.1 + gapBoost);
          appliedShrink = k;
          for (let i = 0; i < 3; i++) {
            finalDist[i] = (1 - k) * finalDist[i] + k * (1 / 3);
          }
        }
      }
      // Dynamic probability floor tied to realized frequencies (rolling window)
      const FREQ_FLOOR_WINDOW = 30;
      const freqWindow = dozenHistory.slice(-FREQ_FLOOR_WINDOW);
      let floor = 0.055;
      if (freqWindow.length >= 12) {
        const counts = [0, 0, 0];
        freqWindow.forEach((v) => (counts[v] += 1));
        const tot = counts[0] + counts[1] + counts[2] || 1;
        const shares = counts.map((c) => c / tot);
        // floor per class = max(0.10, 0.5 * actualShare)
        // we'll apply a class-specific floor; first gather needed raises
        const classFloors = shares.map((s) => Math.max(0.1, 0.5 * s));
        // apply per-class floors
        let addedDyn = 0;
        for (let i = 0; i < 3; i++) {
          if (finalDist[i] < classFloors[i]) {
            addedDyn += classFloors[i] - finalDist[i];
            finalDist[i] = classFloors[i];
          }
        }
        if (addedDyn > 0) {
          // remove proportionally from classes above their floor
          let surplusIdx: number[] = [];
          let surplusTotal = 0;
          for (let i = 0; i < 3; i++) {
            const floorI = classFloors[i];
            const surplus = finalDist[i] - floorI;
            if (surplus > 1e-6) {
              surplusIdx.push(i);
              surplusTotal += surplus;
            }
          }
          if (surplusTotal > 0) {
            surplusIdx.forEach((i) => {
              const floorI = classFloors[i];
              const share = (finalDist[i] - floorI) / surplusTotal;
              finalDist[i] -= share * addedDyn;
            });
          }
          const sDf = finalDist[0] + finalDist[1] + finalDist[2];
          finalDist = [
            finalDist[0] / sDf,
            finalDist[1] / sDf,
            finalDist[2] / sDf,
          ];
        }
        // store average floor used for metrics
        floor = classFloors.reduce((a, b) => a + b, 0) / 3;
      } else {
        // fallback legacy simple floor
        const FLOOR_BASE = 0.055;
        floor = gap > 0.06 ? FLOOR_BASE + 0.015 : FLOOR_BASE;
        let added2 = 0;
        for (let i = 0; i < 3; i++)
          if (finalDist[i] < floor) {
            added2 += floor - finalDist[i];
            finalDist[i] = floor;
          }
        if (added2 > 0) {
          let surplusIdx2: number[] = [];
          let surplusTotal2 = 0;
          for (let i = 0; i < 3; i++) {
            const surplus = finalDist[i] - floor;
            if (surplus > 1e-6) {
              surplusIdx2.push(i);
              surplusTotal2 += surplus;
            }
          }
          if (surplusTotal2 > 0) {
            surplusIdx2.forEach((i) => {
              const share = (finalDist[i] - floor) / surplusTotal2;
              finalDist[i] -= share * added2;
            });
          }
          const sF2 = finalDist[0] + finalDist[1] + finalDist[2];
          finalDist = [
            finalDist[0] / sF2,
            finalDist[1] / sF2,
            finalDist[2] / sF2,
          ];
        }
      }
      // Ensure predicted share coverage: predicted probability each class >= 60% of its realized rolling share after sufficient spins
      if (dozenHistory.length >= 40) {
        const coverWindow = dozenHistory.slice(-FREQ_FLOOR_WINDOW);
        if (coverWindow.length >= 15) {
          const cCounts = [0, 0, 0];
          coverWindow.forEach((v) => (cCounts[v] += 1));
          const ct = cCounts[0] + cCounts[1] + cCounts[2] || 1;
          const cShares = cCounts.map((c) => c / ct);
          let adjAdded = 0;
          const minCov: number[] = cShares.map((s) => 0.6 * s);
          for (let i = 0; i < 3; i++) {
            if (finalDist[i] < minCov[i]) {
              adjAdded += minCov[i] - finalDist[i];
              finalDist[i] = minCov[i];
            }
          }
          if (adjAdded > 0) {
            // remove proportionally from classes above minCov
            let idxs: number[] = [];
            let totalSur = 0;
            for (let i = 0; i < 3; i++) {
              const sur = finalDist[i] - minCov[i];
              if (sur > 1e-6) {
                idxs.push(i);
                totalSur += sur;
              }
            }
            if (totalSur > 0)
              idxs.forEach((i) => {
                const share = (finalDist[i] - minCov[i]) / totalSur;
                finalDist[i] -= share * adjAdded;
              });
            const sCv = finalDist[0] + finalDist[1] + finalDist[2];
            finalDist = [
              finalDist[0] / sCv,
              finalDist[1] / sCv,
              finalDist[2] / sCv,
            ];
          }
        }
      }
      // Final renormalization
      const sF = finalDist[0] + finalDist[1] + finalDist[2];
      finalDist = [finalDist[0] / sF, finalDist[1] / sF, finalDist[2] / sF];
      // record metrics
      dozenAdaptiveRef.current.cap = cap;
      dozenAdaptiveRef.current.floor = floor;
      if (appliedShrink > 0) dozenAdaptiveRef.current.shrink = appliedShrink;
      (dozenAdaptiveRef.current as any).brierLevel =
        dozenBrierAutoRef.current.level;
    })();
    dozenProbsRef.current = finalDist;
    setDozenVersion((v) => v + 1);
  }, [dozenHistory, buildDozenFeatureWindow, ensureDozenModel, windowSize]);

  // Effect for dozens mode training/prediction
  useEffect(() => {
    if (
      controlsRef.current.mode === "roulette" &&
      controlsRef.current.rouletteVariant === "dozens"
    ) {
      trainAndPredictDozens();
    }
  }, [dozenHistory, trainAndPredictDozens]);

  // Reset dozens-specific state when switching into dozens mode (fresh session to avoid stale blending artifacts)
  const lastVariantRef = useRef<string | null>(null);
  useEffect(() => {
    const curKey = `${controlsRef.current.mode}:${controlsRef.current.rouletteVariant}`;
    if (curKey !== lastVariantRef.current) {
      if (
        controlsRef.current.mode === "roulette" &&
        controlsRef.current.rouletteVariant === "dozens"
      ) {
        // reset dozen tracking (but keep model weights optionally)
        setDozenHistory([]);
        setDozenRecords([]);
        dozenProbsRef.current = null;
        dozenPredictionRef.current = { probs: null, predLabel: null };
        setDozenVersion((v) => v + 1);
        dozenMarkovRef.current = [
          [1, 1, 1],
          [1, 1, 1],
          [1, 1, 1],
        ];
        dozenRunPostRef.current = {};
        dozenEwmaRef.current = null;
        dozenAccRef.current = { correct: 0, total: 0 };
      }
      lastVariantRef.current = curKey;
    }
  }, [controlsRef.current.mode, controlsRef.current.rouletteVariant]);

  // moved addObservation after dependencies

  const addObservation = useCallback(
    (label: 0 | 1) => {
      const prev = lastPredictionRef.current;
      const prevProb = prev.effectiveProb;
      const predLabel =
        prevProb != null ? (prevProb >= prev.threshold ? 1 : 0) : null;
      const correct = predLabel != null ? predLabel === label : null;
      if (correct != null) {
        accRef.current.total += 1;
        if (correct) accRef.current.correct += 1;
      }
      if (predLabel != null) {
        setMetrics((m) => {
          let {
            longestWin,
            longestLose,
            currentWin,
            currentLose,
            overPredSuccess,
            overPredTotal,
            sumWin,
            countWin,
            sumLose,
            countLose,
            sumActualRuns,
            countActualRuns,
          } = m;
          if (predLabel === 1) {
            overPredTotal += 1;
            if (correct) overPredSuccess += 1;
          }
          // Handle prediction streak transitions
          // If current losing streak ends because we got a correct prediction now
          if (correct) {
            if (currentLose > 0) {
              sumLose += currentLose;
              countLose += 1;
              currentLose = 0;
            }
            currentWin += 1;
            if (currentWin > longestWin) longestWin = currentWin;
          } else if (correct === false) {
            if (currentWin > 0) {
              sumWin += currentWin;
              countWin += 1;
              currentWin = 0;
            }
            currentLose += 1;
            if (currentLose > longestLose) longestLose = currentLose;
          }
          return {
            longestWin,
            longestLose,
            currentWin,
            currentLose,
            overPredSuccess,
            overPredTotal,
            sumWin,
            countWin,
            sumLose,
            countLose,
            sumActualRuns,
            countActualRuns,
            sumOverRuns: m.sumOverRuns,
            countOverRuns: m.countOverRuns,
            sumBelowRuns: m.sumBelowRuns,
            countBelowRuns: m.countBelowRuns,
          };
        });
      }
      const rlAllowed = history.length >= windowSize * 2; // freeze RL until sufficient samples
      if (
        rlAllowed &&
        controlsRef.current.rlEnabled &&
        probPartsRef.current &&
        predLabel === 1
      ) {
        const eta = controlsRef.current.rlEta;
        const src = probPartsRef.current;
        const sign = correct ? 1 : -1;
        const aligns: Record<string, number> = {
          model: label === 1 ? src.model : 1 - src.model,
          markov: label === 1 ? src.markov : 1 - src.markov,
          streak: label === 1 ? src.streak : 1 - src.streak,
          pattern: label === 1 ? src.pattern : 1 - src.pattern,
          ewma: label === 1 ? src.ewma : 1 - src.ewma,
          bayes: label === 1 ? src.bayes : 1 - src.bayes,
          entropyMod: label === 1 ? src.entropyMod : 1 - src.entropyMod,
        };
        (
          Object.keys(
            rlWeightsRef.current
          ) as (keyof typeof rlWeightsRef.current)[]
        ).forEach((k) => {
          const delta = Math.exp(eta * sign * (aligns[k] - 0.5));
          rlWeightsRef.current[k] = Math.min(
            4,
            Math.max(0.25, rlWeightsRef.current[k] * delta)
          );
        });
        const sum =
          Object.values(rlWeightsRef.current).reduce((a, b) => a + b, 0) || 1;
        (
          Object.keys(
            rlWeightsRef.current
          ) as (keyof typeof rlWeightsRef.current)[]
        ).forEach((k) => {
          rlWeightsRef.current[k] /= sum / 7; // 7 sources
        });
      } else if (
        rlAllowed &&
        controlsRef.current.rlEnabled &&
        predLabel === 0
      ) {
        (
          Object.keys(
            rlWeightsRef.current
          ) as (keyof typeof rlWeightsRef.current)[]
        ).forEach((k) => {
          rlWeightsRef.current[k] = rlWeightsRef.current[k] * 0.98 + 0.02 * 1;
        });
      }
      setRecords((r) => [
        ...r,
        {
          label,
          predLabel,
          prob: prevProb,
          rawProb: prev.rawProb,
          thresholdUsed: prev.threshold,
          invertUsed: prev.invertUsed,
          correct,
        },
      ]);
      setHistory((prevH) => {
        if (prevH.length) {
          const last = prevH[prevH.length - 1];
          markovCountsRef.current[last][label] += 1;
          let rl = 1;
          for (let i = prevH.length - 2; i >= 0; i--) {
            if (prevH[i] === last) rl++;
            else break;
          }
          const key = `${last}:${rl}`;
          const rec = streakPosteriorsRef.current[key] || { cont: 0, rev: 0 };
          if (label === last) rec.cont += 1;
          else rec.rev += 1;
          streakPosteriorsRef.current[key] = rec;
        }
        const alpha = controlsRef.current.ewmaAlpha;
        ewmaRef.current =
          ewmaRef.current == null
            ? label
            : (1 - alpha) * ewmaRef.current + alpha * label;
        const updated = [...prevH, label];
        // Track actual outcome run lengths (when a run ends, accumulate stats)
        if (updated.length >= 2) {
          const prevVal = updated[updated.length - 2];
          const curVal = label;
          if (curVal !== prevVal) {
            // run ended at previous element; compute its length
            let runLen = 1;
            for (let i = updated.length - 2; i > 0; i--) {
              if (updated[i - 1] === prevVal) runLen++;
              else break;
            }
            setMetrics((m) => ({
              ...m,
              sumActualRuns: m.sumActualRuns + runLen,
              countActualRuns: m.countActualRuns + 1,
              sumOverRuns:
                prevVal === 1 ? m.sumOverRuns + runLen : m.sumOverRuns,
              countOverRuns:
                prevVal === 1 ? m.countOverRuns + 1 : m.countOverRuns,
              sumBelowRuns:
                prevVal === 0 ? m.sumBelowRuns + runLen : m.sumBelowRuns,
              countBelowRuns:
                prevVal === 0 ? m.countBelowRuns + 1 : m.countBelowRuns,
            }));
          }
        }
        const cfg = biasConfigRef.current;
        const N = updated.length;
        if (N >= 5) {
          const w = Math.min(cfg.window, N);
          const recentActual = updated.slice(-w);
          const actualRate = recentActual.reduce((a, b) => a + b, 0) / w;
          const recentRecords = records.slice(-w);
          const predValid = recentRecords.filter(
            (r) => r.predLabel != null
          ) as { predLabel: 0 | 1 }[];
          const predRate = predValid.length
            ? predValid.reduce((a, b) => a + b.predLabel, 0) / predValid.length
            : null;
          let status: typeof biasInfo.status = "neutral";
          if (predRate != null) {
            if (predRate > actualRate + cfg.threshold) status = "overBias";
            else if (predRate < actualRate - cfg.threshold)
              status = "underBias";
          }
          if (status === "neutral") biasStableNeutralRef.current += 1;
          else biasStableNeutralRef.current = 0;
          if (
            (status === "overBias" || status === "underBias") &&
            !biasOverrideRef.current
          )
            biasOverrideRef.current = true;
          if (
            biasOverrideRef.current &&
            status === "neutral" &&
            biasStableNeutralRef.current >= 5
          )
            biasOverrideRef.current = false;
          setBiasInfo({
            status,
            gap: +(predRate != null ? predRate - actualRate : 0).toFixed(3),
          });
        }
        controlsRef.current.threshold = 0.5;
        return updated;
      });
    },
    [probOver, records, biasInfo.status]
  );

  const resetModel = () => {
    modelRef.current?.dispose();
    modelRef.current = null;
    setProbOver(null);
    setHistory([]);
    setRecords([]);
    rlWeightsRef.current = {
      model: 1,
      markov: 1,
      streak: 1,
      pattern: 1,
      ewma: 1,
      bayes: 1,
      entropyMod: 1,
    };
    ewmaRef.current = null;
    markovCountsRef.current = [
      [1, 1],
      [1, 1],
    ];
    streakPosteriorsRef.current = {};
    accRef.current = { correct: 0, total: 0 };
    lastTrainCountRef.current = 0;
    recomputePendingRef.current = false;
  };

  useEffect(() => {
    (async () => {
      await ensureBackend();
      // One-time load persisted state (histories & models)
      if (!backendReadyRef.current) return;
      try {
        // Histories
        if (history.length === 0) {
          const h = localStorage.getItem(PKEY("bin_history"));
          if (h) {
            const arr = JSON.parse(h);
            if (Array.isArray(arr) && arr.every((x) => x === 0 || x === 1)) {
              setHistory(arr.slice(-PERSIST_LIMIT));
            }
          }
          const recs = localStorage.getItem(PKEY("bin_records"));
          if (recs) {
            const r = JSON.parse(recs);
            if (Array.isArray(r)) setRecords(r.slice(-PERSIST_LIMIT));
          }
        }
        if (dozenHistory.length === 0) {
          const dh = localStorage.getItem(PKEY("dz_history"));
          if (dh) {
            const arr = JSON.parse(dh);
            if (
              Array.isArray(arr) &&
              arr.every((x) => x === 0 || x === 1 || x === 2)
            ) {
              setDozenHistory(arr.slice(-PERSIST_LIMIT));
            }
          }
          const dr = localStorage.getItem(PKEY("dz_records"));
          if (dr) {
            const r = JSON.parse(dr);
            if (Array.isArray(r)) setDozenRecords(r.slice(-PERSIST_LIMIT));
          }
          // Adaptive state (subset)
          const dstate = localStorage.getItem(PKEY("dz_state"));
          if (dstate) {
            const ds = JSON.parse(dstate);
            if (ds && ds.rel && Array.isArray(ds.rel.predSum)) {
              dozenReliabilityRef.current.predSum = ds.rel.predSum;
              dozenReliabilityRef.current.actual = ds.rel.actual;
            }
            if (ds && ds.calib && Array.isArray(ds.calib.mult)) {
              dozenCalibAdjRef.current.mult = ds.calib.mult;
              if (typeof ds.calib.temp === "number")
                dozenCalibAdjRef.current.temp = ds.calib.temp;
            }
          }
        }
        // Remote pull (after local) – only if we still have little local data or remote may be longer
        if (
          (!history.length || !dozenHistory.length) &&
          remoteStateRef.current.status === "idle"
        ) {
          pullRemoteOnce();
        }
        // Load models (ignore errors)
        if (!modelRef.current) {
          try {
            const loaded = await tf.loadLayersModel(
              `localstorage://${PKEY("bin_model")}`
            );
            modelRef.current = loaded;
          } catch {}
        }
        if (!dozenModelRef.current) {
          try {
            const loadedDz = await tf.loadLayersModel(
              `localstorage://${PKEY("dz_model")}`
            );
            dozenModelRef.current = loadedDz;
          } catch {}
        }
      } catch (e) {
        if (console && console.warn) console.warn("Persist load failed", e);
      }
      if (history.length) {
        setLoading(true);
        await trainAndPredict();
        setLoading(false);
      }
    })();
  }, [history, trainAndPredict, ensureBackend, dozenHistory.length]);

  // Reliability warm-start (pseudo counts) – apply once on mount for dozens calibration if untouched
  useEffect(() => {
    if (
      !dozenHistory.length &&
      dozenReliabilityRef.current.predSum.every((v) => v === 0)
    ) {
      // Seed with mild pseudo evidence (Dirichlet-like) to avoid early extreme temp swings
      const seed = 2; // equivalent to 6 pseudo outcomes total
      const seedVal = seed / 3;
      dozenReliabilityRef.current.predSum = [seedVal, seedVal, seedVal];
      dozenReliabilityRef.current.actual = [seedVal, seedVal, seedVal];
      // Also give calibration multipliers a soft start near 1
      for (let i = 0; i < 3; i++) dozenCalibAdjRef.current.mult[i] = 1;
    }
  }, []);

  return {
    addObservation,
    addDozenObservation: (d: 0 | 1 | 2) => {
      // Snapshot last prediction (prediction should have been for this spin before outcome known)
      const prevPred = dozenPredictionRef.current;
      const snapshotLabel =
        prevPred && typeof prevPred.predLabel === "number"
          ? (prevPred.predLabel as 0 | 1 | 2)
          : null;
      const snapshotProbs: [number, number, number] | null = prevPred?.probs
        ? ([...prevPred.probs] as [number, number, number])
        : null;
      let correct: boolean | null =
        snapshotLabel != null ? snapshotLabel === d : null;
      // Defensive: if snapshot label doesn't match but somehow flagged true, fix.
      if (correct && snapshotLabel !== d) correct = false;
      if (correct != null) {
        dozenAccRef.current.total += 1;
        if (correct) dozenAccRef.current.correct += 1;
      }
      // Update error streak tracker
      const es = dozenPredErrorStreakRef.current;
      if (snapshotLabel != null) {
        if (es.last === snapshotLabel) {
          if (correct === false) es.incorrect += 1;
          else if (correct) es.incorrect = 0;
        } else {
          es.last = snapshotLabel;
          es.incorrect = correct === false ? 1 : 0;
        }
      }
      setDozenRecords((r) => [
        ...r,
        {
          label: d,
          predLabel: snapshotLabel,
          probs:
            !snapshotProbs || snapshotProbs.some((p) => !isFinite(p))
              ? [1 / 3, 1 / 3, 1 / 3]
              : snapshotProbs,
          correct,
        },
      ]);
      // === Calibration / Reliability / Brier tracking (legacy base) ===
      if (snapshotProbs) {
        // Brier score update (multi-class)
        const y = [0, 0, 0];
        y[d] = 1;
        let brier = 0;
        for (let i = 0; i < 3; i++) brier += (snapshotProbs[i] - y[i]) ** 2;
        dozenBrierWindowRef.current.push(brier);
        if (dozenBrierWindowRef.current.length > DZ_BRIER_WINDOW) {
          dozenBrierWindowRef.current.splice(
            0,
            dozenBrierWindowRef.current.length - DZ_BRIER_WINDOW
          );
        }
        // Reliability running sums
        const rel = dozenReliabilityRef.current;
        rel.predSum[0] += snapshotProbs[0];
        rel.predSum[1] += snapshotProbs[1];
        rel.predSum[2] += snapshotProbs[2];
        rel.actual[d] += 1;
        // Rolling buffer for windowed calibration
        dozenCalibBufferRef.current.push({ probs: snapshotProbs, label: d });
        if (dozenCalibBufferRef.current.length > DZ_CALIB_WINDOW) {
          dozenCalibBufferRef.current.splice(
            0,
            dozenCalibBufferRef.current.length - DZ_CALIB_WINDOW
          );
        }
        // Top probability window (for overconfidence gap)
        const topP = Math.max(...snapshotProbs);
        dozenTopProbWindowRef.current.push({ p: topP, correct: !!correct });
        if (dozenTopProbWindowRef.current.length > 90) {
          dozenTopProbWindowRef.current.splice(
            0,
            dozenTopProbWindowRef.current.length - 90
          );
        }
        // Track hi-confidence subset (>=0.45 threshold)
        if (topP >= 0.45) {
          dozenHiConfWindowRef.current.push({ p: topP, correct: !!correct });
          if (dozenHiConfWindowRef.current.length > 120) {
            dozenHiConfWindowRef.current.splice(
              0,
              dozenHiConfWindowRef.current.length - 120
            );
          }
        }
        // Perform calibration adjustments when enough samples
        const calibWin = dozenCalibBufferRef.current;
        if (calibWin.length >= DZ_CALIB_MIN) {
          // Windowed sums
          const sumP: [number, number, number] = [0, 0, 0];
          const sumA: [number, number, number] = [0, 0, 0];
          calibWin.forEach((rec) => {
            sumP[0] += rec.probs[0];
            sumP[1] += rec.probs[1];
            sumP[2] += rec.probs[2];
            sumA[rec.label] += 1;
          });
          // Update multiplicative reliability multipliers
          for (let i = 0; i < 3; i++) {
            if (sumP[i] > 0) {
              const ratio = sumA[i] / sumP[i];
              const old = dozenCalibAdjRef.current.mult[i];
              const updated = old * (1 - DZ_CALIB_LR) + DZ_CALIB_LR * ratio;
              dozenCalibAdjRef.current.mult[i] = Math.min(
                1.6,
                Math.max(0.6, updated)
              );
            }
          }
          // Temperature adaptation based on overconfidence gap (top prob vs accuracy)
          const gapWindow = dozenTopProbWindowRef.current;
          if (gapWindow.length >= 18) {
            const avgTop =
              gapWindow.reduce((a, r) => a + r.p, 0) / gapWindow.length;
            const accTop =
              gapWindow.filter((r) => r.correct).length / gapWindow.length;
            const gap = avgTop - accTop; // positive => overconfident
            let targetTemp = 1;
            if (gap > 0.05) {
              targetTemp = Math.min(1.5, 1 + (gap - 0.05) * 2.2);
            } else if (gap < -0.04) {
              // underconfident
              targetTemp = Math.max(0.75, 1 + gap * 1.5);
            }
            // Smooth temperature transitions
            dozenCalibAdjRef.current.temp =
              0.7 * dozenCalibAdjRef.current.temp + 0.3 * targetTemp;
          }
        }
        // Hi-prob event tracking + cooldown/shock triggers
        const topNow = Math.max(...snapshotProbs);
        if (topNow >= 0.55) {
          // Qualifying hi-prob event (slightly lower threshold than catastrophic cooldown)
          if (!correct) {
            dozenHiProbEventsRef.current.push(1);
          } else {
            dozenHiProbEventsRef.current.push(0);
          }
          if (dozenHiProbEventsRef.current.length > 120) {
            dozenHiProbEventsRef.current.splice(
              0,
              dozenHiProbEventsRef.current.length - 120
            );
          }
          // Update miss rate over last 30 qualifying events
          if (dozenHiProbEventsRef.current.length >= 8) {
            const slice = dozenHiProbEventsRef.current.slice(-30);
            const missRate = slice.reduce((a, b) => a + b, 0) / slice.length;
            dozenHiProbMissRateRef.current = missRate;
          }
          // Track last hi-prob miss flag
          if (!correct && topNow >= 0.6) {
            dozenLastHiProbMissRef.current = true;
          } else if (correct && topNow >= 0.6) {
            dozenLastHiProbMissRef.current = false;
          }
          // Cooldown on catastrophic high-prob miss >=0.6
          if (!correct && topNow >= 0.6) {
            const severity = Math.min(1, (topNow - 0.6) / 0.25); // widen range a bit to 0.85
            dozenHiProbCooldownRef.current = {
              spins: 4 + Math.round(severity * 5), // 4-9 spins
              factor: 1.2 + severity * 0.6, // 1.2 - 1.8
            };
          }
          // === Persistence (throttled) ===
          const now = Date.now();
          if (now - lastPersistRef.current > persistThrottleMs) {
            lastPersistRef.current = now;
            try {
              // histories & records
              localStorage.setItem(
                PKEY("dz_history"),
                JSON.stringify(dozenHistory.slice(-PERSIST_LIMIT))
              );
              localStorage.setItem(
                PKEY("dz_records"),
                JSON.stringify(dozenRecords.slice(-PERSIST_LIMIT))
              );
              localStorage.setItem(
                PKEY("bin_history"),
                JSON.stringify(history.slice(-PERSIST_LIMIT))
              );
              localStorage.setItem(
                PKEY("bin_records"),
                JSON.stringify(records.slice(-PERSIST_LIMIT))
              );
              // adaptive state snapshot
              localStorage.setItem(
                PKEY("dz_state"),
                JSON.stringify({
                  rel: dozenReliabilityRef.current,
                  calib: dozenCalibAdjRef.current,
                })
              );
              // save models (async, fire & forget)
              if (modelRef.current)
                modelRef.current
                  .save(`localstorage://${PKEY("bin_model")}`)
                  .catch(() => {});
              if (dozenModelRef.current)
                dozenModelRef.current
                  .save(`localstorage://${PKEY("dz_model")}`)
                  .catch(() => {});
              // Remote push (fire & forget)
              pushRemote();
            } catch (e) {
              if (console && console.warn)
                console.warn("Persist save failed", e);
            }
          }
          // Shock activation: three hi-prob misses (>=0.6) within last 12 qualifying events OR miss rate >0.65 over last 20
          const shock = dozenShockRef.current as any;
          const recentQual = dozenHiProbEventsRef.current.slice(-20);
          const severeSlice = dozenHiProbEventsRef.current
            .map((m, i) => ({ m, i }))
            .filter(({ i }) => i >= dozenHiProbEventsRef.current.length - 40) // consider last 40 for severity cluster
            .map((o) => o.m);
          const last12 = dozenHiProbEventsRef.current.slice(-12);
          const severeMisses = last12.reduce(
            (a, b) => a + (b === 1 ? 1 : 0),
            0
          );
          const missRate20 =
            recentQual.slice(-20).reduce((a, b) => a + b, 0) /
            (recentQual.slice(-20).length || 1);
          if (!shock.active) {
            if (severeMisses >= 3 || missRate20 > 0.65) {
              const sev = Math.min(
                1,
                severeMisses / 5 +
                  (missRate20 > 0.65 ? (missRate20 - 0.65) * 2 : 0)
              );
              dozenShockRef.current = {
                active: true,
                ttl: 8 + Math.round(sev * 8), // 8-16 spins
                hits: 0,
                consec: 0,
                severity: sev,
              };
              // Progressive cap decay multiplier downward nudge on activation
              dozenCapDecayRef.current.mult = Math.max(
                0.6,
                dozenCapDecayRef.current.mult * (0.92 - 0.1 * sev)
              );
            }
          } else if (shock.active) {
            // While in shock, decrement ttl and enforce entropy later; collect hits
            if (correct) (dozenShockRef.current as any).hits += 1;
            (dozenShockRef.current as any).ttl -= 1;
            if ((dozenShockRef.current as any).ttl <= 0) {
              // Recover: ease cap decay toward 1
              dozenCapDecayRef.current.mult = Math.min(
                1,
                dozenCapDecayRef.current.mult * 1.05
              );
              dozenShockRef.current = {
                active: false,
                ttl: 0,
                hits: 0,
                consec: 0,
                severity: 0,
              };
            }
          }
        }
        // === New: probability history & EWMA reliability ===
        for (let i = 0; i < 3; i++) {
          dozenProbHistoryRef.current[i].push(snapshotProbs[i]);
          if (dozenProbHistoryRef.current[i].length > 120)
            dozenProbHistoryRef.current[i].splice(
              0,
              dozenProbHistoryRef.current[i].length - 120
            );
        }
        const ema = dozenReliabilityEwmaRef.current.ema;
        const alphaE = 0.15;
        for (let i = 0; i < 3; i++) {
          const outcome = d === i ? 1 : 0;
          const err = outcome - snapshotProbs[i];
          // small step toward reducing bias: scaled error half strength
          ema[i] = ema[i] + alphaE * err * 0.5;
        }
        dozenReliabilityEwmaRef.current.total += 1;
        // Directional confusion window (actual vs predicted label)
        dozenConfusionWindowRef.current.push({ a: d, p: snapshotLabel });
        if (dozenConfusionWindowRef.current.length > 60)
          dozenConfusionWindowRef.current.splice(
            0,
            dozenConfusionWindowRef.current.length - 60
          );
        // Opportunity tracking (how often each class is top pick)
        if (snapshotLabel != null)
          dozenOpportunityRef.current.topPickCounts[snapshotLabel] += 1;
        dozenOpportunityRef.current.history.push(d);
        if (dozenOpportunityRef.current.history.length > 400)
          dozenOpportunityRef.current.history.shift();
        // False-negative streaks (pred not chosen but actual)
        for (let i = 0; i < 3; i++) {
          if (d === i && snapshotLabel !== i)
            dozenFalseNegStreakRef.current[i] += 1;
          else if (d === i && snapshotLabel === i)
            dozenFalseNegStreakRef.current[i] = 0;
        }
        // Gap hazard tracking: time since last appearance of each class
        const gh = dozenGapHazardRef.current;
        for (let i = 0; i < 3; i++) gh.gapSince[i] += 1;
        gh.gapSince[d] = 0;
        // appearance gap length for class d
        let gapLen = 0;
        for (
          let i = dozenOpportunityRef.current.history.length - 2;
          i >= 0;
          i--
        ) {
          if (dozenOpportunityRef.current.history[i] === d) break;
          gapLen++;
        }
        const appearKey = `${d}:${gapLen}`;
        gh.gapAppear[appearKey] = (gh.gapAppear[appearKey] || 0) + 1;
        for (let L = 0; L <= gapLen; L++) {
          const totKey = `${d}:>=${L}`;
          gh.gapTotals[totKey] = (gh.gapTotals[totKey] || 0) + 1;
        }
      }
      // Track per-class stats (prediction correctness by predicted class)
      if (snapshotLabel != null) {
        const cls = snapshotLabel;
        const rec = dozenClassStatsRef.current[cls];
        rec.total += 1;
        if (correct) rec.correct += 1;
      }
      // Update extended per-class performance structures & tunnel/pattern stats
      if (snapshotLabel != null && snapshotProbs) {
        const perf = dozenClassPerfRef.current[snapshotLabel];
        perf.attempts += 1;
        if (correct) perf.wins += 1;
        perf.sumTopProb += snapshotProbs[snapshotLabel];
        perf.recent.push({ win: !!correct, p: snapshotProbs[snapshotLabel] });
        if (perf.recent.length > 60) perf.recent.shift();
        if (!correct) perf.consecMisses += 1;
        else perf.consecMisses = 0;
        // Tunnel window
        dozenTunnelWindowRef.current.push({
          cls: snapshotLabel,
          win: !!correct,
        });
        if (dozenTunnelWindowRef.current.length > 15)
          dozenTunnelWindowRef.current.splice(
            0,
            dozenTunnelWindowRef.current.length - 15
          );
        // Pattern perf (now that outcome known) using last distribution hash stored earlier at prediction time via shapeHash of snapshotProbs
        const ph = dozenShapeHash(snapshotProbs);
        if (!dozenPatternPerfRef.current[ph])
          dozenPatternPerfRef.current[ph] = { tries: 0, wins: 0 };
        dozenPatternPerfRef.current[ph].tries += 1;
        if (correct) dozenPatternPerfRef.current[ph].wins += 1;
      }
      // Update miss streak & recent results for decision temperature
      if (snapshotLabel != null) {
        if (correct) dozenPerClassMissRef.current[snapshotLabel] = 0;
        else dozenPerClassMissRef.current[snapshotLabel] += 1;
        if (snapshotProbs) {
          const pChosen = snapshotProbs[snapshotLabel];
          dozenRecentResultsRef.current.push({
            correct: !!correct,
            prob: pChosen,
          });
          // Regret tracking & high-confidence wrong accumulation
          if (snapshotLabel != null) {
            if (correct) {
              dozenRegretRef.current[snapshotLabel] *= DZ_REGRET_DECAY;
            } else {
              dozenRegretRef.current[snapshotLabel] =
                dozenRegretRef.current[snapshotLabel] * DZ_REGRET_DECAY +
                (1 - pChosen);
            }
            if (pChosen >= DZ_HICONF_PROB) {
              dozenHiConfWrongRef.current.push(correct ? 0 : 1);
              if (dozenHiConfWrongRef.current.length > DZ_HICONF_WINDOW) {
                dozenHiConfWrongRef.current.splice(
                  0,
                  dozenHiConfWrongRef.current.length - DZ_HICONF_WINDOW
                );
              }
            }
          }
          if (dozenRecentResultsRef.current.length > 250) {
            dozenRecentResultsRef.current.splice(
              0,
              dozenRecentResultsRef.current.length - 250
            );
          }
        }
      }
      // Track recent prediction list for dynamic gating window
      if (snapshotLabel != null && correct != null) {
        dozenRecentPredsRef.current.push({
          label: snapshotLabel,
          correct: !!correct,
        });
        if (dozenRecentPredsRef.current.length > 100) {
          dozenRecentPredsRef.current.splice(
            0,
            dozenRecentPredsRef.current.length - 100
          );
        }
      }
      setDozenHistory((h) => {
        if (h.length) {
          const last = h[h.length - 1];
          dozenMarkovRef.current[last][d] += 1;
          let rl = 1;
          for (let i = h.length - 2; i >= 0; i--) {
            if (h[i] === last) rl++;
            else break;
          }
          const key = `${last}:${rl}`;
          const rec = dozenRunPostRef.current[key] || { cont: 0, rev: 0 };
          if (d === last) rec.cont += 1;
          else rec.rev += 1;
          dozenRunPostRef.current[key] = rec;
        }
        return [...h, d];
      });
      // Maintain recent actual outcomes window for majority fallback
      dozenRecentWindowRef.current.push(d);
      if (dozenRecentWindowRef.current.length > DZ_MAJ_WINDOW) {
        dozenRecentWindowRef.current.splice(
          0,
          dozenRecentWindowRef.current.length - DZ_MAJ_WINDOW
        );
      }
      // RL weight update (multi-class) centered at 1/3 baseline using probability assigned to true label
      if (controlsRef.current.rlEnabled && dozenLastSourcesRef.current) {
        const srcDists = dozenLastSourcesRef.current;
        // Confidence-aware eta modulation
        let eta = controlsRef.current.rlEta * 1.05;
        if (snapshotLabel != null && snapshotProbs) {
          const pChosen = snapshotProbs[snapshotLabel];
          if (!correct && pChosen < 0.45) eta *= 1.6; // stronger penalty
          else if (correct && pChosen > 0.5)
            eta *= 1.25; // reward only strong corrects
          else eta *= 0.6; // damp neutral outcomes
        }
        (Object.keys(srcDists) as (keyof typeof srcDists)[]).forEach((k) => {
          const dist = srcDists[k];
          const pTrue = dist ? dist[d] : 1 / 3;
          // Hedge against zero by floor and cap
          const edge = pTrue - 1 / 3;
          const reward = correct ? 1 : -1;
          const mult = Math.exp(eta * reward * edge);
          rlWeightsRef.current[k as keyof typeof rlWeightsRef.current] =
            Math.min(
              5,
              Math.max(
                0.2,
                rlWeightsRef.current[k as keyof typeof rlWeightsRef.current] *
                  mult
              )
            );
          // stats
          const stat = dozenSourceStatsRef.current[k as string] || {
            correct: 0,
            total: 0,
          };
          stat.total += 1;
          if (correct) stat.correct += 1;
          dozenSourceStatsRef.current[k as string] = stat;
        });
        // Update model-only performance (using model distribution probability of predicted label vs actual label)
        if (srcDists.model) {
          dozenModelPerfRef.current.total += 1;
          if (correct) dozenModelPerfRef.current.correct += 1;
        }
        // normalize
        const sum =
          Object.values(rlWeightsRef.current).reduce((a, b) => a + b, 0) || 1;
        (
          Object.keys(
            rlWeightsRef.current
          ) as (keyof typeof rlWeightsRef.current)[]
        ).forEach((k) => {
          rlWeightsRef.current[k] /= sum / 7;
        });
      }
      // Immediately compute next prediction distribution after update (async fire and forget)
      setTimeout(() => {
        if (
          controlsRef.current.mode === "roulette" &&
          controlsRef.current.rouletteVariant === "dozens"
        ) {
          trainAndPredictDozens();
        }
      }, 0);
    },
    history,
    records,
    probOver,
    loading,
    backend,
    ready: history.length > windowSize,
    windowSize,
    accuracy: dozenAccRef.current.total
      ? dozenAccRef.current.correct / dozenAccRef.current.total
      : null,
    controls: controlsRef.current,
    controlsVersion,
    setControls,
    resetModel,
    rlWeights: rlWeightsRef.current,
    biasInfo,
    metrics,
    suggestedThreshold: controlsRef.current.suggestedThreshold,
    probParts: probPartsRef.current,
    regimeDiagnostics: {
      biasGap: biasRegimeRef.current.biasGap,
      regimeShift: biasRegimeRef.current.regimeShift,
      patternDampen: biasRegimeRef.current.patternDampen,
      avgRunLenEma: biasRegimeRef.current.avgRunLenEma,
      recentAvgRunLen: biasRegimeRef.current.lastAvgRunLen,
    },
    // Dozens mode data
    rouletteDozenHistory: dozenHistory,
    rouletteDozenProbs: dozenProbsRef.current,
    rouletteDozenPred: dozenPredictionRef.current.predLabel,
    rouletteDozenVersion: dozenVersion,
    rouletteDozenRecords: dozenRecords,
    rouletteDozenAccuracy: dozenAccRef.current.total
      ? dozenAccRef.current.correct / dozenAccRef.current.total
      : null,
    rouletteDozenDiagnostics: {
      classStats: {
        0: {
          acc: dozenClassStatsRef.current[0].total
            ? dozenClassStatsRef.current[0].correct /
              dozenClassStatsRef.current[0].total
            : null,
          total: dozenClassStatsRef.current[0].total,
        },
        1: {
          acc: dozenClassStatsRef.current[1].total
            ? dozenClassStatsRef.current[1].correct /
              dozenClassStatsRef.current[1].total
            : null,
          total: dozenClassStatsRef.current[1].total,
        },
        2: {
          acc: dozenClassStatsRef.current[2].total
            ? dozenClassStatsRef.current[2].correct /
              dozenClassStatsRef.current[2].total
            : null,
          total: dozenClassStatsRef.current[2].total,
        },
      },
      modelPerf: {
        acc: dozenModelPerfRef.current.total
          ? dozenModelPerfRef.current.correct / dozenModelPerfRef.current.total
          : null,
        total: dozenModelPerfRef.current.total,
      },
      errorStreak: {
        last: dozenPredErrorStreakRef.current.last,
        incorrect: dozenPredErrorStreakRef.current.incorrect,
      },
      predFreq: dozenPredFreqRef.current,
      regime: dozenRegimeRef.current,
      decision: dozenDecisionInfoRef.current,
      guard: dozenGuardRef.current,
    },
    rouletteDozenAdaptive: dozenAdaptiveRef.current,
    rouletteDozenExtended: {
      hiProbMissRate: dozenHiProbMissRateRef.current,
      lastHiProbMiss: dozenLastHiProbMissRef.current,
      capDecayMult: dozenCapDecayRef.current.mult,
      shock: dozenShockRef.current,
      alternation: dozenAlternationRef.current,
      abstain: dozenAbstainRef.current,
    },
    remoteState: remoteStateRef.current,
  };
}
