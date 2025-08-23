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
  const [probOver, setProbOver] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [backend, setBackend] = useState<string | null>(null);
  // Roulette dozens mode state (separate simple frequency tracker)
  const [dozenHistory, setDozenHistory] = useState<(0 | 1 | 2)[]>([]);
  const dozenProbsRef = useRef<[number, number, number] | null>(null);
  // Multi-class (dozens) model & tracking
  const dozenModelRef = useRef<tf.LayersModel | null>(null);
  const dozenPredictionRef = useRef<{
    probs: [number, number, number] | null;
    predLabel: 0 | 1 | 2 | null;
  }>({ probs: null, predLabel: null });
  // Heuristic tracking for dozens
  const dozenMarkovRef = useRef([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
  ]);
  const dozenRunPostRef = useRef<Record<string, { cont: number; rev: number }>>(
    {}
  );
  const dozenEwmaRef = useRef<[number, number, number] | null>(null);
  const dozenAccRef = useRef({ correct: 0, total: 0 });
  interface DozenRecord {
    label: 0 | 1 | 2;
    predLabel: 0 | 1 | 2 | null;
    probs: [number, number, number] | null;
    correct: boolean | null;
  }
  const [dozenRecords, setDozenRecords] = useState<DozenRecord[]>([]);
  // Version bump to force UI re-render when dozens prediction/probs update
  const [dozenVersion, setDozenVersion] = useState(0);
  // Track last source distributions for dozens to perform accurate RL updates
  const dozenLastSourcesRef = useRef<Record<
    string,
    [number, number, number]
  > | null>(null);
  // Per-source performance stats for dozens
  const dozenSourceStatsRef = useRef<
    Record<string, { correct: number; total: number }>
  >({});
  const modelRef = useRef<tf.LayersModel | null>(null);
  const markovCountsRef = useRef([
    [1, 1],
    [1, 1],
  ]);
  const ewmaRef = useRef<number | null>(null);
  const streakPosteriorsRef = useRef<
    Record<string, { cont: number; rev: number }>
  >({});
  const accRef = useRef({ correct: 0, total: 0 });
  const probPartsRef = useRef<ProbParts | null>(null);
  const lastPredictionRef = useRef<{
    rawProb: number | null;
    effectiveProb: number | null;
    invertUsed: boolean;
    threshold: number;
  }>({ rawProb: null, effectiveProb: null, invertUsed: false, threshold: 0.5 });
  const rlWeightsRef = useRef({
    model: 1,
    markov: 1,
    streak: 1,
    pattern: 1,
    ewma: 1,
    bayes: 1,
    entropyMod: 1,
  });
  const [biasInfo, setBiasInfo] = useState<{
    predRate: number | null;
    actualRate: number | null;
    window: number;
    status: "neutral" | "over-bias" | "under-bias";
    override: boolean;
  }>({
    predRate: null,
    actualRate: null,
    window: 0,
    status: "neutral",
    override: false,
  });
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
  }, [featureCount]);

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
    m.add(tf.layers.dense({ units: 24, activation: "relu" }));
    m.add(tf.layers.dense({ units: 3, activation: "softmax" }));
    m.compile({
      optimizer: tf.train.adam(0.001),
      loss: "categoricalCrossentropy",
    });
    return m;
  };
  const ensureDozenModel = useCallback(async () => {
    if (!dozenModelRef.current) dozenModelRef.current = buildDozenModel();
    return dozenModelRef.current;
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
    mode: "binary" as "binary" | "roulette", // UI mode toggle
    rouletteVariant: "redblack" as "redblack" | "dozens",
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
        effectiveProb: null,
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
  const DOZEN_TRAIN_START = 8; // give a little more context before first CNN training
  const DOZEN_TRAIN_EVERY = 2; // train every 2 spins for stability
  // track prediction streak to damp persistent bias
  const dozenPredStreakRef = useRef<{ last: number | null; len: number }>({
    last: null,
    len: 0,
  });
  const trainAndPredictDozens = useCallback(async () => {
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
        const xs = tf.tensor3d(xsArr);
        const ys = tf.tensor2d(ysArr);
        await m.fit(xs, ys, {
          epochs: 1,
          verbose: 0,
          batchSize: Math.min(16, xsArr.length),
        });
        xs.dispose();
        ys.dispose();
        dozenLastTrainRef.current = dozenHistory.length;
      }
    }
    const counts = [0, 0, 0];
    dozenHistory.forEach((v) => (counts[v] += 1));
    const dirichlet = counts.map(
      (c) => (c + 1) / (dozenHistory.length + 3)
    ) as [number, number, number];
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
    dozenEwmaRef.current = [0, 1, 2].map(
      (i) => (1 - alpha) * (dozenEwmaRef.current as any)[i] + alpha * inst[i]
    ) as [number, number, number];
    const ewmaDist = dozenEwmaRef.current;
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
    };
    dozenLastSourcesRef.current = sources; // snapshot for RL update
    // Weighting (manual * RL if enabled)
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
    const manual = controlsRef.current;
    // Ramp model weight in gradually to avoid early overfitting dominance
    const n = dozenHistory.length;
    const modelRamp =
      n <= DOZEN_TRAIN_START ? 0 : Math.min(1, (n - DOZEN_TRAIN_START) / 30); // linear ramp over 30 spins
    const weightMap: Record<string, number> = {
      model: manual.wModel * modelRamp * dyn.model,
      markov: manual.wMarkov * dyn.markov,
      streak: manual.wStreak * dyn.streak,
      pattern: manual.wPattern * dyn.pattern,
      ewma: manual.wEwma * dyn.ewma,
      bayes: manual.wBayes * dyn.bayes,
      entropyMod: manual.wEntropy * dyn.entropyMod,
    };
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
    // --- Debiasing & confidence shaping ---
    // 1. Inverse-frequency debias (mild) to prevent early lock-in
    const totalCounts = counts[0] + counts[1] + counts[2];
    if (totalCounts > 0) {
      const gamma = 0.4; // debias strength
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
    const minAllowed = 0.85 * maxEnt; // encourage diversity
    if (entropy < minAllowed && dozenHistory.length < 60) {
      // blend toward uniform proportionally to deficit and data scarcity
      const deficit = (minAllowed - entropy) / minAllowed; // 0..1
      const scarcity = 1 - Math.min(dozenHistory.length / 60, 1); // 1 early -> 0 later
      const blend = Math.min(0.6, deficit * 0.7 * scarcity);
      finalDist = [0, 1, 2].map(
        (i) => (1 - blend) * finalDist[i] + blend * (1 / 3)
      ) as [number, number, number];
    }
    // 3. Prediction streak dampening: if same prediction repeats with mediocre realized performance, soften
    const ps = dozenPredStreakRef.current;
    const topIdx =
      finalDist[0] >= finalDist[1] && finalDist[0] >= finalDist[2]
        ? 0
        : finalDist[1] >= finalDist[2]
        ? 1
        : 2;
    if (ps.last === topIdx) {
      ps.len += 1;
    } else {
      ps.last = topIdx;
      ps.len = 1;
    }
    if (ps.len >= 5 && n < 120) {
      // allow more persistence later when enough data
      // damp top prob a bit and redistribute to others
      const damp = Math.min(0.15 + (ps.len - 4) * 0.03, 0.35);
      const removed = finalDist[topIdx] * damp;
      const remain = finalDist[topIdx] - removed;
      const addEach = removed / 2;
      finalDist = [0, 1, 2].map((i) =>
        i === topIdx ? remain : finalDist[i] + addEach
      ) as [number, number, number];
    }
    // 4. Final renormalize (safety)
    const fs = finalDist[0] + finalDist[1] + finalDist[2];
    finalDist = [finalDist[0] / fs, finalDist[1] / fs, finalDist[2] / fs];
    const pred = topIdx;
    dozenProbsRef.current = finalDist;
    dozenPredictionRef.current = { probs: finalDist, predLabel: pred };
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
            if (predRate > actualRate + cfg.threshold) status = "over-bias";
            else if (predRate < actualRate - cfg.threshold)
              status = "under-bias";
          }
          if (status === "neutral") biasStableNeutralRef.current += 1;
          else biasStableNeutralRef.current = 0;
          if (
            (status === "over-bias" || status === "under-bias") &&
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
            predRate,
            actualRate,
            window: w,
            status,
            override: biasOverrideRef.current,
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
      await tf.ready();
      if (!backend) {
        const hasNavigatorGPU =
          typeof navigator !== "undefined" && (navigator as any).gpu;
        let selected = "webgl"; // default fallback
        if (
          hasNavigatorGPU &&
          (tf as any).engine()?.registryFactory?.["webgpu"]
        ) {
          try {
            // Additional feature probe: request adapter & ensure requestAdapterInfo exists
            const adapter = await (navigator as any).gpu.requestAdapter?.();
            if (
              adapter &&
              typeof (adapter as any).requestAdapterInfo === "function"
            ) {
              await tf.setBackend("webgpu");
              selected = tf.getBackend();
            } else {
              // Skip attempting webgpu backend if required method missing
              await tf.setBackend("webgl");
              selected = tf.getBackend();
            }
          } catch (err) {
            // Silent fallback; avoid spamming console with internal backend errors
            await tf.setBackend("webgl");
            selected = tf.getBackend();
          }
        } else {
          await tf.setBackend("webgl");
          selected = tf.getBackend();
        }
        setBackend(selected);
      }
      if (history.length) {
        setLoading(true);
        await trainAndPredict();
        setLoading(false);
      }
    })();
  }, [history, trainAndPredict, backend]);

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
      // RL weight update (multi-class) centered at 1/3 baseline using probability assigned to true label
      if (controlsRef.current.rlEnabled && dozenLastSourcesRef.current) {
        const srcDists = dozenLastSourcesRef.current;
        const eta = controlsRef.current.rlEta * 0.9; // slightly gentler for multi-class
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
    accuracy: accRef.current.total
      ? accRef.current.correct / accRef.current.total
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
  };
}
