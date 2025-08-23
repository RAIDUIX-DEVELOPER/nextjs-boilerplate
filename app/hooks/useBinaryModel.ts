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
  const featureCount = 8;
  const epochs = 2;
  const batchSize = 32;
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
    const m = tf.sequential();
    m.add(tf.layers.inputLayer({ inputShape: [windowSize, featureCount] }));
    m.add(
      tf.layers.conv1d({
        filters: 32,
        kernelSize: 3,
        activation: "relu",
        padding: "same",
      })
    );
    m.add(
      tf.layers.conv1d({
        filters: 32,
        kernelSize: 5,
        activation: "relu",
        padding: "same",
      })
    );
    m.add(tf.layers.dropout({ rate: 0.1 }));
    m.add(
      tf.layers.conv1d({
        filters: 24,
        kernelSize: 3,
        activation: "relu",
        padding: "same",
      })
    );
    m.add(tf.layers.globalAveragePooling1d());
    m.add(tf.layers.dense({ units: 32, activation: "relu" }));
    m.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
    m.compile({ optimizer: tf.train.adam(0.001), loss: "binaryCrossentropy" });
    return m;
  };
  const ensureModel = useCallback(async () => {
    if (modelRef.current) return modelRef.current;
    modelRef.current = buildModel();
    return modelRef.current;
  }, []);

  // Stable memoized feature builder (prevents effect dependency churn -> retrain loop)
  const buildFeatureWindow = useCallback((seq: number[]): number[][] => {
    const feats: number[][] = [];
    let streak = 0;
    let lastVal = seq[0];
    let timeSinceOver = 0;
    let timeSinceUnder = 0;
    for (let i = 0; i < seq.length; i++) {
      const v = seq[i];
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
      feats.push([
        v,
        Math.min(streak / 20, 1),
        i > 0 && seq[i - 1] !== v ? 1 : 0,
        roll(slice(5)),
        roll(slice(10)),
        roll(slice(20)),
        Math.min(timeSinceOver / 20, 1),
        Math.min(timeSinceUnder / 20, 1),
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
  });

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
  const recomputeCurrentProb = useCallback(async () => {
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
    const lastWinRaw = history.slice(-windowSize);
    const feats = buildFeatureWindow(lastWinRaw);
    const lastTensor = tf.tensor3d([feats]);
    const modelProbT = m.predict(lastTensor) as tf.Tensor;
    const modelProb = (await modelProbT.data())[0];
    lastTensor.dispose();
    modelProbT.dispose();
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
        ].includes(k)
      )
    ) {
      recomputeCurrentProb();
    }
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
    const xs = tf.tensor3d(xsArr);
    const ys = tf.tensor2d(ysArr.map((v) => [v]));
    await m.fit(xs, ys, {
      epochs,
      batchSize: Math.min(batchSize, xsArr.length),
      shuffle: true,
      verbose: 0,
    });
    xs.dispose();
    ys.dispose();
    lastTrainCountRef.current = history.length;
    await recomputeCurrentProb();
  }, [
    history,
    windowSize,
    buildFeatureWindow,
    ensureModel,
    recomputeCurrentProb,
  ]);

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
          };
        });
      }
      if (
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
      } else if (controlsRef.current.rlEnabled && predLabel === 0) {
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
  };
}
