"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgpu";
import "@tensorflow/tfjs-backend-webgl";
import localforage from "localforage";

interface Stats {
  count: number;
  mean: number;
  M2: number;
  std: number;
}

localforage.config({ name: "realtime-forecast" });

const MODEL_KEY = "indexeddb://realtime-forecast-v2"; // bumped due to feature schema change
const HISTORY_KEY = "history-v1";

export function useRealtimeModel() {
  // Reduced window size to start training earlier per requirements.
  const windowSize = 8;
  const maxHistory = 1024;
  const epochsPerUpdate = 2;
  const batchSize = 32;

  const [history, setHistory] = useState<number[]>([]);
  // Classification only: probability next value > threshold
  const [probAbove2, setProbAbove2] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [backend, setBackend] = useState<string | null>(null);
  const statsRef = useRef<Stats>({ count: 0, mean: 0, M2: 0, std: 0 });
  // Regression model removed; only classifier remains.
  const modelRef = useRef<tf.LayersModel | null>(null);
  const updatesRef = useRef(0);
  // Remove AR(1) (not needed for binary classification of >2) but keep placeholder for potential feature reuse.
  const ar1Ref = useRef({ sumXX: 0, sumXY: 0, count: 0, phi: 0 });
  const saveTimeout = useRef<number | undefined>(undefined);
  // Separate classification model (probability next value > threshold)
  const clsModelRef = useRef<tf.LayersModel | null>(null);
  // Online probabilistic calibrator (logistic regression over meta-features)
  const calibWeightsRef = useRef<number[] | null>(null); // length set when first used
  const lastCalibFeatRef = useRef<number[] | null>(null);
  const lastClsBaseProbRef = useRef<number | null>(null);

  // Remove ensemble / bandit / markov for simplified classification-only mode.
  const weightsRef = useRef<[number, number, number] | null>(null);
  const lastPredRef = useRef<any>(null);
  const rewardRef = useRef(0); // could repurpose for accuracy metric
  const markovCountsRef = useRef<any>(null);
  const lastBucketRef = useRef<number | null>(null);
  const eta = 0; // unused
  const thresholdNextHigh = 2.0; // classification decision threshold
  const CLS_MODEL_KEY = "indexeddb://realtime-forecast-cls-v1";

  // Streak probability tracking (continuation vs reversal)
  const maxStreakTrack = 50;
  const lowContCountsRef = useRef<number[]>(Array(maxStreakTrack).fill(0));
  const lowRevCountsRef = useRef<number[]>(Array(maxStreakTrack).fill(0));
  const highContCountsRef = useRef<number[]>(Array(maxStreakTrack).fill(0));
  const highRevCountsRef = useRef<number[]>(Array(maxStreakTrack).fill(0));
  const streakStateRef = useRef<{
    type: "low" | "high" | "mid" | null;
    length: number;
  }>({ type: null, length: 0 });

  const bucketForZ = (z: number) => {
    // Round to nearest int and shift range [-5,5] -> [0,10]
    const clamped = Math.max(-5, Math.min(5, Math.round(z)));
    return clamped + 5;
  };
  const bucketCenter = (idx: number) => idx - 5; // inverse mapping

  const updateStats = (x: number) => {
    const s = statsRef.current;
    s.count += 1;
    const delta = x - s.mean;
    s.mean += delta / s.count;
    const delta2 = x - s.mean;
    s.M2 += delta * delta2;
    s.std = s.count > 1 ? Math.sqrt(s.M2 / (s.count - 1)) : 0;
  };

  const scale = (x: number) => {
    const s = statsRef.current;
    if (s.std === 0) return 0;
    return (x - s.mean) / (s.std + 1e-8);
  };
  const descale = (x: number) => {
    const s = statsRef.current;
    return x * (s.std + 1e-8) + s.mean;
  };

  // Features: valueScaled, deltaScaled, pctChange, localVol, isLow, isHigh, streakLowNorm, streakHighNorm
  // Added below2 streak feature (value <= 2) -> featureCount 9
  const featureCount = 9;
  const zThreshold = 0.5; // z-score threshold for low/high categorization

  // Remove regression model builder

  const buildClsModel = () => {
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

  const ensureClsModel = useCallback(async () => {
    if (clsModelRef.current) return clsModelRef.current;
    try {
      const loaded = await tf.loadLayersModel(CLS_MODEL_KEY);
      const inShape = loaded.inputs[0].shape;
      if (inShape && inShape[2] !== featureCount) {
        loaded.dispose();
        clsModelRef.current = buildClsModel();
      } else {
        if (!(loaded as any).optimizer) {
          loaded.compile({
            optimizer: tf.train.adam(0.001),
            loss: "binaryCrossentropy",
          });
        }
        clsModelRef.current = loaded;
      }
    } catch {
      clsModelRef.current = buildClsModel();
    }
    return clsModelRef.current!;
  }, [featureCount]);

  const ensureModel = useCallback(async () => null, []); // deprecated

  const persistHistory = (h: number[]) => {
    if (saveTimeout.current) window.clearTimeout(saveTimeout.current);
    saveTimeout.current = window.setTimeout(() => {
      localforage.setItem(HISTORY_KEY, h);
    }, 400);
  };

  const buildFeatureWindow = (seq: number[]): number[][] => {
    const s = statsRef.current;
    const globalStd = s.std + 1e-8;
    const out: number[][] = [];
    let streakLow = 0;
    let streakHigh = 0;
    let streakBelow2 = 0;
    for (let i = 0; i < seq.length; i++) {
      const v = seq[i];
      const prev = i > 0 ? seq[i - 1] : v;
      const delta = (v - prev) / globalStd; // scaled delta
      const pct = i > 0 ? (v - prev) / (Math.abs(prev) + 1e-6) : 0;
      // local volatility (rolling std of last k values, k=5) scaled
      const k = 5;
      const start = Math.max(0, i - k + 1);
      const slice = seq.slice(start, i + 1);
      const meanSlice = slice.reduce((a, b) => a + b, 0) / slice.length;
      const varSlice =
        slice.reduce((a, b) => a + (b - meanSlice) ** 2, 0) / slice.length;
      const vol = Math.sqrt(varSlice) / globalStd;
      const z = (v - s.mean) / (s.std + 1e-8);
      const isLow = z < -zThreshold ? 1 : 0;
      const isHigh = z > zThreshold ? 1 : 0;
      if (v <= 2) {
        streakBelow2 += 1;
      } else {
        streakBelow2 = 0;
      }
      if (isLow) {
        streakLow += 1;
        streakHigh = 0;
      } else if (isHigh) {
        streakHigh += 1;
        streakLow = 0;
      } else {
        streakLow = 0;
        streakHigh = 0;
      }
      const streakLowNorm = Math.min(streakLow / 10, 1); // cap normalization
      const streakHighNorm = Math.min(streakHigh / 10, 1);
      const below2Norm = Math.min(streakBelow2 / 10, 1);
      const scaledVal = scale(v);
      out.push([
        scaledVal,
        delta,
        pct,
        vol,
        isLow,
        isHigh,
        streakLowNorm,
        streakHighNorm,
        below2Norm,
      ]);
    }
    return out;
  };

  const ar1Forecast = (_last: number) => 0; // unused

  const trainAndPredict = useCallback(async () => {
    const clsM = await ensureClsModel();
    if (history.length <= windowSize) {
      setProbAbove2(null);
      return;
    }
    const start = 0;
    const windows: number[][][] = [];
    const targets: number[] = [];
    const clsTargets: number[] = [];
    for (let i = start; i < history.length - windowSize; i++) {
      const slice = history.slice(i, i + windowSize);
      const y = history[i + windowSize];
      const feat = buildFeatureWindow(slice);
      windows.push(feat);
      // Predict standardized delta relative to last value in window, avoiding convergence to global mean.
      const lastInWindow = slice[slice.length - 1];
      const stdAdj = statsRef.current.std + 1e-8;
      const delta = (y - lastInWindow) / stdAdj;
      targets.push(delta);
      clsTargets.push(y > thresholdNextHigh ? 1 : 0);
    }
    if (windows.length === 0) return;
    const xs = tf.tensor3d(windows);
    const ysCls = tf.tensor2d(clsTargets.map((v) => [v]));
    await clsM.fit(xs, ysCls, {
      epochs: 1,
      batchSize: Math.min(batchSize, windows.length),
      shuffle: true,
      verbose: 0,
    });
    xs.dispose();
    ysCls.dispose();
    const lastWindowRaw = history.slice(-windowSize);
    const lastFeatures = buildFeatureWindow(lastWindowRaw);
    const lastWindow = tf.tensor3d([lastFeatures]);
    const s = statsRef.current;
    const lastVal = history[history.length - 1];
    const zLast = (lastVal - s.mean) / (s.std + 1e-8);
    const currentBucket = bucketForZ(zLast);
    // Independent classification probability (does not affect regression ensemble)
    try {
      const clsPred = clsM.predict(lastWindow) as tf.Tensor;
      const clsData = await clsPred.data();
      let p = clsData[0];
      if (!isFinite(p)) p = 0.5;
      if (p < 0) p = 0;
      if (p > 1) p = 1;
      // Adjust probability upward when long streak of <=2 occurred (anticipate break)
      let below2Streak = 0;
      for (let i = history.length - 1; i >= 0; i--) {
        if (history[i] <= thresholdNextHigh) below2Streak++;
        else break;
        if (below2Streak > 50) break;
      }
      const below2Norm = Math.min(below2Streak / 10, 1); // 0..1
      const alpha = 0.8; // strength of streak influence
      const logit = Math.log((p + 1e-6) / (1 - p + 1e-6));
      const adjustedLogit = logit + alpha * below2Norm;
      p = 1 / (1 + Math.exp(-adjustedLogit));
      // Meta-feature calibrator (online logistic): Build feature vector
      // Features: [bias, baseP, below2Norm, ar1Phi, markovDeltaSign, predictedDeltaSign, varianceNorm]
      const ar1PhiVal = ar1Ref.current.phi || 0;
      const feat: number[] = [1, p, below2Norm, ar1PhiVal, 0, 0, 1];
      if (
        !calibWeightsRef.current ||
        calibWeightsRef.current.length !== feat.length
      ) {
        calibWeightsRef.current = Array(feat.length).fill(0);
      }
      // Compute calibrated probability
      const w = calibWeightsRef.current;
      let z = 0;
      for (let i = 0; i < feat.length; i++) z += w[i] * feat[i];
      const pCalib = 1 / (1 + Math.exp(-z));
      const blended = 0.5 * p + 0.5 * pCalib; // blend base and calibrated for stability
      setProbAbove2(blended);
      lastCalibFeatRef.current = feat;
      lastClsBaseProbRef.current = p;
      setProbAbove2(p);
      clsPred.dispose();
      // Add probability to lastPredRef for reward scoring
      // (We only store for scoring, not used in ensemble weights)
      lastPredRef.current = { clsProb: blended };
    } catch {
      setProbAbove2(null);
    }
    // Store last classification prob only (for reward/online calibration)
    // Current bucket becomes 'from' state for next transition
    lastBucketRef.current = currentBucket;
    lastWindow.dispose();
    updatesRef.current += 1;
    if (updatesRef.current % 10 === 0) {
      try {
        await clsM.save(CLS_MODEL_KEY);
      } catch {}
    }
  }, [history, ensureModel]);

  const addValue = useCallback(async (value: number) => {
    if (!isFinite(value)) return;
    if (value < 1) value = 1; // enforce minimum of 1
    // --- Update streak continuation / reversal statistics BEFORE stats update ---
    const sPrev = statsRef.current;
    const prevMean = sPrev.mean;
    const prevStd = sPrev.std;
    const zPrevVal = history.length
      ? (history[history.length - 1] - prevMean) / (prevStd + 1e-8)
      : 0;
    const classify = (z: number) =>
      z < -zThreshold ? "low" : z > zThreshold ? "high" : "mid";
    const prevClass = classify(zPrevVal);
    const streakState = streakStateRef.current;
    // Determine if continuation or reversal relative to previous streak
    if (
      streakState.type === prevClass &&
      (prevClass === "low" || prevClass === "high")
    ) {
      // continuing streak (increment length but record continuation count for previous length)
      const idx = Math.min(maxStreakTrack - 1, streakState.length);
      if (prevClass === "low") lowContCountsRef.current[idx] += 1;
      else highContCountsRef.current[idx] += 1;
      streakState.length += 1;
    } else {
      // reversal event for previous streak type
      if (streakState.type === "low" && streakState.length > 0) {
        const idx = Math.min(maxStreakTrack - 1, streakState.length);
        lowRevCountsRef.current[idx] += 1;
      } else if (streakState.type === "high" && streakState.length > 0) {
        const idx = Math.min(maxStreakTrack - 1, streakState.length);
        highRevCountsRef.current[idx] += 1;
      }
      // start new streak if prevClass is low/high else reset
      if (prevClass === "low" || prevClass === "high") {
        streakState.type = prevClass;
        streakState.length = 1;
      } else {
        streakState.type = "mid";
        streakState.length = 0;
      }
    }
    // --- Evaluate previous prediction (reward + weights update) ---
    if (lastPredRef.current) {
      const clsProb = lastPredRef.current.clsProb as number | undefined;
      if (clsProb !== undefined) {
        const y = value > thresholdNextHigh ? 1 : 0;
        const correct = (clsProb >= 0.5 ? 1 : 0) === y;
        rewardRef.current += correct ? 1 : 0; // simple accuracy count
        if (lastCalibFeatRef.current && calibWeightsRef.current) {
          const x = lastCalibFeatRef.current;
          let z = 0;
          for (let i = 0; i < calibWeightsRef.current.length; i++)
            z += calibWeightsRef.current[i] * x[i];
          const pCalib = 1 / (1 + Math.exp(-z));
          const gradFactor = pCalib - y;
          const lr = 0.05;
          for (let i = 0; i < calibWeightsRef.current.length; i++)
            calibWeightsRef.current[i] -= lr * gradFactor * x[i];
        }
      }
      lastPredRef.current = null;
    }
    // Update AR(1) accumulators using previous value if exists
    setHistory((prev) => {
      if (prev.length > 0) {
        const prevVal = prev[prev.length - 1];
        const a = ar1Ref.current;
        a.sumXX += prevVal * prevVal;
        a.sumXY += prevVal * value;
        a.count += 1;
        a.phi = a.sumXX !== 0 ? a.sumXY / (a.sumXX + 1e-8) : 0;
      }
      updateStats(value);
      const next = [...prev, value];
      if (next.length > maxHistory) next.splice(0, next.length - maxHistory);
      persistHistory(next);
      return next;
    });
    // Markov transition update (previous bucket -> current bucket)
    // Markov removed
  }, []);

  useEffect(() => {
    (async () => {
      const stored = await localforage.getItem<number[]>(HISTORY_KEY);
      if (stored && stored.length) {
        // Rebuild stats
        stored.forEach((v) => updateStats(v));
        setHistory(stored);
      }
    })();
  }, []);

  useEffect(() => {
    if (history.length === 0) return;
    let cancelled = false;
    (async () => {
      setLoading(true);
      await tf.ready();
      if (!backend) {
        if (tf.engine().registryFactory["webgpu"]) {
          try {
            await tf.setBackend("webgpu");
            await tf.ready();
            setBackend("webgpu");
          } catch {
            await tf.setBackend("webgl");
            setBackend(tf.getBackend());
          }
        } else {
          await tf.setBackend("webgl");
          setBackend(tf.getBackend());
        }
      }
      if (!cancelled) {
        await trainAndPredict();
        setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [history, trainAndPredict, backend]);

  const ready = history.length > windowSize;

  const resetModel = useCallback(async () => {
    setLoading(true);
    // Clear persisted history
    try {
      await localforage.removeItem(HISTORY_KEY);
    } catch {}
    // Remove saved models
    try {
      await tf.io.removeModel(MODEL_KEY);
    } catch {}
    try {
      await tf.io.removeModel(CLS_MODEL_KEY);
    } catch {}
    // Dispose existing model
    if (modelRef.current) {
      modelRef.current.dispose();
      modelRef.current = null;
    }
    if (clsModelRef.current) {
      clsModelRef.current.dispose();
      clsModelRef.current = null;
    }
    // Reset refs
    statsRef.current = { count: 0, mean: 0, M2: 0, std: 0 };
    ar1Ref.current = { sumXX: 0, sumXY: 0, count: 0, phi: 0 };
    updatesRef.current = 0;
    weightsRef.current = [1 / 3, 1 / 3, 1 / 3];
    lastPredRef.current = null;
    rewardRef.current = 0;
    lowContCountsRef.current = Array(maxStreakTrack).fill(0);
    lowRevCountsRef.current = Array(maxStreakTrack).fill(0);
    highContCountsRef.current = Array(maxStreakTrack).fill(0);
    highRevCountsRef.current = Array(maxStreakTrack).fill(0);
    streakStateRef.current = { type: null, length: 0 };
    markovCountsRef.current = Array.from({ length: 11 }, () =>
      Array(11).fill(0)
    );
    lastBucketRef.current = null;
    if (saveTimeout.current) {
      clearTimeout(saveTimeout.current);
      saveTimeout.current = undefined;
    }
    // Reset state
    setHistory([]);
    setProbAbove2(null);
    setProbAbove2(null);
    calibWeightsRef.current = null;
    lastCalibFeatRef.current = null;
    lastClsBaseProbRef.current = null;
    setLoading(false);
  }, []);

  return {
    addValue,
    prediction: null,
    variance: null,
    history,
    loading,
    stats: {
      count: statsRef.current.count,
      mean: statsRef.current.mean,
      std: statsRef.current.std,
    },
    backend,
    ready,
    windowSize,
    ar1Phi: 0,
    weights: null as any,
    rewardScore: rewardRef.current,
    probAbove2,
    resetModel,
  };
}
