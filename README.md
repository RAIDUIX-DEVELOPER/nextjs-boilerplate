<div align="center">
	<h1>Realtime Sequence Forecast & Roulette Dozens Ensemble</h1>
	<p>Adaptive in‑browser forecasting engine (binary & roulette dozens) combining neural, Bayesian, Markov, run‑length, pattern, entropy & reinforcement‑tuned signals. 100% client‑side (TF.js WebGPU/WebGL) with snapshot persistence + optional remote sync.</p>
</div>

---

## Table of Contents
1. Executive Overview
2. End‑to‑End Data Flow
3. Core Hook & State Topology
4. Probability Pipeline (Binary)
5. Multi‑Class Dozens Mode Architecture
6. Reinforcement Weight Adaptation Math
7. Auto‑Tuning & Diagnostic Suggestions
8. Bias / Regime / Risk Safeguards
9. Calibration, Reliability & Temperature Control (Dozens)
10. Persistence & Remote Sync Design
11. Performance & Throttling Strategies
12. Configuration Surface
13. Extensibility Points
14. Troubleshooting Guide
15. Installation / Commands
16. Privacy & Disclaimer

---

## 1. Executive Overview
User supplies a stream of outcomes:
* Binary Over/Under (or Red/Black proxy) – label ∈ {0,1}
* Roulette Dozens – label ∈ {0,1,2}

After each observation the engine updates statistical structures, (re)trains light models when gated conditions are met, recalculates probability components, blends them via logit ensemble, applies adaptive safety / regime logic, records metrics, updates reinforcement weights, persists state, and (optionally) pushes a throttled snapshot to a remote endpoint.

## 2. End‑to‑End Data Flow
```
UI Event (button / scrape) --> addObservation / addDozenObservation
	-> Update histories & records
	-> Update run-length & Markov counts
	-> Update streak / pattern / Bayesian / EWMA structures
	-> RL weight update (post-outcome)
	-> Metrics + diagnostics update
	-> Schedule prediction recompute (throttled)
	-> Maybe train model(s) (gated cadence)
	-> Persist snapshot (debounced) -> localStorage (+ remote push if signature changed)
	-> UI re-renders via state + derived diagnostics
```

## 3. Core Hook & State Topology (`useBinaryModel.ts`)
Key refs/state clusters:
* Histories: `history` (binary), `dozenHistory` (multi-class)
* Records (prediction + outcome pairs) for analysis & metrics
* Models: binary CNN, dozens Conv1D (optional) – lazily built
* Probability components store (`probPartsRef`)
* Reinforcement weights (`rlWeightsRef`) normalized around 1
* Bias / regime tracking (recent prediction vs actual gap, run-length EMA)
* Dozens reliability, calibration, hi probability miss trackers, shock/cooldown, exploration safeguards
* Persistence & remote sync refs (last snapshot sig, throttles, disable flags)

## 4. Probability Pipeline (Binary Mode)
Components producing p ∈ (0,1):
| Symbol | Description |
| ------ | ----------- |
| model | CNN output on latest window features |
| markov | P(next=1 | last) from Laplace-smoothed transition row |
| streak | Continuation vs reversal posterior using (symbol,runLen) counts |
| pattern | Run-length distribution continuation adjustment with regime damping |
| ewma | Exponential moving average of class frequency |
| bayes | Beta(a,b) posterior mean for global Over frequency |
| entropyMod | Entropy-based modulation shifting toward recent dominant side in low-entropy regimes |

Blending:
1. Convert each pᵢ → logit(pᵢ)
2. Weight wᵢ = (manualWeightᵢ × rlWeightᵢ) / Σ
3. blendedLogit = Σ wᵢ·logit(pᵢ)
4. rawProb = sigmoid(blendedLogit)
5. Adjustments:
	 * Bias gap correction (logit shift toward neutrality)
	 * Pattern regime dampening (scales pattern deviation)
	 * Anti-streak moderation (pull toward reversal after long runs)
	 * Optional inversion (final p' = 1 - p)

## 5. Multi‑Class Dozens Mode Architecture
Additional layers beyond binary:
* Model distribution (3-way) over 1st/2nd/3rd dozen
* Markov-like transition counts for last dozen
* Run-length continuation vs reversal per prior dozen
* Multi-source distribution blending with RL weights (normalized to 7 sources inclusive)
* Reliability sums (Σ predicted prob vs Σ actual) per class
* Calibration buffer + multiplicative per-class scaling + adaptive temperature
* Hi-prob event tracking (≥ thresholds) for cooldown or shock activation
* Shock state: temporary entropy enforcement & cap decay adjustments after clusters of high-confidence misses
* Gap hazard modeling: time since class occurrence to avoid starvation / tunnel vision
* Regret tracking: penalize repeated over-commitment to underperforming class
* Anti-tunnel heuristics & abstain logic (optionally gating predictions, can be extended)

## 6. Reinforcement Weight Adaptation Math
For each source s with current weight wₛ and distribution contribution pₛ (probability assigned to true label):
```
edge = pₛ - baseRate   // baseRate = 0.5 (binary) or 1/3 (dozens)
reward = +1 if correct else -1
Δ = eta * reward * edge
wₛ' = clamp(wₛ * exp(Δ), [0.2,5])
Normalize all w' to keep mean ≈ 1 (divide by (Σ w' / N)).
```
Confidence modulation & damp factors adjust eta (stronger updates for confident mistakes, milder for ambiguous hits).

## 7. Auto‑Tuning & Diagnostic Suggestions
Diagnostic builder inspects sliding windows for:
* Over / under confidence gap (avg top prob vs realized accuracy)
* Elevated Brier score
* Long actual streak + low top-prob (reversal risk)
* Repeated misses on same class / label

Triggers accumulate counters; once thresholds crossed, specific manual weights are softly nudged (bounded multiplicative factors) then persisted.

## 8. Bias / Regime / Risk Safeguards
| Mechanism | Trigger | Effect |
| --------- | ------- | ------ |
| Bias Gap | Pred rate vs actual rate divergence | Logit centering shift |
| Regime Damp | Recent avg run length < EMA | Scale pattern deviation toward 0.5 |
| Anti-Streak | Run length ≥ threshold | Blend partially toward reversal probability |
| Cooldown (Dozens) | High-prob miss (≥0.6) | Temporary factor increasing exploration / dampening |
| Shock | Cluster of hi-prob misses or miss-rate spike | Extended entropy / cap decay adjustments |
| Hi-Prob Miss Rate | Rolling qualifying events | Monitors reliability, interacts with shock logic |

## 9. Calibration, Reliability & Temperature (Dozens)
Steps:
1. Maintain rolling window of predicted distributions & outcomes.
2. For each class i compute Σ predicted pᵢ and actual count aᵢ.
3. Multiplicative reliability multiplier mᵢ ← blend(old, aᵢ/Σpᵢ) (bounded 0.6–1.6)
4. Compute top-prob gap (avg top p − accuracy of top events) → target temperature
5. Smooth temperature toward target; apply before final distribution normalization to temper overconfidence.

Final per-class adjusted probability:
```
qᵢ = softmax( (log(pᵢ * mᵢ) / T) )
```

## 10. Persistence & Remote Sync
Local: versioned localStorage keys (namespaced) store:
* Histories & records (binary + dozens) truncated to `PERSIST_LIMIT` (5000)
* Full snapshot (controls, rlWeights, last predictions, reliability, calibration, metrics)
* Serialized models via `tf.io` localstorage handler

Remote: POST `/api/state` with consolidated snapshot payload.
* Throttled by `remotePushThrottleMs` (10s) unless explicit remote force
* Signature = lengths tuple (`binHistory|binRecords|dzHistory|dzRecords`)
* Only push if signature changed (idempotent, lightweight)
* 501 response marks remote disabled (silences further attempts)

Force paths:
* Manual "Save Now" button
* Page hide / unload event (bypass remote throttle)

## 11. Performance & Throttling
| Aspect | Strategy |
| ------ | -------- |
| Probability recompute | 60ms throttle + coalesced control-change debounce (50ms) |
| Model training | Gated by sample count & cadence (`TRAIN_START`, `TRAIN_EVERY`); only retrain if new samples since last train |
| Persistence | Local debounce (`persistThrottleMs`) + signature diff for remote |
| Rendering | Derived diagnostics computed memoized inside hook; heavy arrays truncated |
| Histories | Hard cap (5000) to bound memory & snapshot size |

## 12. Configuration Surface (Selected)
| Key | Meaning | Default |
| --- | ------- | ------- |
| `windowSize` | CNN sequence length | 24 |
| `epochs` | Mini training epochs | 4 (example) |
| `rlEta` | Base RL LR | 0.04 (example) |
| `persistThrottleMs` | Local snapshot min interval | 2500 |
| `remotePushThrottleMs` | Minimum remote sync spacing | 10000 |
| `PERSIST_LIMIT` | Max retained spins per store | 5000 |

## 13. Extensibility Points
* Add new probability component: compute p, add to `probParts`, extend blending weights & RL update loop.
* Alternate neural backbone: swap small CNN for Transformer-lite (1–2 heads) if sequence semantics evolve.
* Add abstention: insert thresholded calibration gating returning null predictions.
* Introduce server federation: push gradients or compressed counts (currently full snapshot only).

## 14. Troubleshooting
| Symptom | Likely Cause | Action |
| ------- | ------------ | ------ |
| UI freezes on rapid slider drag | Excess recomputes | Throttle already added; raise 60ms value if still heavy |
| Repeated remote POST spam | Force push path triggering | Verify only manual save/unload uses `remoteForce` |
| WebGPU adapter errors (Windows) | Unsupported adapter info call | WebGL fallback; noise is benign |
| Probabilities stuck at 0.5 | Insufficient samples (< window) | Continue feeding data until window fills |
| Overconfidence spikes | Calibration window too small | Increase window or inspect hi-prob miss metrics |

## 15. Installation & Commands
Prerequisites: Node 18+.
```bash
git clone <your-fork-or-repo-url>
cd nextjs-boilerplate
npm install
npm run dev
```
Optional:
```bash
npm run build && npm start   # production
npm run lint                 # lint (if configured)
```

## 16. Privacy & Disclaimer
All computation + training is local. Remote sync (if enabled) transmits only the snapshot JSON you already generate client‑side. No guarantees of forecasting performance; for educational / experimental use only.

---
Contributions welcome—open a PR to add components, new calibration layers, or alternative reinforcement schemes.
