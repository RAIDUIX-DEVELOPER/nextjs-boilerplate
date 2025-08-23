<div align="center">
	<h1>Realtime Binary Forecast Demo</h1>
	<p>
		Adaptive online ensemble predicting the next binary event ("Over" vs "Below") using a mix of neural, probabilistic, streak / pattern and information–theoretic signals. Runs fully in-browser with TensorFlow.js (WebGPU/WebGL fallback) – no server training.
	</p>
</div>

## ✨ Overview

You supply a stream of binary outcomes (click the OVER / BELOW buttons or enable auto‑scrape). After every observation the app:

1. Updates fast Bayesian & Markov style statistics
2. Maintains run‑length (streak) continuation vs reversal posteriors
3. Updates an EWMA of the class balance
4. Re-trains (throttled) a small 1D CNN on the recent feature window
5. Computes auxiliary probability sources (model, markov, streak, pattern, ewma, bayes, entropy modulation)
6. Blends them in logit space with manual weights × adaptive RL weights
7. Applies regime / bias corrections (anti-streak + dynamic pattern dampening + bias-gap logit shift)
8. Outputs a final probability P(OVER) and records prediction correctness for metrics & RL weight adaptation

The decision threshold is kept at 0.5 (the ensemble tries to sharpen probability not shift threshold). An optional inversion toggle can flip strategy quickly if environment changes abruptly.

## 🔧 Architecture

| Layer                         | Purpose                                                                                                                 |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `app/hooks/useBinaryModel.ts` | Core state + training + probability synthesis + metrics.                                                                |
| 1D CNN (Conv1D + GAP + Dense) | Learns short temporal motifs & dependencies over sliding window.                                                        |
| Markov Counts                 | Smoothing counts with Laplace prior for last symbol transition probability.                                             |
| Streak Posteriors             | For each (symbol, run-length) track continuation vs reversal counts.                                                    |
| Pattern Heuristic             | Uses empirical distribution of recent run lengths to estimate continuation probability.                                 |
| EWMA                          | Smoothed class base-rate for stability.                                                                                 |
| Bayesian Global Rate          | Beta(a,b) posterior for OVER frequency (configurable a,b).                                                              |
| Entropy Modulation            | Recent window entropy drives decisiveness tilt favoring current symbol when low entropy.                                |
| RL Weighting                  | Multiplicative update of source weights when an OVER prediction is attempted; soft mean reversion when BELOW predicted. |
| Bias / Regime Controls        | Detect sustained prediction-vs-actual rate gap + run-length regime shift; apply damping / logit offset.                 |

### Feature Vector (per timestep)

For each point inside a training window (size 24 by default):

1. Current value (0/1)
2. Normalized current run length
3. Flip indicator (did value change from prior?)
4. Rolling mean (5)
5. Rolling mean (10)
6. Rolling mean (20)
7. Time since last OVER (capped & normalized)
8. Time since last UNDER (capped & normalized)

### Probability Sources

Each source produces a probability in (0,1). They are converted to logits, combined by normalized weights, then transformed back via sigmoid. Afterwards heuristics & bias corrections may adjust the raw probability.

### Anti-Streak & Regime Dampening

- Anti-streak pulls extreme probabilities back toward reversal when current run length exceeds a threshold.
- Regime dampening reduces pattern source deviation when current average run length < historical EMA (prevents long-streak overconfidence after shift to choppy regime).

### Bias Gap Correction

If predicted OVER rate diverges from actual OVER rate over a sliding window (>5% gap), a proportional logit offset nudges probabilities toward neutrality (without collapsing all signals to 0.5).

## 📊 Metrics Collected

- Accuracy (cumulative)
- Current & longest win / lose prediction streaks
- Average winning & losing streak length (completed streaks)
- Average actual outcome run length
- OVER prediction hit ratio (when model explicitly chose OVER)
- Source probabilities & normalized weights (manual × RL)
- Regime diagnostics (run-length EMA, current run avg, bias gap, pattern damp factor)

## 🧪 Reinforcement Weight Adaptation

When an OVER prediction is made, each source weight w is scaled by exp(eta _ sign _ (alignment - 0.5)), clipped, then re-normalized across sources. When BELOW predicted, weights decay softly toward 1 (stability). This encourages sources that aligned with correct OVER predictions.

## 🚀 Installation & Run

Prerequisites: Node.js 18+ recommended (supports WebGPU enabling flags in some browsers) and npm / pnpm / yarn.

```bash
git clone <your-fork-or-repo-url>
cd nextjs-boilerplate
npm install   # or pnpm install / yarn / bun install
npm run dev   # starts dev server at http://localhost:3000
```

Open the app and begin clicking OVER / BELOW to feed the sequence. After ~5 samples the probability components appear; after window size is exceeded the CNN begins training. Use the control panel to adjust weights, toggle inversion, set Bayesian priors, or enable auto-scrape if implemented.

### Production Build

```bash
npm run build
npm start
```

### Lint (if ESLint configured)

```bash
npm run lint
```

## 🛠 Configuration Knobs

Accessible in the control panel / hook constants:

- `windowSize` – training sequence window (24)
- `epochs` / `batchSize`
- Manual source weights (`wModel`, `wMarkov`, etc.)
- RL learning rate (`rlEta`) & enable toggle
- Anti-streak toggle
- Invert strategy toggle
- Bayesian prior (a,b)
- Entropy window length
- EWMA alpha

## ♻️ Resetting

Use the Reset button to clear history, model weights, RL state, metrics & bias tracking.

## 🧩 Extending

Ideas:

- Add Wilson score intervals for confidence display
- Add a run-length histogram visual & geometric/neg-bin fit
- Introduce separate continuation vs reversal specialized models
- Persist session to IndexedDB for continuity
- Export/import state snapshots

## 🔐 Privacy

All computation and data stay in the browser (no remote training). Auto-scrape endpoint (if used) returns only numeric values; adapt accordingly for your data source.

## ⚠️ Disclaimer

This is a demonstration / exploratory tool. Not financial advice or a production forecasting system.

## 📝 License

MIT (adjust as desired).

---

Feel free to open issues or make improvements. Enjoy experimenting with adaptive probabilistic blending in the browser.
