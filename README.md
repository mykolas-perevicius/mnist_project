# MNIST Data Mining Showcase

Single-file, GPU-first MNIST classifier that compares shallow MLP, deeper MLP, and a small CNN. It is framed as a data mining mini-project with reproducible runs, clear metrics, and error analysis you can cite.

## Why this is data mining
- Supervised learning on a canonical, balanced dataset (28x28 grayscale digits).
- Feature learning progression: dense-only (handful of hidden layers) vs convolutional filters.
- Evaluation beyond accuracy: top-k accuracy, “Next-Best Option” (soft errors), confusion matrix, per-class stats.
- Reproducibility knobs (seed, batch size, validation split) and transparent device usage (GPU preferred).

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install tensorflow[and-cuda] scikit-learn
python mnist_portfolio.py --demo --model mlp-deep --device auto
```
`--demo` runs 1 epoch on a reduced split for a fast sanity check.

## Typical runs
- Full MLP (deep) on GPU:  
  `mnist_portfolio.py --model mlp-deep --device gpu --require-gpu --save-report results/report.txt --save-metrics results/metrics.json --save-model results/models`
- CNN baseline (higher accuracy, more compute):  
  `mnist_portfolio.py --model cnn --device gpu --require-gpu`
- Comparative run (all presets, sequential):  
  `mnist_portfolio.py --model all --device auto --save-report results/compare.txt --save-metrics results/compare.json`
- CPU fallback: add `--device cpu` (omit `--require-gpu`).

Key flags:
- `--model` accepts a single preset, a comma list, or `all`.
- `--epochs`, `--batch-size`, `--validation-split` for training control.
- `--train-limit` / `--test-limit` for small experiments.
- `--save-report`, `--save-metrics`, `--save-model` to export artifacts.

## GPU results (RTX 3090, full MNIST)
| Model        | Params   | Test acc | Next-Best Option | Top-2 | Top-3 | Notes |
|--------------|----------|----------|------------------|-------|-------|-------|
| MLP shallow  | 0.10M    | 0.9744   | 72.66%           | 0.9930| 0.9976| Fastest; good soft errors. |
| MLP deep     | 0.24M    | 0.9804   | 66.84%           | 0.9935| 0.9984| Higher accuracy; slightly lower soft-error rate. |
| CNN          | 0.42M    | 0.9918   | 86.59%           | 0.9989| 0.9998| Best overall; learns spatial features. |

Artifacts from the run:
- Reports/metrics: `results/compare.txt`, `results/compare.json`
- Saved models (SavedModel): `results/models/mlp-shallow`, `results/models/mlp-deep`, `results/models/cnn`

## Outputs and how to read them
- Accuracy and loss (test set): primary classification metric.
- Top-k accuracy: ranking-style metric; how often the correct class is in the top k probabilities.
- Next-Best Option: % of misclassified samples where the model’s 2nd choice is correct (evidence the learned manifold is close).
- Confusion matrix and per-class stats: which digits drive errors (e.g., 4 vs 9, 5 vs 8).

## Suggested write-up structure (for a report)
1) Problem & data: MNIST as supervised digit classification; balanced classes; simple preprocessing (normalize + channel add).
2) Methods: shallow MLP (fewer parameters), deep MLP (more capacity), CNN (spatial filters). Device strategy (GPU-first) and seed for reproducibility.
3) Evaluation: accuracy + top-k + Next-Best Option + confusion matrix. Note training time differences and parameter counts.
4) Error analysis: highlight confusable pairs, whether 2nd-best probabilities are high, and implications for data quality/augmentation.
5) Recommendations: data augmentation (shifts/rotations), modest regularization, or larger CNN for better accuracy; consider latency/compute trade-offs.

## Real-world angles and trade-offs
- Capacity vs. speed: shallow MLP trains fastest; CNN is slower but typically most accurate.
- Robustness: CNN handles spatial variance better; MLPs are more sensitive to digit placement.
- Deployment: MLPs are lighter for edge devices; CNN better when accuracy is paramount.
- Data quality: High Next-Best Option suggests ambiguous handwriting—flag for labeling review or augmentation.

## Reproducibility notes
- Uses a global seed (`--seed`, default 42) for NumPy and TensorFlow.
- GPU selection is explicit; `--require-gpu` fails fast if no GPU is visible.
- Metrics and reports are exportable for audit trails (`results/*.json`, `results/*.txt`).

## Directory layout
- `mnist_portfolio.py` — main entrypoint.
- `archive_assignment/` — original assignment artifacts (kept for reference).

## Next steps (optional polish)
- Add data augmentation switches (random shifts/rotations) to study robustness.
- Log-friendly output (e.g., CSV for learning curves).
- Small EDA block: class counts and pixel histograms to complement the model analysis.

## Visual/easy-to-talk points
- Accuracy ladder: CNN > deep MLP > shallow MLP; CNN jumps ~1.7% absolute over shallow MLP with modest parameter increase.
- Soft-error view: CNN has the highest Next-Best Option rate (86.6%), showing confident second guesses when wrong.
- Deployment trade-offs: shallow MLP is the lightest for edge; CNN for best accuracy; deep MLP is a middle ground.
