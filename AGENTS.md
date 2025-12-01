# Handoff Notes for Agents

Repository: `/home/myko/CS482` (git initialized, branch `main` and `portfolio` exist). No remote is set yet.

## Goal
Polished MNIST “data mining showcase” portfolio with GPU results, ready for GitHub. All artifacts and code are committed locally.

## Key Files
- `mnist_portfolio.py`: Single-entry GPU-first training script with presets (`mlp-shallow`, `mlp-deep`, `cnn`), multi-model runs (`--model all` or comma list), Next-Best Option analysis, top-k accuracy, confusion matrix, JSON/text export, and SavedModel export. Supports Keras/H5 saves if an extension is provided.
- `README.md`: Documentation, usage, GPU results table (RTX 3090), artifact locations, talking points/trade-offs.
- `results/compare.txt`, `results/compare.json`: Metrics from full GPU run (all presets).
- `results/models/`: SavedModel exports for all presets (GPU run).
- `archive_assignment/`: Original assignment artifacts (kept for reference).
- `.gitignore`: Ignores venvs, caches, notebook/HTML artifacts, OS noise.

## Latest GPU Results (RTX 3090, full MNIST)
- `mlp-shallow` (0.10M params): test acc 0.9744, Next-Best 72.66%, top-2 0.9930, top-3 0.9976.
- `mlp-deep` (0.24M params): test acc 0.9804, Next-Best 66.84%, top-2 0.9935, top-3 0.9984.
- `cnn` (0.42M params): test acc 0.9918, Next-Best 86.59%, top-2 0.9989, top-3 0.9998.
Artifacts: `results/compare.txt`, `results/compare.json`, SavedModels under `results/models/`.

## Branches / Git
- Repo initialized; `main` has the commit; branch `portfolio` created from `main`.
- No remote configured.
- Current branch: `portfolio`.

## Push Instructions (for user/next agent)
Set remote and push (replace `<URL>` with actual origin):
```bash
git remote add origin <URL>
git push -u origin main
git push -u origin portfolio   # optional
```

## Potential Next Enhancements (optional)
- Add matplotlib plots (learning curves, confusion matrix heatmaps) and embed PNGs in README.
- Add data augmentation flags to study robustness.
- Provide CSV exports of learning curves for logging.

## Environment Notes
- TensorFlow 2.20 with CUDA works on user’s WSL (RTX 3090). SavedModels already generated.
- Local Codex sandbox could not access GPU; rely on user’s environment for GPU runs.
