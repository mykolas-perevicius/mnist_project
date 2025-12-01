#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_full_gpu.sh [tag]
# - Optional [tag] replaces the timestamp in the results folder name.
# - Override defaults via env vars: AUG=strong EPOCHS=15 BATCH=512 SEED=42 CUDA_VISIBLE_DEVICES=0

export PYTHONHASHSEED=${PYTHONHASHSEED:-42}
export TF_DETERMINISTIC_OPS=${TF_DETERMINISTIC_OPS:-1}
export TF_CUDNN_DETERMINISTIC=${TF_CUDNN_DETERMINISTIC:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

AUG=${AUG:-strong}
EPOCHS=${EPOCHS:-15}
BATCH=${BATCH:-512}
SEED=${SEED:-42}

STAMP=${1:-$(date -u +"%Y%m%dT%H%M%SZ")}
RUN_DIR="results/exp_${STAMP}"
mkdir -p "$RUN_DIR"/logs "$RUN_DIR"/plots "$RUN_DIR"/histories "$RUN_DIR"/models

# Snapshot environment/GPU
nvidia-smi > "$RUN_DIR/logs/nvidia-smi.txt" || true
python - <<'PY' > "$RUN_DIR/logs/env.json"
import tensorflow as tf, platform, json, os
print(json.dumps({
    'tf_version': tf.__version__,
    'cuda_built': tf.test.is_built_with_cuda(),
    'gpus': [x.name for x in tf.config.list_physical_devices('GPU')],
    'python': platform.python_version(),
    'platform': platform.platform(),
    'env': {k:v for k,v in os.environ.items() if k.startswith(('CUDA','TF_','PYTHONHASH','CUBLAS','CUDNN'))}
}, indent=2))
PY

echo "Running: tag=${STAMP} aug=${AUG} epochs=${EPOCHS} batch=${BATCH} seed=${SEED} gpu=${CUDA_VISIBLE_DEVICES}"

python mnist_portfolio.py \
  --model all \
  --device gpu --require-gpu \
  --augmentation "${AUG}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH}" \
  --seed "${SEED}" \
  --save-report "$RUN_DIR/compare.txt" \
  --save-metrics "$RUN_DIR/compare.json" \
  --save-model "$RUN_DIR/models" \
  --save-plots "$RUN_DIR/plots" \
  --save-history-csv "$RUN_DIR/histories" \
  | tee "$RUN_DIR/logs/train.log"

echo "Run complete. Artifacts in $RUN_DIR"
