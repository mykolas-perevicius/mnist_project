#!/usr/bin/env python3
"""
MNIST portfolio script: GPU-first training + educational analysis in one file.

Features:
  - GPU-preferred device selection with clear diagnostics and safe fallbacks.
  - MLP (shallow/deep) and CNN presets; easy CLI overrides for epochs/batch size.
  - Next-Best Option error analysis, confusion matrix, top-k accuracy.
  - Optional artifacts: metrics JSON, text report, saved model.
  - Reproducible seeding and an optional fast demo mode for smoke-testing.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
from typing import Any, Dict, Tuple

# WSL GPU tip: expose Windows-side CUDA libs to the linker before importing TF.
_wsl_cuda_path = "/usr/lib/wsl/lib"
if os.path.isdir(_wsl_cuda_path):
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in ld_path.split(":") if p]
    if _wsl_cuda_path not in parts:
        parts.insert(0, _wsl_cuda_path)
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 128
DEFAULT_VAL_SPLIT = 0.1
DEFAULT_AUGMENTATION = "none"

MODEL_PRESETS = {
    "mlp-shallow": {"hidden_layers": [128], "epochs": 10},
    "mlp-deep": {"hidden_layers": [256, 128], "epochs": 12},
    "cnn": {"epochs": 8},
}


# -----------------------------------------------------------------------------#
# Utility helpers
# -----------------------------------------------------------------------------#
def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def resolve_device(preference: str, require_gpu: bool) -> Tuple[str, Dict[str, Any]]:
    """
    Decide which device to use and gather diagnostics.

    preference: "gpu", "cpu", or "auto" (prefer GPU if present).
    require_gpu: if True, exit when no GPU is available.
    """
    info: Dict[str, Any] = {
        "preference": preference,
        "require_gpu": require_gpu,
        "built_with_cuda": None,
        "physical_gpus": [],
        "logical_gpus": [],
        "errors": [],
        "target_device": "/CPU:0",
        "using_gpu": False,
        "wsl_cuda_path_in_env": _wsl_cuda_path in os.environ.get("LD_LIBRARY_PATH", ""),
    }

    info["built_with_cuda"] = bool(tf.test.is_built_with_cuda())

    try:
        info["physical_gpus"] = tf.config.list_physical_devices("GPU")
    except Exception as exc:  # pragma: no cover
        info["errors"].append(f"list_physical_devices failed: {exc!r}")

    try:
        info["logical_gpus"] = tf.config.list_logical_devices("GPU")
    except Exception as exc:  # pragma: no cover
        info["errors"].append(f"list_logical_devices failed: {exc!r}")

    gpus_available = bool(info["physical_gpus"])

    if preference == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception as exc:
            info["errors"].append(f"set_visible_devices([], 'GPU') failed: {exc!r}")
        target = "/CPU:0"
    elif preference in {"gpu", "auto"}:
        if gpus_available:
            target = "/GPU:0"
            info["using_gpu"] = True
            try:
                for gpu in info["physical_gpus"]:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as exc:
                info["errors"].append(f"memory growth failed: {exc!r}")
        else:
            target = "/CPU:0"
    else:
        raise ValueError(f"Unknown device preference: {preference}")

    info["target_device"] = target

    if require_gpu and target != "/GPU:0":
        sys.exit("Requested GPU-only run, but no GPU was found by TensorFlow.")

    return target, info


def print_device_banner(info: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("Device configuration")
    print("=" * 60)
    print(f"Preference      : {info['preference']}")
    print(f"Require GPU     : {info['require_gpu']}")
    print(f"Built w/ CUDA   : {info['built_with_cuda']}")
    print(f"Physical GPUs   : {info['physical_gpus']}")
    print(f"Logical GPUs    : {info['logical_gpus']}")
    print(f"Using device    : {info['target_device']}")
    if info.get("wsl_cuda_path_in_env") is False and os.path.isdir(_wsl_cuda_path):
        print("Note: /usr/lib/wsl/lib was not present in LD_LIBRARY_PATH; added for this run.")
    if info["errors"]:
        print("\nNon-fatal device setup notes:")
        for err in info["errors"]:
            print(f"  - {err}")
    print("=" * 60 + "\n")


# -----------------------------------------------------------------------------#
# Data
# -----------------------------------------------------------------------------#
def load_and_preprocess_mnist(train_limit: int | None, test_limit: int | None) -> Tuple:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    if train_limit:
        x_train = x_train[:train_limit]
        y_train = y_train[:train_limit]
    if test_limit:
        x_test = x_test[:test_limit]
        y_test = y_test[:test_limit]

    print(f"Train images: {x_train.shape}  Labels: {y_train.shape}")
    print(f"Test  images: {x_test.shape}  Labels: {y_test.shape}\n")
    return (x_train, y_train), (x_test, y_test)


# -----------------------------------------------------------------------------#
# Models
# -----------------------------------------------------------------------------#
def make_augmentation_layers(level: str) -> keras.Sequential | None:
    """Create lightweight augmentation layers that only run during training."""
    level = level.lower()
    if level == "none":
        return None

    if level == "light":
        return keras.Sequential(
            [
                layers.RandomRotation(0.08),
                layers.RandomTranslation(0.08, 0.08),
            ],
            name="augment_light",
        )

    if level == "strong":
        return keras.Sequential(
            [
                layers.RandomRotation(0.15),
                layers.RandomTranslation(0.12, 0.12),
                layers.RandomZoom(0.2),
            ],
            name="augment_strong",
        )

    raise ValueError(f"Unknown augmentation level: {level}")


def build_mlp(hidden_layers: list[int], augment: keras.Sequential | None = None) -> keras.Model:
    inputs = keras.Input(shape=(28, 28, 1), name="input_image")
    x = inputs
    if augment is not None:
        x = augment(x)
    x = layers.Flatten(name="flatten")(x)
    for idx, units in enumerate(hidden_layers, start=1):
        x = layers.Dense(units, activation="relu", name=f"dense_{idx}")(x)
    outputs = layers.Dense(10, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="MLP")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_cnn(augment: keras.Sequential | None = None) -> keras.Model:
    inputs = keras.Input(shape=(28, 28, 1), name="input_image")
    x = inputs
    if augment is not None:
        x = augment(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(10, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="CNN")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# -----------------------------------------------------------------------------#
# Analysis
# -----------------------------------------------------------------------------#
def next_best_option_analysis(probs: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)
    pred_labels = np.argmax(probs, axis=1)

    mis_idx = np.where(pred_labels != y_true)[0]
    num_mis = len(mis_idx)

    close_call_count = 0
    per_class_mis = np.zeros(10, dtype=int)
    per_class_close = np.zeros(10, dtype=int)

    for idx in mis_idx:
        true_label = int(y_true[idx])
        per_class_mis[true_label] += 1

        prob_vec = probs[idx]
        sorted_indices = np.argsort(prob_vec)
        top1 = sorted_indices[-1]
        top2 = sorted_indices[-2]

        if top2 == true_label:
            close_call_count += 1
            per_class_close[true_label] += 1

    overall_percentage = (close_call_count / num_mis * 100.0) if num_mis > 0 else 0.0
    per_class_percentage = np.zeros(10, dtype=float)
    for digit in range(10):
        if per_class_mis[digit] > 0:
            per_class_percentage[digit] = (
                per_class_close[digit] / per_class_mis[digit] * 100.0
            )

    return {
        "num_samples": int(len(y_true)),
        "num_misclassified": int(num_mis),
        "num_close_calls": int(close_call_count),
        "overall_percentage": float(overall_percentage),
        "per_class_misclassified": per_class_mis.tolist(),
        "per_class_close_calls": per_class_close.tolist(),
        "per_class_percentage": per_class_percentage.tolist(),
    }


def compute_top_k_accuracy(probs: np.ndarray, y_true: np.ndarray, k: int = 2) -> float:
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)
    top_k_indices = np.argsort(probs, axis=1)[:, -k:]
    hits = sum(label in top_k_indices[i] for i, label in enumerate(y_true))
    return hits / len(y_true)


def summarize_history(history: keras.callbacks.History) -> Dict[str, float]:
    hist = history.history
    return {
        "final_train_loss": float(hist["loss"][-1]),
        "final_train_accuracy": float(hist["accuracy"][-1]),
        "final_val_loss": float(hist["val_loss"][-1]) if "val_loss" in hist else None,
        "final_val_accuracy": float(hist["val_accuracy"][-1]) if "val_accuracy" in hist else None,
    }


# -----------------------------------------------------------------------------#
# Orchestration
# -----------------------------------------------------------------------------#
def train_and_evaluate(
    model_builder,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: str,
    epochs: int,
    batch_size: int,
    validation_split: float,
    verbose: int = 2,
):
    with tf.device(device):
        model = model_builder()
        print("\nModel summary:")
        model.summary()
        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
        )
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        probs = model.predict(x_test, batch_size=batch_size, verbose=0)
    return model, history, test_loss, test_acc, probs


def save_model_artifact(model: keras.Model, base_path: pathlib.Path, preset_name: str, multi: bool) -> pathlib.Path:
    """
    Save Keras model with sensible defaults:
      - If target has .keras or .h5, use Keras save.
      - Otherwise, export SavedModel into a directory.
      - When multiple presets are run and a file extension is provided,
        suffix the filename with the preset name.
    """
    target = base_path

    if multi:
        if base_path.suffix:
            target = base_path.with_name(f"{base_path.stem}-{preset_name}{base_path.suffix}")
        else:
            target = base_path / preset_name

    if target.suffix in {".keras", ".h5"}:
        target.parent.mkdir(parents=True, exist_ok=True)
        model.save(target)
    else:
        target.mkdir(parents=True, exist_ok=True)
        # Keras 3 export -> SavedModel (good for TF Serving / TFLite conversion)
        model.export(target)

    print(f"Saved model to {target}")
    return target


def build_report(
    model_name: str,
    test_loss: float,
    test_acc: float,
    nb_stats: Dict[str, Any],
    top2: float,
    top3: float,
    cls_report: str,
    confusion: np.ndarray,
    device: str,
    augmentation: str,
) -> str:
    lines = [
        f"Model: {model_name}",
        f"Device: {device}",
        f"Augmentation   : {augmentation}",
        f"Test loss: {test_loss:.4f}",
        f"Test accuracy: {test_acc:.4f}",
        f"Next-Best Option (overall): {nb_stats['overall_percentage']:.2f}%",
        f"Top-2 accuracy: {top2:.4f}",
        f"Top-3 accuracy: {top3:.4f}",
        "",
        "Per-class 'Next-Best Option' percentages (only among misclassifications):",
    ]
    for digit, pct in enumerate(nb_stats["per_class_percentage"]):
        lines.append(f"  Digit {digit}: {pct:5.1f}%")

    lines.append("\nConfusion matrix:\n")
    lines.append(np.array2string(confusion))
    lines.append("\nClassification report:\n")
    lines.append(cls_report)
    return "\n".join(lines)


def resolve_generic_path(base_path: pathlib.Path, preset_name: str, multi: bool, default_filename: str) -> pathlib.Path:
    """
    If base_path has a suffix, use it as the filename (adding preset when multi).
    Otherwise, treat base_path as a directory and append default_filename
    (adding preset when multi).
    """
    if base_path.suffix:
        if multi:
            return base_path.with_name(f"{base_path.stem}-{preset_name}{base_path.suffix}")
        return base_path

    base_path.mkdir(parents=True, exist_ok=True)
    stem, ext = os.path.splitext(default_filename)
    filename = f"{stem}-{preset_name}{ext}" if multi else default_filename
    return base_path / filename


def resolve_plot_path(base_path: pathlib.Path, preset_name: str, multi: bool, kind: str) -> pathlib.Path:
    """
    Plot outputs are always treated as a directory-style target with stable names.
    If base_path has a suffix, its stem becomes the prefix and the parent holds the plots.
    """
    if base_path.suffix:
        directory = base_path.parent
        prefix = base_path.stem
    else:
        directory = base_path
        prefix = "mnist"

    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}-{kind}"
    if multi:
        filename += f"-{preset_name}"
    filename += ".png"
    return directory / filename


def save_history_csv(history: keras.callbacks.History, path: pathlib.Path) -> pathlib.Path | None:
    metrics = history.history
    if not metrics:
        print("No training history to save.")
        return None

    keys = list(metrics.keys())
    epochs = len(metrics[keys[0]])
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", *keys])
        for idx in range(epochs):
            row = [idx + 1] + [metrics[key][idx] for key in keys]
            writer.writerow(row)

    print(f"Saved learning-curve CSV to {path}")
    return path


def _get_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        print("matplotlib is not installed; skipping plot exports.")
        return None
    return plt


def plot_learning_curves(history: keras.callbacks.History, title: str, out_path: pathlib.Path) -> pathlib.Path | None:
    plt = _get_matplotlib_pyplot()
    if plt is None:
        return None

    hist = history.history
    plt.figure(figsize=(7, 4))
    if "accuracy" in hist:
        plt.plot(hist["accuracy"], label="train accuracy")
    if "val_accuracy" in hist:
        plt.plot(hist["val_accuracy"], label="val accuracy")
    if "loss" in hist:
        plt.plot(hist["loss"], label="train loss", linestyle="--", alpha=0.7)
    if "val_loss" in hist:
        plt.plot(hist["val_loss"], label="val loss", linestyle="--", alpha=0.7)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot to {out_path}")
    return out_path


def plot_confusion_matrix(confusion: np.ndarray, title: str, out_path: pathlib.Path) -> pathlib.Path | None:
    plt = _get_matplotlib_pyplot()
    if plt is None:
        return None

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(10),
        yticks=np.arange(10),
        xticklabels=list(range(10)),
        yticklabels=list(range(10)),
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = confusion.max() / 2.0 if confusion.max() > 0 else 0
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(
                j,
                i,
                format(confusion[i, j], "d"),
                ha="center",
                va="center",
                color="white" if confusion[i, j] > thresh else "black",
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {out_path}")
    return out_path


# -----------------------------------------------------------------------------#
# CLI
# -----------------------------------------------------------------------------#
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MNIST models with GPU-first defaults and rich analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="mlp-deep",
        help=(
            "Model preset to train. Accepts a single preset "
            f"({', '.join(MODEL_PRESETS.keys())}) or a comma list, "
            "or 'all' to run every preset."
        ),
    )
    parser.add_argument("--epochs", type=int, help="Override epoch count (applies to all selected models).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="gpu")
    parser.add_argument("--require-gpu", action="store_true", help="Fail fast if no GPU is visible.")
    parser.add_argument("--validation-split", type=float, default=DEFAULT_VAL_SPLIT)
    parser.add_argument(
        "--augmentation",
        choices=["none", "light", "strong"],
        default=DEFAULT_AUGMENTATION,
        help="Optional on-the-fly data augmentation (training only).",
    )
    parser.add_argument("--train-limit", type=int, default=None, help="Optional train set cap for quick runs.")
    parser.add_argument("--test-limit", type=int, default=None, help="Optional test set cap for quick runs.")
    parser.add_argument("--demo", action="store_true", help="Shortcut for a quick smoke test (1 epoch, small split).")
    parser.add_argument("--save-metrics", type=pathlib.Path, help="Path to write metrics JSON.")
    parser.add_argument("--save-report", type=pathlib.Path, help="Path to write a readable text report.")
    parser.add_argument("--save-model", type=pathlib.Path, help="Directory to export the trained model (SavedModel).")
    parser.add_argument(
        "--save-history-csv",
        type=pathlib.Path,
        help="Optional CSV export of learning curves (accuracy/loss per epoch).",
    )
    parser.add_argument(
        "--save-plots",
        type=pathlib.Path,
        help="Directory or filename prefix to store PNG plots (learning curves + confusion matrix).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    # Resolve device
    device, device_info = resolve_device(args.device, args.require_gpu)
    print_device_banner(device_info)

    # Data
    train_limit = args.train_limit
    test_limit = args.test_limit
    epochs_override = args.epochs
    if args.demo:
        train_limit = train_limit or 6000
        test_limit = test_limit or 2000
        epochs_override = epochs_override or 1
        print("Demo mode: using smaller splits and 1 epoch for a quick smoke test.\n")

    (x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist(train_limit, test_limit)

    # Model selection (supports comma-separated or 'all')
    if args.model.lower() == "all":
        model_names = list(MODEL_PRESETS.keys())
    else:
        model_names = [name.strip() for name in args.model.split(",") if name.strip()]
    unknown = [m for m in model_names if m not in MODEL_PRESETS]
    if unknown:
        sys.exit(f"Unknown model preset(s): {unknown}")

    multi_run = len(model_names) > 1
    all_reports = []
    all_metrics = []

    for model_name in model_names:
        preset = MODEL_PRESETS[model_name]
        epochs = epochs_override or preset["epochs"]

        def builder(model_name=model_name):
            augment_layer = make_augmentation_layers(args.augmentation)
            if model_name == "cnn":
                return build_cnn(augment_layer)
            return build_mlp(preset["hidden_layers"], augment_layer)

        print(f"\n=== Training preset: {model_name} ===")
        model, history, test_loss, test_acc, probs = train_and_evaluate(
            builder,
            x_train,
            y_train,
            x_test,
            y_test,
            device=device,
            epochs=epochs,
            batch_size=args.batch_size,
            validation_split=args.validation_split,
        )

        # Analysis
        y_pred = np.argmax(probs, axis=1)
        nb_stats = next_best_option_analysis(probs, y_test)
        top2_acc = compute_top_k_accuracy(probs, y_test, k=2)
        top3_acc = compute_top_k_accuracy(probs, y_test, k=3)
        cls_report = classification_report(y_test, y_pred, digits=4)
        hist_summary = summarize_history(history)
        confusion = confusion_matrix(y_test, y_pred)

        # Reporting
        report_text = build_report(
            model_name=model.name,
            test_loss=test_loss,
            test_acc=test_acc,
            nb_stats=nb_stats,
            top2=top2_acc,
            top3=top3_acc,
            cls_report=cls_report,
            confusion=confusion,
            device=device,
            augmentation=args.augmentation,
        )
        print(report_text)
        all_reports.append(report_text)

        metrics = {
            "model": model.name,
            "preset": model_name,
            "device": device,
            "augmentation": args.augmentation,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "top2_accuracy": top2_acc,
            "top3_accuracy": top3_acc,
            "next_best_option": nb_stats,
            "confusion_matrix": confusion.tolist(),
            "history": hist_summary,
        }
        all_metrics.append(metrics)

        if args.save_history_csv:
            history_path = resolve_generic_path(
                base_path=args.save_history_csv,
                preset_name=model_name,
                multi=multi_run,
                default_filename="history.csv",
            )
            save_history_csv(history, history_path)

        if args.save_plots:
            curve_path = resolve_plot_path(
                base_path=args.save_plots,
                preset_name=model_name,
                multi=multi_run,
                kind="learning",
            )
            plot_learning_curves(history, f"{model.name} learning curves", curve_path)

            conf_path = resolve_plot_path(
                base_path=args.save_plots,
                preset_name=model_name,
                multi=multi_run,
                kind="confusion",
            )
            plot_confusion_matrix(confusion, f"{model.name} confusion matrix", conf_path)

        if args.save_model:
            save_model_artifact(
                model=model,
                base_path=args.save_model,
                preset_name=model_name,
                multi=multi_run,
            )

    # Persist artifacts if requested
    if args.save_metrics:
        args.save_metrics.parent.mkdir(parents=True, exist_ok=True)
        with args.save_metrics.open("w", encoding="utf-8") as f:
            json.dump(all_metrics if len(all_metrics) > 1 else all_metrics[0], f, indent=2)
        print(f"\nSaved metrics to {args.save_metrics}")

    if args.save_report:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        full_report = "\n\n".join(all_reports)
        args.save_report.write_text(full_report, encoding="utf-8")
        print(f"Saved report to {args.save_report}")

    print("\nRun complete.")


if __name__ == "__main__":
    main()
