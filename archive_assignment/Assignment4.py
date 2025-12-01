"""
MNIST Neural Network Assignment + Optional GPU-Enhanced Analysis (WSL2-ready)

Implements:
  - Model A (shallow MLP)
  - Model B (deeper MLP)
  - Training & test evaluation
  - "Next-Best Option" error analysis
  - Extended analysis: confusion matrix, per-class stats, top-k accuracy
  - Optional Model C (CNN) as a stronger baseline

Behavior:
  - If a GPU is visible to TensorFlow, it will be used automatically.
  - If no GPU is found, the script runs entirely on CPU and prints a clear message.
"""

# -------------------------------------------------------------------
# Environment configuration (BEFORE importing TensorFlow)
# -------------------------------------------------------------------
import os

# Keep C++ logs quieter (but don't hide real errors)
# 0 = all logs, 1 = filter INFO, 2 = filter INFO+WARNING, 3 = only ERROR
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report

# -------------------------------------------------------------------
# Config flags (optional project knobs)
# -------------------------------------------------------------------
ENABLE_DEEPER_ANALYSIS = True      # confusion matrix, per-class stats, top-k accuracy
ENABLE_CNN_EXPERIMENT = True       # train a small CNN (Model C)
ENABLE_PLOTTING = False            # set True if you want matplotlib plots

SEED = 42
EPOCHS_MLP = 10
EPOCHS_CNN = 8
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1

np.random.seed(SEED)
tf.random.set_seed(SEED)


# -------------------------------------------------------------------
# Device configuration
# -------------------------------------------------------------------
def configure_device():
    print("============================================================")
    print("Device configuration")
    print("============================================================")

    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")

    if gpus:
        print("✅ GPU(s) detected by TensorFlow:")
        for gpu in gpus:
            print("  -", gpu)
        # Optional: enable memory growth to avoid grabbing all GPU RAM at once
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth enabled on available GPU(s).")
        except Exception as e:
            print("Warning: could not enable GPU memory growth:", e)
    else:
        print("⚠️  No GPU detected by TensorFlow. The script will run on CPU.")
        print("    If you expect a GPU in WSL2, check:")
        print("      1) `nvidia-smi` works inside WSL2")
        print("      2) You installed the GPU build:  pip install 'tensorflow[and-cuda]'")
        print("")

    print("CPU devices seen by TensorFlow:")
    for c in cpus:
        print("  -", c)
    print("============================================================\n")


# -------------------------------------------------------------------
# Data loading & preprocessing
# -------------------------------------------------------------------
def load_and_preprocess_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension: (28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    print(f"Train images: {x_train.shape} Train labels: {y_train.shape}")
    print(f"Test  images: {x_test.shape} Test  labels: {y_test.shape}\n")
    return (x_train, y_train), (x_test, y_test)


# -------------------------------------------------------------------
# Model builders
# -------------------------------------------------------------------
def build_mlp_model(num_hidden_layers=1, hidden_units=10, name=None):
    inputs = keras.Input(shape=(28, 28, 1), name="input_image")
    x = layers.Flatten(name="flatten")(inputs)
    for i in range(num_hidden_layers):
        x = layers.Dense(hidden_units, activation="relu", name=f"hidden_{i+1}")(x)
    outputs = layers.Dense(10, activation="softmax", name="output")(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=name or f"MLP_{num_hidden_layers}x{hidden_units}",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_model():
    """
    Optional Model C: CNN baseline that usually outperforms the MLPs.
    """
    inputs = keras.Input(shape=(28, 28, 1), name="input_image")
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1")(inputs)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(10, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="CNN_model")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# -------------------------------------------------------------------
# Training helper
# -------------------------------------------------------------------
def train_model(model, x_train, y_train, epochs,
                batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, verbose=2):
    print(f"\n===== Training {model.name} =====")
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose,
    )
    return history


# -------------------------------------------------------------------
# Next-Best Option error analysis
# -------------------------------------------------------------------
def next_best_option_analysis(probs, y_true):
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
        sorted_indices = np.argsort(prob_vec)  # ascending
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

    stats = {
        "num_samples": int(len(y_true)),
        "num_misclassified": int(num_mis),
        "num_close_calls": int(close_call_count),
        "overall_percentage": float(overall_percentage),
        "per_class_misclassified": per_class_mis,
        "per_class_close_calls": per_class_close,
        "per_class_percentage": per_class_percentage,
    }
    return stats


# -------------------------------------------------------------------
# Additional analysis utilities (optional project section)
# -------------------------------------------------------------------
def compute_top_k_accuracy(probs, y_true, k=2):
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)
    top_k_indices = np.argsort(probs, axis=1)[:, -k:]
    hits = sum(label in top_k_indices[i] for i, label in enumerate(y_true))
    return hits / len(y_true)


def print_confusion_and_report(y_true, y_pred, title=""):
    cm = confusion_matrix(y_true, y_pred)
    print("\n" + "=" * 60)
    if title:
        print(f"Confusion Matrix & Classification Report for {title}")
    else:
        print("Confusion Matrix & Classification Report")
    print("=" * 60)
    print("Confusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))
    return cm


def show_example_misclassifications(x, y_true, probs, num_examples=10,
                                    seed=SEED, close_call_only=True):
    import matplotlib.pyplot as plt

    probs = np.asarray(probs)
    y_true = np.asarray(y_true)
    pred_labels = np.argmax(probs, axis=1)

    mis_idx = np.where(pred_labels != y_true)[0]
    if len(mis_idx) == 0:
        print("No misclassifications to visualize.")
        return

    rng = np.random.default_rng(seed)
    rng.shuffle(mis_idx)

    shown = 0
    plt.figure(figsize=(12, 6))
    for idx in mis_idx:
        if shown >= num_examples:
            break
        img = x[idx].squeeze()
        true_label = int(y_true[idx])
        prob_vec = probs[idx]
        sorted_indices = np.argsort(prob_vec)
        top1 = int(sorted_indices[-1])
        top2 = int(sorted_indices[-2])
        top1_p = prob_vec[top1]
        top2_p = prob_vec[top2]

        if close_call_only and top2 != true_label:
            continue

        shown += 1
        plt.subplot(2, (num_examples + 1) // 2, shown)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(
            f"True: {true_label}\n"
            f"Pred: {top1} ({top1_p:.2f})\n"
            f"2nd: {top2} ({top2_p:.2f})"
        )

    if shown == 0:
        print("No 'close call' misclassifications found to visualize.")
        return

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------
def main():
    configure_device()

    # ------------------------
    # Part 1: Data Preparation
    # ------------------------
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist()

    # ------------------------
    # Part 2: Model Building
    # ------------------------
    model_A = build_mlp_model(num_hidden_layers=1, hidden_units=10, name="Model_A_shallow")
    model_B = build_mlp_model(num_hidden_layers=2, hidden_units=10, name="Model_B_deep")

    print("Model A Summary:")
    model_A.summary()
    print("\nModel B Summary:")
    model_B.summary()

    # ------------------------
    # Part 3: Training & Evaluation
    # ------------------------
    train_model(model_A, x_train, y_train, epochs=EPOCHS_MLP)
    train_model(model_B, x_train, y_train, epochs=EPOCHS_MLP)

    test_loss_A, test_acc_A = model_A.evaluate(x_test, y_test, verbose=0)
    test_loss_B, test_acc_B = model_B.evaluate(x_test, y_test, verbose=0)

    print("\n" + "=" * 60)
    print("Test Accuracy Results")
    print("=" * 60)
    print(f"Model A (1 hidden layer) - Test Accuracy: {test_acc_A:.4f}")
    print(f"Model B (2 hidden layers) - Test Accuracy: {test_acc_B:.4f}")

    # Decide which model to use for error analysis
    accuracy_diff = abs(test_acc_A - test_acc_B)
    if accuracy_diff < 0.005:
        chosen_model = model_B
        chosen_name = "Model B (deep) [chosen: accuracy similar to Model A]"
    elif test_acc_B > test_acc_A:
        chosen_model = model_B
        chosen_name = "Model B (deep) [chosen: higher accuracy]"
    else:
        chosen_model = model_A
        chosen_name = "Model A (shallow) [chosen: higher accuracy]"

    print("\nUsing model for Next-Best Option analysis:", chosen_name)

    # ------------------------
    # Part 4: Next-Best Option Error Analysis
    # ------------------------
    print("\nRunning Next-Best Option analysis on test set...")
    probs_chosen = chosen_model.predict(x_test, batch_size=256, verbose=0)
    y_pred_chosen = np.argmax(probs_chosen, axis=1)

    nb_stats = next_best_option_analysis(probs_chosen, y_test)

    print("\n" + "=" * 60)
    print("Next-Best Option Error Analysis")
    print("=" * 60)
    print(f"Total test samples: {nb_stats['num_samples']}")
    print(f"Total misclassifications: {nb_stats['num_misclassified']}")
    print(f"Number of 'Close Call' errors (2nd choice was correct): {nb_stats['num_close_calls']}")
    print(f"'Next-Best Option' Percentage: {nb_stats['overall_percentage']:.2f}%")

    summary_sentence = (
        f"For {chosen_name}, the Next-Best Option percentage was "
        f"{nb_stats['overall_percentage']:.2f}%, which suggests that even when "
        f"the model misclassifies an image, it often assigns substantial probability "
        f"to the correct digit, indicating that it is learning useful features."
    )
    print("\nSuggested summary sentence for your report:\n")
    print(summary_sentence)

    # ------------------------------------------------------------
    # Optional extended project section
    # ------------------------------------------------------------
    if ENABLE_DEEPER_ANALYSIS:
        print("\n" + "=" * 60)
        print("Extended Analysis Section (Optional Project)")
        print("=" * 60)

        _ = print_confusion_and_report(y_test, y_pred_chosen, title=chosen_model.name)

        print("\nPer-class 'Next-Best Option' percentages (only among misclassified examples):")
        per_class_mis = nb_stats["per_class_misclassified"]
        per_class_pct = nb_stats["per_class_percentage"]
        for digit in range(10):
            print(
                f"Digit {digit}: misclassified {per_class_mis[digit]:4d} times, "
                f"2nd-best was correct in {per_class_pct[digit]:5.1f}% of those cases"
            )

        top2_acc = compute_top_k_accuracy(probs_chosen, y_test, k=2)
        top3_acc = compute_top_k_accuracy(probs_chosen, y_test, k=3)
        print("\nTop-k accuracy on test set:")
        print(f"Top-1 accuracy (standard): {np.mean(y_pred_chosen == y_test):.4f}")
        print(f"Top-2 accuracy:           {top2_acc:.4f}")
        print(f"Top-3 accuracy:           {top3_acc:.4f}")

        if ENABLE_PLOTTING:
            print("\nShowing example 'close call' misclassifications...")
            show_example_misclassifications(
                x_test,
                y_test,
                probs_chosen,
                num_examples=10,
                close_call_only=True,
            )

        if ENABLE_CNN_EXPERIMENT:
            print("\n" + "=" * 60)
            print("Training optional CNN model (Model C) for comparison")
            print("=" * 60)
            model_C = build_cnn_model()
            model_C.summary()
            train_model(model_C, x_train, y_train, epochs=EPOCHS_CNN)
            test_loss_C, test_acc_C = model_C.evaluate(x_test, y_test, verbose=0)
            print(f"\nModel C (CNN) - Test Accuracy: {test_acc_C:.4f}")
            print(
                "Note: This CNN is not required for the assignment, "
                "but it is a stronger baseline and a nice extension for a project write-up."
            )


if __name__ == "__main__":
    main()
