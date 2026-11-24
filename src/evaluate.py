"""Evaluation: metrics, error analysis, calibration, Grad-CAM, and plots."""
import csv
import math
import os

import numpy as np

from src.utils import save_json


def label_smoothing_entropy_floor(eps: float, num_classes: int) -> float | None:
    """Compute theoretical minimum cross-entropy floor H(q) for Keras label smoothing."""
    if not (0.0 < eps < 1.0) or num_classes <= 1:
        return None
    p_other = eps / num_classes
    p_main = (1.0 - eps) + p_other
    return float(-(p_main * math.log(p_main) + (num_classes - 1) * p_other * math.log(p_other)))


def compute_top_k_accuracy(y_true: np.ndarray, y_pred_probs: np.ndarray, k: int) -> float:
    """Compute top-k categorical accuracy with k clamped to [1, num_classes]."""
    y_true_arr = np.asarray(y_true, dtype=int)
    probs = np.asarray(y_pred_probs, dtype=np.float64)
    if len(y_true_arr) == 0:
        return 0.0
    k_eff = max(1, min(int(k), probs.shape[1]))
    top_k_preds = np.argsort(probs, axis=1)[:, -k_eff:]
    hits = np.any(top_k_preds == y_true_arr[:, None], axis=1)
    return float(np.mean(hits))


def compute_calibration(
    y_true: np.ndarray, y_pred_probs: np.ndarray, n_bins: int = 10
) -> dict:
    """Compute Expected Calibration Error (ECE) and confidence statistics."""
    y_true_arr = np.asarray(y_true, dtype=int)
    probs = np.asarray(y_pred_probs, dtype=np.float64)
    if len(y_true_arr) == 0:
        return {
            "ece": 0.0,
            "n_bins": n_bins,
            "mean_conf_all": 0.0,
            "mean_conf_correct": None,
            "mean_conf_wrong": None,
            "bins": [],
        }

    confs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    correct = (preds == y_true_arr).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    n_total = len(y_true_arr)

    for idx in range(n_bins):
        low, high = float(bin_edges[idx]), float(bin_edges[idx + 1])
        if idx == 0:
            mask = (confs >= low) & (confs <= high)
        else:
            mask = (confs > low) & (confs <= high)
        count = int(np.sum(mask))
        if count > 0:
            acc = float(np.mean(correct[mask]))
            avg_conf = float(np.mean(confs[mask]))
            ece += (count / n_total) * abs(acc - avg_conf)
        else:
            acc = None
            avg_conf = None
        bins.append({
            "low": round(low, 4),
            "high": round(high, 4),
            "count": count,
            "accuracy": round(acc, 6) if acc is not None else None,
            "confidence": round(avg_conf, 6) if avg_conf is not None else None,
        })

    correct_mask = correct == 1.0
    wrong_mask = ~correct_mask
    return {
        "ece": float(ece),
        "n_bins": int(n_bins),
        "mean_conf_all": float(np.mean(confs)),
        "mean_conf_correct": float(np.mean(confs[correct_mask])) if np.any(correct_mask) else None,
        "mean_conf_wrong": float(np.mean(confs[wrong_mask])) if np.any(wrong_mask) else None,
        "bins": bins,
    }


def extract_confused_pairs(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    class_names: list[str],
    top_n: int = 20,
) -> list[dict]:
    """Return top off-diagonal (true_class -> pred_class) error pairs."""
    if top_n <= 0 or len(y_true) == 0:
        return []
    from sklearn.metrics import confusion_matrix

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_hat, labels=labels)
    row_totals = cm.sum(axis=1)
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i == j:
                continue
            count = int(cm[i, j])
            if count <= 0:
                continue
            pct = float(100.0 * count / row_totals[i]) if row_totals[i] > 0 else 0.0
            pairs.append({
                "true_class": class_names[i],
                "pred_class": class_names[j],
                "count": count,
                "pct_of_true_class": round(pct, 2),
            })
    pairs.sort(key=lambda r: (-r["count"], -r["pct_of_true_class"], r["true_class"]))
    return pairs[: int(top_n)]


def extract_misclassified(
    paths: list[str],
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: list[str],
    top_k: int = 3,
) -> list[dict]:
    """Return misclassified samples sorted by descending prediction confidence."""
    y_true_arr = np.asarray(y_true, dtype=int)
    probs = np.asarray(y_pred_probs, dtype=np.float64)
    if len(y_true_arr) == 0:
        return []

    preds = np.argmax(probs, axis=1)
    k_eff = max(1, min(int(top_k), len(class_names)))
    records = []
    for idx in range(len(y_true_arr)):
        t_idx = int(y_true_arr[idx])
        p_idx = int(preds[idx])
        if t_idx == p_idx:
            continue
        order = np.argsort(probs[idx])[::-1][:k_eff]
        records.append({
            "index": idx,
            "path": paths[idx],
            "true_index": t_idx,
            "pred_index": p_idx,
            "true_class": class_names[t_idx],
            "pred_class": class_names[p_idx],
            "pred_confidence": float(probs[idx, p_idx]),
            "true_class_prob": float(probs[idx, t_idx]),
            "top_k_preds": [
                {"class": class_names[int(c)], "prob": float(probs[idx, int(c)])}
                for c in order
            ],
        })
    records.sort(key=lambda r: (-r["pred_confidence"], r["index"]))
    return records


def _predict(model, dataset):
    """Return (y_true_int, y_pred_prob) for a batched dataset."""
    y_true, y_pred = [], []
    for images, labels in dataset:
        probs = np.asarray(model(images, training=False), dtype=np.float32)
        y_pred.append(probs)
        y_true.append(np.asarray(labels))
    y_true_arr = np.concatenate(y_true)
    y_pred_arr = np.concatenate(y_pred)
    if y_true_arr.ndim > 1:  # one-hot
        y_true_arr = np.argmax(y_true_arr, axis=-1)
    return y_true_arr.astype(int), y_pred_arr


def _ema(values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Compute exponential moving average for noisy step-level curves."""
    if len(values) == 0:
        return values
    out = np.empty_like(values, dtype=np.float64)
    out[0] = float(values[0])
    for i in range(1, len(values)):
        out[i] = alpha * float(values[i]) + (1.0 - alpha) * out[i - 1]
    return out


def _plot_step_gradient_descent(
    step_csv: str,
    history: dict | None,
    run_dir: str,
    tag: str,
    alpha: float = 0.05,
) -> str | None:
    """Render batch-level loss + EMA + epoch-end val_loss markers from step_history.csv."""
    steps, epochs, losses = [], [], []
    with open(step_csv, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            steps.append(int(row["global_step"]))
            epochs.append(int(row["epoch"]))
            losses.append(float(row["loss"]))
    if not steps:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps_arr = np.asarray(steps, dtype=np.int64)
    epochs_arr = np.asarray(epochs, dtype=np.int64)
    losses_arr = np.asarray(losses, dtype=np.float64)
    ema_arr = _ema(losses_arr, alpha=alpha)

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(steps_arr, losses_arr, color="#5b8db8", alpha=0.35, linewidth=1.0, label="Batch train loss (raw)")
    ax.plot(steps_arr, ema_arr, color="#1f4e79", linewidth=2.0, label="EMA (alpha=%.2f)" % alpha)

    if history and history.get("val_loss"):
        epoch_end_steps = []
        epoch_vals = []
        val_losses = history["val_loss"]
        for ep_idx, v_loss in enumerate(val_losses, start=1):
            mask = epochs_arr == ep_idx
            if np.any(mask):
                epoch_end_steps.append(int(steps_arr[mask][-1]))
                epoch_vals.append(float(v_loss))
        if epoch_end_steps:
            ax.plot(epoch_end_steps, epoch_vals, "o--", color="#d95f02", markersize=4.5,
                    linewidth=1.5, label="Epoch-end val_loss")

    ax.set_title("Mini-Batch Gradient Descent Dynamics (%s)" % tag)
    ax.set_xlabel("Global Step (mini-batch)")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5)
    plt.tight_layout()
    path = os.path.join(run_dir, "%s_step_gradient_descent.png" % tag)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plots(y_true, y_pred, class_names, history, cfg, run_dir, tag):
    """Confusion matrix + loss/accuracy (+LR/floor) + optional step-level GD curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    ev = cfg["eval"]
    out = {}

    if ev["confusion_matrix"]:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(
            y_true, y_pred.argmax(axis=1), labels=list(range(len(class_names)))
        )
        side = max(8, len(class_names) * 0.34)
        fig = plt.figure(figsize=(side, side * 0.85))
        sns.heatmap(
            cm, annot=len(class_names) <= 20, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            cbar=True, annot_kws={"size": 7},
        )
        plt.xlabel("Prediksi")
        plt.ylabel("Aktual")
        plt.title("Confusion Matrix (%s)" % tag)
        plt.xticks(rotation=90, fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout()
        path = os.path.join(run_dir, "%s_confusion_matrix.png" % tag)
        plt.savefig(path, dpi=150)
        plt.close(fig)
        out["confusion_matrix"] = path

    if history:
        epochs = list(range(1, len(history.get("loss", [])) + 1))
        lr_key = "learning_rate" if "learning_rate" in history else ("lr" if "lr" in history else None)
        n_panels = 3 if lr_key else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(5.4 * n_panels, 4.4))

        ax_loss = axes[0]
        ax_loss.plot(epochs, history.get("loss", []), label="train")
        if "val_loss" in history and history["val_loss"]:
            val_loss = history["val_loss"]
            ax_loss.plot(epochs, val_loss, label="val")
            best_ep = int(np.argmin(val_loss)) + 1
            ax_loss.scatter([best_ep], [min(val_loss)], color="#d95f02", zorder=5,
                            label="best val (ep %d)" % best_ep)
        if cfg["train"]["loss"] == "categorical_crossentropy":
            floor = label_smoothing_entropy_floor(
                float(cfg["train"]["label_smoothing"]), len(class_names)
            )
            if floor is not None:
                ax_loss.axhline(floor, color="#666666", linestyle="--", linewidth=1.1,
                                label="LS floor (%.4f)" % floor)
        ax_loss.set_title("Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.grid(alpha=0.3)
        ax_loss.legend(fontsize=8)

        ax_acc = axes[1]
        ax_acc.plot(epochs, history.get("accuracy", []), label="train")
        if "val_accuracy" in history and history["val_accuracy"]:
            val_acc = history["val_accuracy"]
            ax_acc.plot(epochs, val_acc, label="val")
            best_ep_a = int(np.argmax(val_acc)) + 1
            ax_acc.scatter([best_ep_a], [max(val_acc)], color="#1b9e77", zorder=5,
                           label="best val (ep %d)" % best_ep_a)
        ax_acc.set_title("Accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.grid(alpha=0.3)
        ax_acc.legend(fontsize=8)

        if lr_key:
            ax_lr = axes[2]
            ax_lr.plot(epochs, history[lr_key], color="#7570b3", marker=".")
            ax_lr.set_yscale("log")
            ax_lr.set_title("Learning Rate")
            ax_lr.set_xlabel("Epoch")
            ax_lr.grid(alpha=0.3)

        plt.tight_layout()
        path = os.path.join(run_dir, "%s_loss_accuracy.png" % tag)
        plt.savefig(path, dpi=150)
        plt.close(fig)
        out["loss_accuracy"] = path

    step_csv = os.path.join(run_dir, "step_history.csv")
    if os.path.isfile(step_csv):
        step_plot = _plot_step_gradient_descent(step_csv, history, run_dir, tag)
        if step_plot:
            out["step_gradient_descent"] = step_plot

    return out


def _sample_grid(model, paths, true_labels, class_names, count, cfg, run_dir, tag):
    """Grid of evenly spaced samples with true vs predicted label."""
    if count <= 0 or not paths:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.data import preprocess_image

    indices = np.linspace(0, len(paths) - 1, min(count, len(paths))).astype(int)
    displays, batches = [], []
    for idx in indices:
        display_rgb, batch = preprocess_image(
            paths[idx], cfg["data"], cfg["model"]["arch"]
        )
        displays.append(display_rgb)
        batches.append(batch)

    batch_all = np.concatenate(batches, axis=0)
    all_probs = np.asarray(model(batch_all, training=False), dtype=np.float32)

    cols = 4
    rows = int(np.ceil(len(indices) / cols))
    fig = plt.figure(figsize=(4 * cols, 3.4 * rows))
    for slot, (idx, display_rgb, probs) in enumerate(
        zip(indices, displays, all_probs), start=1
    ):
        pred = int(np.argmax(probs))
        plt.subplot(rows, cols, slot)
        plt.imshow(display_rgb)
        color = "green" if pred == true_labels[idx] else "red"
        plt.title(
            "%s\npred: %s (%.2f)" % (
                class_names[true_labels[idx]], class_names[pred], float(probs[pred])
            ),
            fontsize=8, color=color,
        )
        plt.axis("off")
    plt.tight_layout()
    path = os.path.join(run_dir, "%s_sample_predictions.png" % tag)
    plt.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _calibration_plot(calibration: dict, y_true: np.ndarray, y_pred_probs: np.ndarray,
                      run_dir: str, tag: str) -> str:
    """Plot reliability diagram + confidence histogram (correct vs misclassified)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    confs = np.max(y_pred_probs, axis=1)
    preds = np.argmax(y_pred_probs, axis=1)
    correct_mask = preds == y_true

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    centers = [0.5 * (b["low"] + b["high"]) for b in calibration["bins"] if b["count"] > 0]
    accs = [b["accuracy"] for b in calibration["bins"] if b["count"] > 0]
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    if centers:
        axes[0].plot(centers, accs, marker="o", color="#2f7d5d", label="Model")
    axes[0].set_title("Reliability Diagram (%s, ECE=%.4f)" % (tag, calibration["ece"]))
    axes[0].set_xlabel("Mean Confidence")
    axes[0].set_ylabel("Empirical Accuracy")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    bins_hist = np.linspace(0.0, 1.0, 15)
    if np.any(correct_mask):
        axes[1].hist(confs[correct_mask], bins=bins_hist, alpha=0.65,
                     color="#2f7d5d", label="Benar (%d)" % int(np.sum(correct_mask)))
    if np.any(~correct_mask):
        axes[1].hist(confs[~correct_mask], bins=bins_hist, alpha=0.75,
                     color="#c44e52", label="Salah (%d)" % int(np.sum(~correct_mask)))
    axes[1].set_title("Distribusi Confidence Prediksi (%s)" % tag)
    axes[1].set_xlabel("Max Softmax Probability")
    axes[1].set_ylabel("Jumlah Sampel")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(run_dir, "%s_confidence_calibration.png" % tag)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _misclassified_grid(misclassified: list[dict], count: int, cfg: dict,
                        run_dir: str, tag: str) -> str | None:
    """Plot grid of highest-confidence misclassified samples."""
    if count <= 0 or not misclassified:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.data import preprocess_image

    picked = misclassified[: int(count)]
    cols = min(4, len(picked))
    rows = int(np.ceil(len(picked) / cols))
    fig = plt.figure(figsize=(4.0 * cols, 3.6 * rows))
    for slot, item in enumerate(picked, start=1):
        display_rgb, _ = preprocess_image(
            item["path"], cfg["data"], cfg["model"]["arch"]
        )
        plt.subplot(rows, cols, slot)
        plt.imshow(display_rgb)
        plt.title(
            "T: %s\nP: %s (%.2f)" % (
                item["true_class"], item["pred_class"], item["pred_confidence"]
            ),
            fontsize=7.5, color="#b02a37",
        )
        plt.axis("off")
    plt.suptitle("High-Confidence Misclassified Samples (%s)" % tag, y=1.01, fontsize=11)
    plt.tight_layout()
    path = os.path.join(run_dir, "%s_misclassified_grid.png" % tag)
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _gradcam_grid(model, paths: list[str], y_true: np.ndarray, y_pred_probs: np.ndarray,
                  misclassified: list[dict], class_names: list[str],
                  count: int, cfg: dict, run_dir: str, tag: str) -> str | None:
    """Plot Original | Grad-CAM Overlay pairs prioritizing misclassified samples."""
    if count <= 0 or not paths:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.gradcam import explain_image

    selected = []
    seen = set()
    for item in misclassified:
        idx = item["index"]
        if idx not in seen:
            selected.append((idx, False))
            seen.add(idx)
        if len(selected) >= count:
            break

    if len(selected) < count:
        preds = np.argmax(y_pred_probs, axis=1)
        correct_indices = [i for i in range(len(paths)) if preds[i] == y_true[i] and i not in seen]
        if correct_indices:
            need = min(count - len(selected), len(correct_indices))
            pos = np.linspace(0, len(correct_indices) - 1, need).astype(int)
            for p in pos:
                selected.append((correct_indices[int(p)], True))

    if not selected:
        return None

    pairs_per_row = 2 if len(selected) > 1 else 1
    cols = pairs_per_row * 2
    rows = int(np.ceil(len(selected) / pairs_per_row))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.6 * rows))
    axes_flat = np.atleast_1d(axes).ravel()

    alpha = cfg["eval"].get("gradcam_alpha", 0.4)
    for ax in axes_flat:
        ax.axis("off")

    for pair_idx, (sample_idx, is_correct) in enumerate(selected):
        info = explain_image(
            model, paths[sample_idx], cfg, class_names,
            top_k=cfg["eval"]["top_k"], alpha=alpha,
        )
        ax_orig = axes_flat[pair_idx * 2]
        ax_cam = axes_flat[pair_idx * 2 + 1]

        ax_orig.imshow(info["original_rgb"])
        ax_orig.set_title("Orig: %s" % class_names[int(y_true[sample_idx])], fontsize=7.5)
        ax_orig.axis("off")

        ax_cam.imshow(info["overlay_rgb"])
        color = "#1f6f43" if is_correct else "#b02a37"
        ax_cam.set_title(
            "CAM: %s (%.2f)" % (info["pred_class"], info["pred_prob"]),
            fontsize=7.5, color=color,
        )
        ax_cam.axis("off")

    plt.suptitle("Grad-CAM Audit (%s)" % tag, y=1.01, fontsize=11)
    plt.tight_layout()
    path = os.path.join(run_dir, "%s_gradcam_grid.png" % tag)
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def evaluate(model, cfg, bundle=None, run_dir=None, history=None, tag=None, quiet: bool = False):
    """Evaluate on cfg['eval']['eval_split']. Writes metrics, error analysis, and plots."""
    from sklearn.metrics import (
        accuracy_score, classification_report, f1_score, precision_score, recall_score,
    )

    from src.data import build_dataset, prepare_datasets

    ev = cfg["eval"]
    split = ev["eval_split"]
    tag = tag or split
    bundle = bundle or prepare_datasets(cfg, verbose=not quiet)
    run_dir = run_dir or os.path.join(cfg["runtime"]["runs_dir"], "eval")
    os.makedirs(run_dir, exist_ok=True)

    split_data = bundle[split]
    runtime_ctx = dict(cfg["runtime"])
    runtime_ctx["_num_classes"] = len(bundle["class_names"])
    dataset = build_dataset(
        split_data["paths"], split_data["labels"], cfg["data"], runtime_ctx,
        cfg["model"], training=False, batch_size=ev["eval_batch_size"],
        one_hot=cfg["train"]["loss"] != "sparse_categorical_crossentropy",
        split_name=split,
    )

    y_true, y_pred = _predict(model, dataset)
    y_hat = y_pred.argmax(axis=1)
    class_names = bundle["class_names"]
    labels = list(range(len(class_names)))
    k_eff = max(1, min(int(ev["top_k"]), len(class_names)))

    metrics = {
        "split": split,
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "top_k": k_eff,
        "top_k_accuracy": compute_top_k_accuracy(y_true, y_pred, k_eff),
        "precision_macro": float(precision_score(
            y_true, y_hat, average="macro", zero_division=0
        )),
        "recall_macro": float(recall_score(
            y_true, y_hat, average="macro", zero_division=0
        )),
        "f1_macro": float(f1_score(y_true, y_hat, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_hat, average="weighted", zero_division=0)),
    }

    print("\n=== Evaluasi (%s, %d gambar) ===" % (split, len(y_true)))
    print("  accuracy        : %.4f" % metrics["accuracy"])
    print("  top-%d accuracy  : %.4f" % (k_eff, metrics["top_k_accuracy"]))
    print("  precision macro : %.4f" % metrics["precision_macro"])
    print("  recall macro    : %.4f" % metrics["recall_macro"])
    print("  f1 macro        : %.4f" % metrics["f1_macro"])
    print("  f1 weighted     : %.4f" % metrics["f1_weighted"])

    if ev["classification_report"]:
        report = classification_report(
            y_true, y_hat, labels=labels, target_names=class_names, zero_division=0
        )
        if not quiet:
            print("\n" + report)
        with open(os.path.join(run_dir, "%s_classification_report.txt" % tag),
                  "w", encoding="utf-8") as handle:
            handle.write(report)

    if ev["per_class_metrics"]:
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_hat, labels=labels, zero_division=0
        )
        per_class = [
            {
                "class": name,
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        ]
        metrics["per_class"] = per_class
        save_json(os.path.join(run_dir, "%s_per_class.json" % tag), per_class)
        with open(os.path.join(run_dir, "%s_per_class.csv" % tag),
                  "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["class", "precision", "recall", "f1", "support"]
            )
            writer.writeheader()
            for row in per_class:
                writer.writerow({
                    "class": row["class"],
                    "precision": "%.4f" % row["precision"],
                    "recall": "%.4f" % row["recall"],
                    "f1": "%.4f" % row["f1"],
                    "support": row["support"],
                })

    plots = {}
    if ev["save_plots"]:
        plots = _plots(y_true, y_pred, class_names, history, cfg, run_dir, tag)
        grid = _sample_grid(
            model, split_data["paths"], split_data["labels"],
            class_names, ev["sample_predictions"], cfg, run_dir, tag,
        )
        if grid:
            plots["sample_predictions"] = grid

    misclassified = []
    if ev.get("error_analysis", True):
        calibration = compute_calibration(y_true, y_pred, n_bins=10)
        metrics["calibration"] = calibration
        print("  ECE (10 bins)   : %.4f" % calibration["ece"])

        confused_pairs = extract_confused_pairs(
            y_true, y_hat, class_names, top_n=ev.get("top_confused_pairs", 20)
        )
        metrics["top_confused_pairs"] = confused_pairs
        save_json(os.path.join(run_dir, "%s_confused_pairs.json" % tag), confused_pairs)
        with open(os.path.join(run_dir, "%s_confused_pairs.csv" % tag),
                  "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["true_class", "pred_class", "count", "pct_of_true_class"],
            )
            writer.writeheader()
            writer.writerows(confused_pairs)

        if confused_pairs and not quiet:
            print("\nTop confused pairs:")
            for pair in confused_pairs[:10]:
                print("  %s -> %s : %d (%.2f%%)" % (
                    pair["true_class"], pair["pred_class"],
                    pair["count"], pair["pct_of_true_class"],
                ))

        misclassified = extract_misclassified(
            split_data["paths"], y_true, y_pred, class_names, top_k=k_eff
        )
        metrics["num_misclassified"] = len(misclassified)
        with open(os.path.join(run_dir, "%s_misclassified.csv" % tag),
                  "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "path", "true_class", "pred_class",
                    "pred_confidence", "true_class_prob",
                ],
            )
            writer.writeheader()
            for item in misclassified:
                writer.writerow({
                    "path": item["path"],
                    "true_class": item["true_class"],
                    "pred_class": item["pred_class"],
                    "pred_confidence": "%.6f" % item["pred_confidence"],
                    "true_class_prob": "%.6f" % item["true_class_prob"],
                })

        if ev["save_plots"]:
            plots["confidence_calibration"] = _calibration_plot(
                calibration, y_true, y_pred, run_dir, tag
            )
            mis_grid = _misclassified_grid(
                misclassified, ev.get("misclassified_samples", 16), cfg, run_dir, tag
            )
            if mis_grid:
                plots["misclassified_grid"] = mis_grid

    if ev.get("gradcam", True) and ev["save_plots"] and ev.get("gradcam_samples", 8) > 0:
        if not misclassified:
            misclassified = extract_misclassified(
                split_data["paths"], y_true, y_pred, class_names, top_k=k_eff
            )
        cam_grid = _gradcam_grid(
            model, split_data["paths"], y_true, y_pred, misclassified,
            class_names, ev.get("gradcam_samples", 8), cfg, run_dir, tag,
        )
        if cam_grid:
            plots["gradcam"] = cam_grid

    if plots:
        metrics["plots"] = plots

    save_json(os.path.join(run_dir, "%s_metrics.json" % tag), metrics)
    return metrics
