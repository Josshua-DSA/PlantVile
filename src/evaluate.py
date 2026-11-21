"""Evaluation: metrics, confusion matrix, training curves, sample predictions."""
import os

import keras
import numpy as np

from src.data import build_dataset, prepare_datasets
from src.utils import save_json


def _predict(model, dataset):
    """Return (y_true_int, y_pred_prob) for a batched dataset."""
    y_true, y_pred = [], []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        y_pred.append(probs)
        y_true.append(np.asarray(labels))
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    if y_true.ndim > 1:  # one-hot
        y_true = np.argmax(y_true, axis=-1)
    return y_true.astype(int), y_pred


def _plots(y_true, y_pred, class_names, history, cfg, run_dir, tag):
    """Confusion matrix + loss/accuracy curves + sample prediction grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    ev = cfg["eval"]
    out = {}

    if ev["confusion_matrix"]:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred.argmax(axis=1),
                              labels=list(range(len(class_names))))
        side = max(8, len(class_names) * 0.34)
        plt.figure(figsize=(side, side * 0.85))
        sns.heatmap(cm, annot=len(class_names) <= 20, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    cbar=True, annot_kws={"size": 7})
        plt.xlabel("Prediksi")
        plt.ylabel("Aktual")
        plt.title("Confusion Matrix (%s)" % tag)
        plt.xticks(rotation=90, fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout()
        path = os.path.join(run_dir, "%s_confusion_matrix.png" % tag)
        plt.savefig(path, dpi=150)
        plt.close()
        out["confusion_matrix"] = path

    if history:
        epochs = range(1, len(history.get("loss", [])) + 1)
        plt.figure(figsize=(12, 4.5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history.get("loss", []), label="train")
        if "val_loss" in history:
            plt.plot(epochs, history["val_loss"], label="val")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history.get("accuracy", []), label="train")
        if "val_accuracy" in history:
            plt.plot(epochs, history["val_accuracy"], label="val")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.legend()
        plt.tight_layout()
        path = os.path.join(run_dir, "%s_loss_accuracy.png" % tag)
        plt.savefig(path, dpi=150)
        plt.close()
        out["loss_accuracy"] = path

    return out


def _sample_grid(model, paths, true_labels, class_names, count, cfg, run_dir, tag):
    """Grid of images with true vs predicted label."""
    if count <= 0:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = cfg["data"]
    indices = np.linspace(0, len(paths) - 1, min(count, len(paths))).astype(int)
    cols = 4
    rows = int(np.ceil(len(indices) / cols))
    plt.figure(figsize=(4 * cols, 3.4 * rows))
    for slot, idx in enumerate(indices, start=1):
        image = keras.utils.load_img(paths[idx],
                                     target_size=(data["img_height"], data["img_width"]))
        array = keras.utils.img_to_array(image)
        batch = np.expand_dims(array, 0)
        if data["normalize"] == "imagenet" and cfg["model"]["arch"] != "simple_cnn":
            from keras.applications import (
                efficientnet, mobilenet_v2, resnet, vgg16,
            )
            preprocess = {
                "vgg16": vgg16.preprocess_input,
                "resnet50": resnet.preprocess_input,
                "mobilenetv2": mobilenet_v2.preprocess_input,
                "efficientnetb0": efficientnet.preprocess_input,
            }[cfg["model"]["arch"]]
            batch = preprocess(batch)
        else:
            batch = batch / 255.0
        probs = model.predict(batch, verbose=0)[0]
        pred = int(np.argmax(probs))
        plt.subplot(rows, cols, slot)
        plt.imshow(array.astype("uint8"))
        color = "green" if pred == true_labels[idx] else "red"
        plt.title("%s\npred: %s (%.2f)" % (
            class_names[true_labels[idx]], class_names[pred], float(probs[pred])),
            fontsize=8, color=color)
        plt.axis("off")
    plt.tight_layout()
    path = os.path.join(run_dir, "%s_sample_predictions.png" % tag)
    plt.savefig(path, dpi=140)
    plt.close()
    return path


def evaluate(model, cfg, bundle=None, run_dir=None, history=None, tag=None):
    """Evaluate on cfg['eval']['eval_split']. Writes metrics + plots into run_dir."""
    from sklearn.metrics import (
        accuracy_score, classification_report, f1_score, precision_score, recall_score,
    )

    ev = cfg["eval"]
    split = ev["eval_split"]
    tag = tag or split
    bundle = bundle or prepare_datasets(cfg)
    run_dir = run_dir or os.path.join(cfg["runtime"]["runs_dir"], "eval")
    os.makedirs(run_dir, exist_ok=True)

    split_data = bundle[split]
    dataset = build_dataset(
        split_data["paths"], split_data["labels"], cfg["data"], cfg["runtime"],
        cfg["model"], training=False, batch_size=ev["eval_batch_size"],
        one_hot=cfg["train"]["loss"] != "sparse_categorical_crossentropy",
    )

    y_true, y_pred = _predict(model, dataset)
    y_hat = y_pred.argmax(axis=1)
    class_names = bundle["class_names"]
    labels = list(range(len(class_names)))

    metrics = {
        "split": split,
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "precision_macro": float(precision_score(y_true, y_hat, average="macro",
                                                 zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_hat, average="macro",
                                           zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_hat, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_hat, average="weighted", zero_division=0)),
    }

    print("\n=== Evaluasi (%s, %d gambar) ===" % (split, len(y_true)))
    print("  accuracy        : %.4f" % metrics["accuracy"])
    print("  precision macro : %.4f" % metrics["precision_macro"])
    print("  recall macro    : %.4f" % metrics["recall_macro"])
    print("  f1 macro        : %.4f" % metrics["f1_macro"])
    print("  f1 weighted     : %.4f" % metrics["f1_weighted"])

    if ev["classification_report"]:
        report = classification_report(y_true, y_hat, labels=labels,
                                       target_names=class_names, zero_division=0)
        print("\n" + report)
        with open(os.path.join(run_dir, "%s_classification_report.txt" % tag),
                  "w", encoding="utf-8") as handle:
            handle.write(report)

    if ev["per_class_metrics"]:
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_hat, labels=labels, zero_division=0)
        per_class = [
            {"class": name, "precision": float(precision[i]), "recall": float(recall[i]),
             "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(class_names)
        ]
        metrics["per_class"] = per_class
        save_json(os.path.join(run_dir, "%s_per_class.json" % tag), per_class)
        with open(os.path.join(run_dir, "%s_per_class.csv" % tag), "w",
                  encoding="utf-8") as handle:
            handle.write("class,precision,recall,f1,support\n")
            for row in per_class:
                handle.write("%s,%.4f,%.4f,%.4f,%d\n" % (
                    row["class"], row["precision"], row["recall"], row["f1"],
                    row["support"]))

    if ev["save_plots"]:
        metrics["plots"] = _plots(y_true, y_pred, class_names, history, cfg, run_dir, tag)
        grid = _sample_grid(model, split_data["paths"], split_data["labels"],
                            class_names, ev["sample_predictions"], cfg, run_dir, tag)
        if grid:
            metrics["plots"]["sample_predictions"] = grid

    save_json(os.path.join(run_dir, "%s_metrics.json" % tag), metrics)
    return metrics
