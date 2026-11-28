"""Export Pure FP32 Keras + TFLite (FP32, FP16, INT8) models, benchmark mobile/web metrics, and build UI handoff bundle.

Run from repository root:
    python -m scripts.export_tflite --runs-dir results/runs --data-root data/PlantVillage --eval-split val --build-handoff
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from src.data import build_dataset, prepare_datasets, preprocess_image  # noqa: E402
from src.evaluate import compute_calibration  # noqa: E402
from src.utils import configure_device, save_json, set_seed  # noqa: E402

LOGGER = logging.getLogger("export_tflite")

DEFAULT_STAGE2_RUNS = [
    "m2_mobilenetv2_ft",
    "m3b_effnetb0_ft",
    "m4b_mobilenetv3l_ft",
]


def rewrite_dtype_policy(obj):
    """Recursively rewrite mixed_float16 DTypePolicy entries in Keras model config to float32."""
    if isinstance(obj, dict):
        if obj.get("class_name") == "DTypePolicy" and obj.get("config", {}).get("name") == "mixed_float16":
            out = copy.deepcopy(obj)
            out["config"]["name"] = "float32"
            return out
        return {k: rewrite_dtype_policy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rewrite_dtype_policy(item) for item in obj]
    if obj == "mixed_float16":
        return "float32"
    return obj


def finalize_all_states(model: keras.Model) -> None:
    """Refresh cached Normalization mean/variance after set_weights() across nested layers."""
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            finalize_all_states(layer)
        elif hasattr(layer, "finalize_state"):
            layer.finalize_state()


def export_pure_fp32_keras(
    run_id: str,
    ckpt_path: Path,
    cfg: dict,
    out_path: Path,
    check_paths: list[str],
    check_labels: list[int],
) -> tuple[keras.Model, dict]:
    """Convert mixed-precision checkpoint into optimizer-free pure float32 .keras and verify parity."""
    keras.mixed_precision.set_global_policy("float32")
    raw_model = keras.models.load_model(str(ckpt_path), compile=False)

    fp32_cfg = rewrite_dtype_policy(raw_model.get_config())
    fp32_build = keras.Model.from_config(fp32_cfg)
    raw_weights = raw_model.get_weights()
    fp32_weights = fp32_build.get_weights()
    if len(raw_weights) != len(fp32_weights):
        raise RuntimeError(f"Weight count mismatch for {run_id}: {len(raw_weights)} vs {len(fp32_weights)}")
    for idx, (rw, fw) in enumerate(zip(raw_weights, fp32_weights)):
        if rw.shape != fw.shape:
            raise RuntimeError(f"Weight shape mismatch at {idx} for {run_id}: {rw.shape} vs {fw.shape}")

    fp32_build.set_weights([np.asarray(w, dtype=np.float32) for w in raw_weights])
    finalize_all_states(fp32_build)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fp32_build.save(str(out_path), include_optimizer=False)
    fp32_model = keras.models.load_model(str(out_path), compile=False)

    seen_cls: set[int] = set()
    sample_paths: list[str] = []
    for p, lbl in zip(check_paths, check_labels):
        if lbl not in seen_cls:
            seen_cls.add(lbl)
            sample_paths.append(p)
    if sample_paths:
        batches = [preprocess_image(p, cfg["data"], cfg["model"]["arch"])[1] for p in sample_paths]
        x_check = np.concatenate(batches, axis=0)
        raw_p = np.asarray(raw_model(x_check, training=False), dtype=np.float64)
        fp32_p = np.asarray(fp32_model(x_check, training=False), dtype=np.float64)
        diffs = np.abs(raw_p - fp32_p)
        mean_diff = float(np.mean(diffs))
        max_diff = float(np.max(diffs))
        agree = float(np.mean(np.argmax(raw_p, axis=1) == np.argmax(fp32_p, axis=1)))
        if mean_diff >= 0.01 or max_diff >= 0.05 or agree < 0.99:
            raise RuntimeError(
                f"FP32 parity failed for {run_id}: mean_diff={mean_diff:.6f}, "
                f"max_diff={max_diff:.6f}, agree={agree:.4f}"
            )
    else:
        mean_diff, max_diff, agree = 0.0, 0.0, 1.0

    info = {
        "raw_mb": round(ckpt_path.stat().st_size / (1024 * 1024), 3),
        "fp32_keras_mb": round(out_path.stat().st_size / (1024 * 1024), 3),
        "parity_mean_abs_diff": round(mean_diff, 6),
        "parity_max_abs_diff": round(max_diff, 6),
        "parity_argmax_agree": round(agree, 6),
    }
    return fp32_model, info


def make_representative_dataset_fn(
    train_paths: list[str],
    train_labels: list[int],
    cfg: dict,
    per_class: int = 20,
    seed: int = 42,
):
    """Build deterministic stratified representative dataset strictly from the train split."""
    rng = np.random.default_rng(seed)
    by_cls: dict[int, list[str]] = {}
    for p, lbl in zip(train_paths, train_labels):
        by_cls.setdefault(int(lbl), []).append(p)

    calib_batches: list[np.ndarray] = []
    for cls_idx in sorted(by_cls.keys()):
        candidates = sorted(by_cls[cls_idx])
        if len(candidates) > per_class:
            chosen = rng.choice(len(candidates), size=per_class, replace=False)
            picked = [candidates[int(i)] for i in sorted(chosen)]
        else:
            picked = candidates
        for p in picked:
            _, batch = preprocess_image(p, cfg["data"], cfg["model"]["arch"])
            calib_batches.append(batch.astype(np.float32))

    def representative_dataset():
        for batch in calib_batches:
            yield [batch]

    return representative_dataset, len(calib_batches)


def convert_keras_to_tflite(
    fp32_model: keras.Model,
    quant: str,
    rep_dataset_fn=None,
) -> tuple[bytes, dict]:
    """Convert pure float32 Keras model to TFLite bytes ('fp32', 'fp16', or 'int8')."""
    meta: dict = {"quant": quant, "ops_fallback": False}

    def _build_converter():
        return tf.lite.TFLiteConverter.from_keras_model(fp32_model)

    if quant == "fp32":
        converter = _build_converter()
        return converter.convert(), meta

    if quant == "fp16":
        converter = _build_converter()
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        return converter.convert(), meta

    if quant == "int8":
        if rep_dataset_fn is None:
            raise ValueError("INT8 quantization requires rep_dataset_fn")
        converter = _build_converter()
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_dataset_fn
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        try:
            return converter.convert(), meta
        except Exception as exc:
            LOGGER.warning("Strict INT8 builtin conversion fell back to mixed ops: %s", exc)
            converter = _build_converter()
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = rep_dataset_fn
            converter.inference_input_type = tf.float32
            converter.inference_output_type = tf.float32
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS,
            ]
            meta["ops_fallback"] = True
            return converter.convert(), meta

    raise ValueError(f"Unknown quantization mode: {quant}")


def _predict_keras(fp32_model: keras.Model, dataset) -> tuple[np.ndarray, np.ndarray]:
    @tf.function(jit_compile=False)
    def forward(bx):
        return fp32_model(bx, training=False)

    y_true, y_pred = [], []
    for bx, by in dataset:
        probs = forward(bx).numpy().astype(np.float32)
        y_pred.append(probs)
        y_true.append(np.asarray(by, dtype=int))
    return np.concatenate(y_true), np.concatenate(y_pred)


def _load_tflite_interpreter(tflite_path: Path, num_threads: int = 4):
    """Allocate a TFLite Interpreter, falling back without default delegates if XNNPACK fails."""
    try:
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=num_threads)
        interpreter.allocate_tensors()
        return interpreter
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "xnnpack" not in msg and "delegate" not in msg:
            raise
        LOGGER.warning(
            "Default XNNPACK delegate failed for %s (%s); falling back to BUILTIN_WITHOUT_DEFAULT_DELEGATES.",
            tflite_path.name,
            exc,
        )
        interpreter = tf.lite.Interpreter(
            model_path=str(tflite_path),
            num_threads=num_threads,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
        interpreter.allocate_tensors()
        return interpreter


def _predict_tflite(tflite_path: Path, dataset, num_threads: int = 4) -> tuple[np.ndarray, np.ndarray]:
    interpreter = _load_tflite_interpreter(tflite_path, num_threads=num_threads)
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    in_idx, out_idx = in_det["index"], out_det["index"]

    y_true, y_pred = [], []
    for bx, by in dataset:
        bx_np = np.asarray(bx, dtype=np.float32)
        batch_probs = np.empty((bx_np.shape[0], out_det["shape"][-1]), dtype=np.float32)
        for i in range(bx_np.shape[0]):
            interpreter.set_tensor(in_idx, bx_np[i : i + 1])
            interpreter.invoke()
            batch_probs[i] = interpreter.get_tensor(out_idx)[0]
        y_pred.append(batch_probs)
        y_true.append(np.asarray(by, dtype=int))
    return np.concatenate(y_true), np.concatenate(y_pred)


def _benchmark_keras_latency(
    fp32_model: keras.Model, sample_batch: np.ndarray, warmup: int, runs: int
) -> dict[str, float]:
    x_tf = tf.convert_to_tensor(sample_batch, dtype=tf.float32)

    @tf.function(jit_compile=False)
    def infer(x):
        return fp32_model(x, training=False)

    for _ in range(warmup):
        _ = infer(x_tf).numpy()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = infer(x_tf).numpy()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "mean_ms": round(float(np.mean(arr)), 2),
    }


def _benchmark_tflite_latency(
    tflite_path: Path, sample_batch: np.ndarray, num_threads: int, warmup: int, runs: int
) -> dict[str, float]:
    interpreter = _load_tflite_interpreter(tflite_path, num_threads=num_threads)
    in_idx = interpreter.get_input_details()[0]["index"]
    out_idx = interpreter.get_output_details()[0]["index"]
    x = np.asarray(sample_batch, dtype=np.float32)

    for _ in range(warmup):
        interpreter.set_tensor(in_idx, x)
        interpreter.invoke()
        _ = interpreter.get_tensor(out_idx)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(in_idx, x)
        interpreter.invoke()
        _ = interpreter.get_tensor(out_idx)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "mean_ms": round(float(np.mean(arr)), 2),
    }


def bootstrap_ci_metrics(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    num_classes: int,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Compute vectorized 95% bootstrap confidence intervals for accuracy and macro F1."""
    n = len(y_true)
    if n == 0:
        return {"acc_ci95": (0.0, 0.0), "f1_ci95": (0.0, 0.0)}
    rng = np.random.default_rng(seed)
    accs = np.empty(n_boot, dtype=np.float64)
    f1s = np.empty(n_boot, dtype=np.float64)
    pair_codes = y_true * num_classes + y_hat
    correct_flags = (y_true == y_hat).astype(np.float64)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        accs[b] = float(np.mean(correct_flags[idx]))
        cm = np.bincount(pair_codes[idx], minlength=num_classes * num_classes).reshape(
            num_classes, num_classes
        )
        tp = np.diag(cm).astype(np.float64)
        fp = cm.sum(axis=0).astype(np.float64) - tp
        fn = cm.sum(axis=1).astype(np.float64) - tp
        denom = 2.0 * tp + fp + fn
        f1_per_cls = np.divide(2.0 * tp, denom, out=np.zeros_like(denom), where=denom > 0)
        f1s[b] = float(np.mean(f1_per_cls))

    return {
        "acc_ci95": (round(float(np.percentile(accs, 2.5)), 4), round(float(np.percentile(accs, 97.5)), 4)),
        "f1_ci95": (round(float(np.percentile(f1s, 2.5)), 4), round(float(np.percentile(f1s, 97.5)), 4)),
    }


def exact_mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact two-sided McNemar test between two classifiers on identical ground truth."""
    corr_a = pred_a == y_true
    corr_b = pred_b == y_true
    b01 = int(np.sum(corr_a & (~corr_b)))  # A right, B wrong
    b10 = int(np.sum((~corr_a) & corr_b))  # A wrong, B right
    n_disc = b01 + b10
    p_val = float(binomtest(min(b01, b10), n=n_disc, p=0.5, alternative="two-sided").pvalue) if n_disc > 0 else 1.0
    return {"a_right_b_wrong": b01, "a_wrong_b_right": b10, "p_value": p_val}


def _fit_val_temperature(fp32_model: keras.Model, val_ds, seed: int = 42) -> dict:
    """Fit temperature scaling strictly on val split logits (never touches test)."""
    layers_list = fp32_model.layers
    last_layer = layers_list[-1]
    feature_model = keras.Model(fp32_model.inputs, layers_list[-2].output)
    kernel = tf.cast(last_layer.kernel, tf.float32)
    bias = tf.cast(last_layer.bias, tf.float32) if last_layer.use_bias else None

    @tf.function(jit_compile=False)
    def forward_logits(bx):
        feat = tf.cast(feature_model(bx, training=False), tf.float32)
        logits = tf.matmul(feat, kernel)
        if bias is not None:
            logits = logits + bias
        return logits

    y_parts, logit_parts = [], []
    for bx, by in val_ds:
        logit_parts.append(forward_logits(bx).numpy().astype(np.float64))
        y_parts.append(np.asarray(by, dtype=int))
    y_val = np.concatenate(y_parts)
    logits = np.concatenate(logit_parts)

    def nll(log_mat, targets, t):
        s = log_mat / float(t)
        m = np.max(s, axis=1, keepdims=True)
        lse = np.squeeze(m, axis=1) + np.log(np.sum(np.exp(s - m), axis=1))
        return float(np.mean(lse - s[np.arange(len(targets)), targets]))

    def fit_1d(log_mat, targets, lo=0.05, hi=5.0, iters=60):
        phi = (math.sqrt(5.0) - 1.0) / 2.0
        a, b = float(lo), float(hi)
        c, d = b - phi * (b - a), a + phi * (b - a)
        fc, fd = nll(log_mat, targets, c), nll(log_mat, targets, d)
        for _ in range(iters):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - phi * (b - a)
                fc = nll(log_mat, targets, c)
            else:
                a, c, fc = c, d, fd
                d = a + phi * (b - a)
                fd = nll(log_mat, targets, d)
        return float((a + b) / 2.0)

    def softmax_np(x):
        sh = x - np.max(x, axis=1, keepdims=True)
        ex = np.exp(sh)
        return ex / np.sum(ex, axis=1, keepdims=True)

    raw_probs = softmax_np(logits)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_probs = np.zeros_like(raw_probs)
    fold_ts = []
    for tr, ev in skf.split(logits, y_val):
        t_f = fit_1d(logits[tr], y_val[tr])
        fold_ts.append(t_f)
        oof_probs[ev] = softmax_np(logits[ev] / t_f)

    t_full = fit_1d(logits, y_val)
    ece_before = compute_calibration(y_val, raw_probs, n_bins=10)["ece"]
    ece_after = compute_calibration(y_val, oof_probs, n_bins=10)["ece"]
    return {
        "T_full_val": round(float(t_full), 5),
        "oof_T_mean": round(float(np.mean(fold_ts)), 5),
        "oof_T_std": round(float(np.std(fold_ts, ddof=1)), 5),
        "val_ece_before": round(float(ece_before), 6),
        "val_ece_after_oof": round(float(ece_after), 6),
    }


def _evaluate_predictions(
    y_true: np.ndarray,
    probs: np.ndarray,
    num_classes: int,
    seed: int,
) -> tuple[dict, np.ndarray]:
    y_hat = np.argmax(probs, axis=1)
    labels = list(range(num_classes))
    per_cls_rec = recall_score(y_true, y_hat, average=None, labels=labels, zero_division=0)
    cal = compute_calibration(y_true, probs, n_bins=10)
    cis = bootstrap_ci_metrics(y_true, y_hat, num_classes, n_boot=1000, seed=seed)
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_hat)), 6),
        "precision_macro": round(float(precision_score(y_true, y_hat, average="macro", zero_division=0)), 6),
        "recall_macro": round(float(recall_score(y_true, y_hat, average="macro", zero_division=0)), 6),
        "f1_macro": round(float(f1_score(y_true, y_hat, average="macro", zero_division=0)), 6),
        "worst_class_recall": round(float(np.min(per_cls_rec)), 6),
        "ece": round(float(cal["ece"]), 6),
        "acc_ci95": list(cis["acc_ci95"]),
        "f1_ci95": list(cis["f1_ci95"]),
    }
    return metrics, y_hat


def _write_handoff_files(
    handoff_dir: Path,
    class_names: list[str],
    run_artifacts: dict,
    summary_rows: list[dict],
) -> None:
    """Create standalone dual-target UI handoff bundle (Cloud Web + On-Device Mobile)."""
    handoff_dir.mkdir(parents=True, exist_ok=True)
    models_root = handoff_dir / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    (handoff_dir / "labels.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    save_json(str(handoff_dir / "class_names.json"), class_names)

    # Pick recommended models (highest keras FP32 f1_macro, tie-break smallest keras size)
    keras_rows = [r for r in summary_rows if r["format"] == "keras_fp32"]
    keras_rows.sort(key=lambda r: (-r["f1_macro"], r["size_mb"]))
    rec_web = keras_rows[0]["run_id"] if keras_rows else ""

    tflite_rows = [r for r in summary_rows if r["format"] in ("tflite_fp16", "tflite_int8")]
    # Prefer variant with <= 0.005 F1 drop vs keras_fp32 and smallest size
    keras_f1_map = {r["run_id"]: r["f1_macro"] for r in keras_rows}
    eligible_mobile = [
        r for r in tflite_rows
        if (keras_f1_map.get(r["run_id"], 0.0) - r["f1_macro"]) <= 0.005
    ]
    pool_mobile = eligible_mobile or tflite_rows
    pool_mobile.sort(key=lambda r: (r["size_mb"], -r["f1_macro"]))
    rec_mobile_row = pool_mobile[0] if pool_mobile else None

    models_meta = {}
    for run_id, art in run_artifacts.items():
        dst_dir = models_root / run_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(art["keras_fp32_path"], dst_dir / art["keras_fp32_path"].name)
        for q, tpath in art["tflite_paths"].items():
            shutil.copy2(tpath, dst_dir / tpath.name)

        arch = art["arch"]
        prep_rule = (
            "rgb_uint8 -> float32 [0,255] -> (x/127.5)-1.0 (MobileNetV2)"
            if arch == "mobilenetv2"
            else "rgb_uint8 -> float32 [0,255] (internal Rescaling layer inside model)"
        )
        models_meta[run_id] = {
            "arch": arch,
            "input_shape": [1, art["img_height"], art["img_width"], 3],
            "input_dtype": "float32",
            "preprocessing": prep_rule,
            "temperature_T": art["temperature"]["T_full_val"],
            "temperature_oof_mean": art["temperature"]["oof_T_mean"],
            "val_ece_before": art["temperature"]["val_ece_before"],
            "val_ece_after_oof": art["temperature"]["val_ece_after_oof"],
            "files": {
                "keras_fp32": f"models/{run_id}/{art['keras_fp32_path'].name}",
                "tflite_fp32": f"models/{run_id}/{art['tflite_paths']['fp32'].name}",
                "tflite_fp16": f"models/{run_id}/{art['tflite_paths']['fp16'].name}",
                "tflite_int8": f"models/{run_id}/{art['tflite_paths']['int8'].name}",
            },
            "benchmarks": [r for r in summary_rows if r["run_id"] == run_id],
        }

    metadata = {
        "project": "PlantVile",
        "num_classes": len(class_names),
        "class_names_file": "class_names.json",
        "labels_file": "labels.txt",
        "recommended_web_model": rec_web,
        "recommended_mobile_model": rec_mobile_row["run_id"] if rec_mobile_row else rec_web,
        "recommended_mobile_quant": rec_mobile_row["format"].replace("tflite_", "") if rec_mobile_row else "fp16",
        "low_confidence_threshold": 0.70,
        "eval_split": summary_rows[0]["eval_split"] if summary_rows else "val",
        "models": models_meta,
    }
    save_json(str(handoff_dir / "model_metadata.json"), metadata)

    # Write standalone inference_api.py (no dependency on repo src/)
    api_code = '''"""Standalone inference & Grad-CAM helper for PlantVile UI backends (Cloud Web / Laptop / Lite).

Zero dependency on training repository source files (`src/`).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
import tensorflow as tf
from PIL import Image


def _softmax_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Apply post-hoc temperature scaling directly to softmax probabilities: softmax(log(p)/T)."""
    if temperature <= 0 or abs(temperature - 1.0) < 1e-6:
        return probs
    log_p = np.log(np.clip(probs, 1e-12, 1.0))
    scaled = log_p / float(temperature)
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return (exp_x / np.sum(exp_x, axis=-1, keepdims=True)).astype(np.float32)


def _preprocess(image_or_path, height: int, width: int, arch: str) -> tuple[np.ndarray, np.ndarray]:
    """Match training tf.data bilinear resize & per-architecture normalization exactly."""
    if isinstance(image_or_path, (str, os.PathLike)):
        raw = tf.io.read_file(os.fspath(image_or_path))
        decoded = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img_01 = tf.image.convert_image_dtype(decoded, tf.float32)
    elif isinstance(image_or_path, Image.Image):
        arr = np.asarray(image_or_path.convert("RGB"), dtype=np.float32) / 255.0
        img_01 = tf.convert_to_tensor(arr, dtype=tf.float32)
    else:
        raise TypeError("image_or_path must be a file path or PIL.Image.Image")

    resized_01 = tf.image.resize(img_01, [height, width], method="bilinear")
    display_rgb = np.clip(np.round(resized_01.numpy() * 255.0), 0, 255).astype(np.uint8)
    img_255 = resized_01 * 255.0
    if arch == "mobilenetv2":
        tensor = keras.applications.mobilenet_v2.preprocess_input(img_255)
    elif arch == "mobilenetv3_large":
        tensor = keras.applications.mobilenet_v3.preprocess_input(img_255)
    elif arch == "efficientnetb0":
        tensor = keras.applications.efficientnet.preprocess_input(img_255)
    else:
        tensor = img_255
    batch = np.expand_dims(np.asarray(tensor, dtype=np.float32), axis=0)
    return display_rgb, batch


def _find_last_spatial_layer(model: keras.Model) -> int:
    for idx in range(len(model.layers) - 1, -1, -1):
        layer = model.layers[idx]
        if isinstance(layer, keras.Model):
            try:
                if len(layer.output.shape) == 4:
                    return idx
            except Exception:
                pass
    for idx in range(len(model.layers) - 1, -1, -1):
        if isinstance(model.layers[idx], keras.layers.Conv2D):
            return idx
    raise ValueError("No 4D spatial layer found for Grad-CAM")


def _compute_gradcam_overlay(
    model: keras.Model, batch: np.ndarray, display_rgb: np.ndarray, class_index: int | None = None, alpha: float = 0.4
) -> np.ndarray:
    from matplotlib import colormaps

    split_idx = _find_last_spatial_layer(model)
    x = tf.convert_to_tensor(batch, dtype=tf.float32)
    for idx in range(split_idx + 1):
        layer = model.layers[idx]
        if not isinstance(layer, keras.layers.InputLayer):
            x = layer(x, training=False)
    conv_out = x

    with tf.GradientTape() as tape:
        tape.watch(conv_out)
        feat = conv_out
        for layer in model.layers[split_idx + 1 : -1]:
            feat = layer(feat, training=False)
        last = model.layers[-1]
        if isinstance(last, keras.layers.Dense) and getattr(last.activation, "__name__", "") == "softmax":
            logits = tf.matmul(tf.cast(feat, tf.float32), tf.cast(last.kernel, tf.float32))
            if last.use_bias:
                logits = logits + tf.cast(last.bias, tf.float32)
            probs = tf.nn.softmax(logits, axis=-1)
            target = int(tf.argmax(probs[0])) if class_index is None else int(class_index)
            score = logits[:, target]
        else:
            probs = tf.cast(last(feat, training=False), tf.float32)
            target = int(tf.argmax(probs[0])) if class_index is None else int(class_index)
            score = probs[:, target]

    grads = tape.gradient(score, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.nn.relu(tf.reduce_sum(conv_out[0] * pooled, axis=-1)).numpy().astype(np.float32)
    max_v = float(heatmap.max())
    if max_v > 1e-12:
        heatmap /= max_v
    h, w = display_rgb.shape[:2]
    hm_img = Image.fromarray((np.clip(heatmap, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    hm_resized = np.asarray(hm_img.resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    colored = (colormaps["jet"](hm_resized)[..., :3] * 255.0).astype(np.float32)
    blended = (1.0 - alpha) * display_rgb.astype(np.float32) + alpha * colored
    return np.clip(blended, 0, 255).astype(np.uint8)


class PlantDiseasePredictor:
    """Unified inference API supporting Keras FP32 (+Grad-CAM) and TFLite (FP32/FP16/INT8)."""

    def __init__(
        self,
        bundle_dir: str | Path,
        model_key: str | None = None,
        backend: str = "keras",
        quant: str = "fp16",
        num_threads: int = 4,
    ) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.meta = json.loads((self.bundle_dir / "model_metadata.json").read_text(encoding="utf-8"))
        self.class_names: list[str] = json.loads((self.bundle_dir / "class_names.json").read_text(encoding="utf-8"))
        self.backend = backend.lower()
        if self.backend not in ("keras", "tflite"):
            raise ValueError("backend must be 'keras' or 'tflite'")

        if model_key is None:
            model_key = (
                self.meta["recommended_web_model"]
                if self.backend == "keras"
                else self.meta["recommended_mobile_model"]
            )
        self.model_key = model_key
        self.model_info = self.meta["models"][model_key]
        self.arch = self.model_info["arch"]
        _, self.height, self.width, _ = self.model_info["input_shape"]
        self.temperature = float(self.model_info["temperature_T"])
        self.low_conf_threshold = float(self.meta.get("low_confidence_threshold", 0.70))

        if self.backend == "keras":
            kpath = self.bundle_dir / self.model_info["files"]["keras_fp32"]
            self._keras_model = keras.models.load_model(str(kpath), compile=False)

            @tf.function(jit_compile=False)
            def _traced(x):
                return self._keras_model(x, training=False)

            self._keras_fn = _traced
            # Warmup trace once
            dummy = tf.zeros([1, self.height, self.width, 3], dtype=tf.float32)
            _ = self._keras_fn(dummy)
        else:
            qkey = f"tflite_{quant.lower()}"
            if qkey not in self.model_info["files"]:
                raise ValueError(f"Quant variant not found: {qkey}")
            tpath = self.bundle_dir / self.model_info["files"][qkey]
            try:
                self._interpreter = tf.lite.Interpreter(model_path=str(tpath), num_threads=num_threads)
                self._interpreter.allocate_tensors()
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "xnnpack" not in msg and "delegate" not in msg:
                    raise
                self._interpreter = tf.lite.Interpreter(
                    model_path=str(tpath),
                    num_threads=num_threads,
                    experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
                )
                self._interpreter.allocate_tensors()
            self._in_idx = self._interpreter.get_input_details()[0]["index"]
            self._out_idx = self._interpreter.get_output_details()[0]["index"]

    def predict(
        self,
        image_or_path,
        top_k: int = 3,
        with_gradcam: bool = False,
        apply_temperature: bool = True,
    ) -> dict:
        display_rgb, batch = _preprocess(image_or_path, self.height, self.width, self.arch)
        if self.backend == "keras":
            raw_probs = self._keras_fn(tf.convert_to_tensor(batch, dtype=tf.float32)).numpy()[0]
        else:
            self._interpreter.set_tensor(self._in_idx, batch)
            self._interpreter.invoke()
            raw_probs = self._interpreter.get_tensor(self._out_idx)[0]

        probs = _softmax_temperature(raw_probs[None, :], self.temperature)[0] if apply_temperature else raw_probs
        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])
        order = np.argsort(probs)[::-1][: max(1, min(int(top_k), len(self.class_names)))]

        gradcam_rgb = None
        gradcam_note = None
        if with_gradcam:
            if self.backend == "keras":
                gradcam_rgb = _compute_gradcam_overlay(self._keras_model, batch, display_rgb, class_index=pred_idx)
            else:
                gradcam_note = "Grad-CAM memerlukan backend='keras'"

        return {
            "model_key": self.model_key,
            "backend": self.backend,
            "pred_index": pred_idx,
            "pred_class": self.class_names[pred_idx],
            "confidence": round(conf, 6),
            "low_confidence_warning": conf < self.low_conf_threshold,
            "temperature_applied": apply_temperature,
            "top_k": [
                {"class": self.class_names[int(i)], "confidence": round(float(probs[int(i)]), 6)}
                for i in order
            ],
            "original_rgb": display_rgb,
            "gradcam_overlay_rgb": gradcam_rgb,
            "gradcam_note": gradcam_note,
        }
'''
    (handoff_dir / "inference_api.py").write_text(api_code, encoding="utf-8")

    # Write UI_INTEGRATION_README.md
    readme = f"""# PlantVile — Panduan Integrasi UI (Web Cloud & Ekspansi Mobile)

Paket ini siap pakai tanpa perlu mengimpor folder training (`src/`).

## 1. Pilihan Backend untuk Web + Backend Cloud
- **Opsi A (`backend="keras"`, rekomendasi fitur penuh):** Memakai `*_infer_fp32.keras` (~10–18 MB). Mendukung prediksi cepat (`@tf.function`) **dan visualisasi Grad-CAM** di website.
- **Opsi B (`backend="tflite", quant="fp16"` atau `"int8"`):** Memakai `.tflite` (~2–5 MB). Sangat hemat RAM untuk cloud hosting tier gratis (Render/Railway/HF Spaces 512 MB), tetapi **tidak menghasilkan Grad-CAM**.

### Contoh Cepat di FastAPI / Streamlit / Flask
```python
from inference_api import PlantDiseasePredictor

# Rekomendasi Web dengan Grad-CAM:
predictor = PlantDiseasePredictor("ui_handoff_bundle", backend="keras")

# Atau hemat RAM dengan TFLite:
# predictor = PlantDiseasePredictor("ui_handoff_bundle", backend="tflite", quant="fp16")

res = predictor.predict("daun_tomat.jpg", top_k=3, with_gradcam=True, apply_temperature=True)
print(res["pred_class"], res["confidence"], res["low_confidence_warning"])
# res["gradcam_overlay_rgb"] -> numpy uint8 (224, 224, 3) siap diubah ke PNG/base64!
```

## 2. Kontrak Tensor & Preprocessing (Jika Ekspansi ke Flutter/Android Native)
- Input tensor: `[1, 224, 224, 3]`, tipe `float32`, resize `bilinear`.
- `labels.txt`: 15 kelas berurutan indeks `0..14`.
- Preprocessing per arsitektur (lihat detail di `model_metadata.json`):
  - `mobilenetv2`: `(pixel_0_to_255 / 127.5) - 1.0` -> rentang `[-1, 1]`.
  - `efficientnetb0` & `mobilenetv3_large`: biarkan `[0, 255]` (`float32`), karena layer `Rescaling` tertanam di dalam grafik `.tflite` / `.keras`.
- Post-processing opsional (kalibrasi probabilitas): aplikasikan `temperature_T` dari `model_metadata.json` (`softmax(log(p) / T)`), lalu tampilkan peringatan jika `confidence < {metadata['low_confidence_threshold']}`.
"""
    (handoff_dir / "UI_INTEGRATION_README.md").write_text(readme, encoding="utf-8")

    # Write android_benchmark_guide.md
    guide_lines = [
        "# Panduan Pengukuran Latency Asli di Smartphone Android (ADB `benchmark_model`)",
        "",
        "Jika proyek dikembangkan ke aplikasi *on-device* Android, gunakan biner resmi TensorFlow Lite Benchmark (`android_aarch64_benchmark_model`) lewat ADB:",
        "",
        "1. Aktifkan **USB Debugging** di HP Android, sambungkan via kabel, lalu pastikan terdeteksi:",
        "   ```bash",
        "   adb devices",
        "   ```",
        "2. Unduh `android_aarch64_benchmark_model` (dari halaman rilis/nightly TensorFlow Lite Benchmark Tools), lalu push ke `/data/local/tmp`:",
        "   ```bash",
        "   adb push android_aarch64_benchmark_model /data/local/tmp/",
        "   adb shell chmod +x /data/local/tmp/android_aarch64_benchmark_model",
        "   ```",
        "3. Push file `.tflite` pilihan (misal varian `FP16` dan `INT8`) lalu jalankan benchmark (50 run, 4 thread CPU + opsi NNAPI):",
        "   ```bash",
    ]
    for run_id, minfo in models_meta.items():
        for q in ("fp16", "int8"):
            rel_file = minfo["files"][f"tflite_{q}"]
            fname = Path(rel_file).name
            guide_lines.extend([
                f"   # {run_id} ({q.upper()})",
                f"   adb push {rel_file} /data/local/tmp/{fname}",
                f"   adb shell /data/local/tmp/android_aarch64_benchmark_model --graph=/data/local/tmp/{fname} --num_threads=4 --warmup_runs=10 --num_runs=50",
            ])
    guide_lines.extend([
        "   ```",
        "4. Catat nilai `Inference timings in us: avg=...` (bagi 1.000 untuk memperoleh **ms**) ke dalam tabel kelayakan perangkat mobile di laporan.",
        "",
    ])
    (handoff_dir / "android_benchmark_guide.md").write_text("\n".join(guide_lines), encoding="utf-8")


def process_runs(
    runs_dir: Path,
    run_ids: list[str],
    data_root: Path,
    eval_split: str,
    out_dir: Path,
    build_handoff: bool,
    handoff_dir: Path,
    calib_per_class: int,
    bench_warmup: int,
    bench_runs: int,
    num_threads: int,
    max_eval_samples: int,
    seed: int,
) -> dict:
    """Export, quantize, evaluate, and benchmark all requested Stage 2 runs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    preds_by_key: dict[tuple[str, str], np.ndarray] = {}
    y_true_ref: np.ndarray | None = None
    run_artifacts: dict = {}
    class_names_ref: list[str] = []

    for run_id in run_ids:
        rdir = runs_dir / run_id
        ckpt_path = rdir / "best.keras"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        cfg = json.loads((rdir / "config_used.json").read_text(encoding="utf-8"))
        saved_classes = json.loads((rdir / "class_names.json").read_text(encoding="utf-8"))
        cfg_local = copy.deepcopy(cfg)
        cfg_local["data"]["root"] = str(data_root)
        cfg_local["data"]["cache"] = "none"
        cfg_local["runtime"]["device"] = "cpu"
        cfg_local["runtime"]["strategy"] = "none"

        bundle = prepare_datasets(cfg_local, verbose=False)
        class_names = bundle["class_names"]
        if class_names != saved_classes:
            raise RuntimeError(f"Class names mismatch for {run_id}")
        class_names_ref = class_names
        num_classes = len(class_names)

        run_out = out_dir / run_id
        run_out.mkdir(parents=True, exist_ok=True)
        fp32_keras_path = run_out / f"{run_id}_infer_fp32.keras"
        LOGGER.info("[%s] Exporting pure float32 .keras...", run_id)
        fp32_model, keras_info = export_pure_fp32_keras(
            run_id, ckpt_path, cfg_local, fp32_keras_path,
            bundle["val"]["paths"], bundle["val"]["labels"],
        )

        # Build val dataset for strict val-only Temperature Scaling
        runtime_ctx = dict(cfg_local["runtime"])
        runtime_ctx["_num_classes"] = num_classes
        val_ds_ts = build_dataset(
            bundle["val"]["paths"], bundle["val"]["labels"],
            cfg_local["data"], runtime_ctx, cfg_local["model"],
            training=False, batch_size=64, one_hot=False, split_name="val",
        )
        temp_info = _fit_val_temperature(fp32_model, val_ds_ts, seed=seed)

        # Calibration dataset strictly from train split
        rep_fn, n_calib = make_representative_dataset_fn(
            bundle["train"]["paths"], bundle["train"]["labels"],
            cfg_local, per_class=calib_per_class, seed=seed,
        )
        LOGGER.info("[%s] Converting TFLite (FP32, FP16, INT8 with %d train calib imgs)...", run_id, n_calib)
        tflite_paths: dict[str, Path] = {}
        tflite_meta: dict[str, dict] = {}
        for q in ("fp32", "fp16", "int8"):
            tbytes, tmeta = convert_keras_to_tflite(fp32_model, q, rep_dataset_fn=rep_fn if q == "int8" else None)
            tpath = run_out / f"{run_id}_{q}.tflite"
            tpath.write_bytes(tbytes)
            tflite_paths[q] = tpath
            tflite_meta[q] = tmeta

        # Prepare target eval dataset
        eval_paths = list(bundle[eval_split]["paths"])
        eval_labels = list(bundle[eval_split]["labels"])
        if max_eval_samples > 0 and len(eval_paths) > max_eval_samples:
            step_idx = np.linspace(0, len(eval_paths) - 1, max_eval_samples).astype(int)
            eval_paths = [eval_paths[i] for i in step_idx]
            eval_labels = [eval_labels[i] for i in step_idx]

        eval_ds = build_dataset(
            eval_paths, eval_labels,
            cfg_local["data"], runtime_ctx, cfg_local["model"],
            training=False, batch_size=64, one_hot=False, split_name=eval_split,
        )
        _, sample_batch = preprocess_image(eval_paths[0], cfg_local["data"], cfg_local["model"]["arch"])

        LOGGER.info("[%s] Evaluating & benchmarking Keras FP32 on split='%s' (n=%d)...", run_id, eval_split, len(eval_paths))
        yk, pk = _predict_keras(fp32_model, eval_ds)
        y_true_ref = yk
        mk, y_hat_k = _evaluate_predictions(yk, pk, num_classes, seed=seed)
        lat_k = _benchmark_keras_latency(fp32_model, sample_batch, warmup=bench_warmup, runs=bench_runs)
        preds_by_key[(run_id, "keras_fp32")] = y_hat_k
        base_f1 = mk["f1_macro"]

        summary_rows.append({
            "run_id": run_id,
            "arch": cfg["model"]["arch"],
            "eval_split": eval_split,
            "samples": len(eval_paths),
            "format": "keras_fp32",
            "size_mb": keras_info["fp32_keras_mb"],
            "delta_f1_vs_fp32": 0.0,
            "p50_ms": lat_k["p50_ms"],
            "p95_ms": lat_k["p95_ms"],
            "mean_ms": lat_k["mean_ms"],
            **mk,
        })

        for q in ("fp32", "fp16", "int8"):
            tpath = tflite_paths[q]
            LOGGER.info("[%s] Evaluating TFLite %s...", run_id, q.upper())
            yt, pt = _predict_tflite(tpath, eval_ds, num_threads=num_threads)
            mt, y_hat_t = _evaluate_predictions(yt, pt, num_classes, seed=seed)
            lat_t = _benchmark_tflite_latency(
                tpath, sample_batch, num_threads=num_threads, warmup=bench_warmup, runs=bench_runs
            )
            fmt = f"tflite_{q}"
            preds_by_key[(run_id, fmt)] = y_hat_t
            summary_rows.append({
                "run_id": run_id,
                "arch": cfg["model"]["arch"],
                "eval_split": eval_split,
                "samples": len(eval_paths),
                "format": fmt,
                "size_mb": round(tpath.stat().st_size / (1024 * 1024), 3),
                "delta_f1_vs_fp32": round(mt["f1_macro"] - base_f1, 6),
                "p50_ms": lat_t["p50_ms"],
                "p95_ms": lat_t["p95_ms"],
                "mean_ms": lat_t["mean_ms"],
                **mt,
            })

        run_artifacts[run_id] = {
            "arch": cfg["model"]["arch"],
            "img_height": int(cfg["data"]["img_height"]),
            "img_width": int(cfg["data"]["img_width"]),
            "keras_fp32_path": fp32_keras_path,
            "tflite_paths": tflite_paths,
            "temperature": temp_info,
        }

    # Exact McNemar pairwise tests
    mcnemar_rows: list[dict] = []
    if y_true_ref is not None:
        # 1) Within-architecture quantization impact vs keras_fp32
        for run_id in run_ids:
            base_pred = preds_by_key[(run_id, "keras_fp32")]
            for fmt in ("tflite_fp32", "tflite_fp16", "tflite_int8"):
                res = exact_mcnemar(y_true_ref, base_pred, preds_by_key[(run_id, fmt)])
                mcnemar_rows.append({
                    "comparison": f"{run_id}: keras_fp32 vs {fmt}",
                    **res,
                })
        # 2) Cross-architecture comparisons on matching formats
        for fmt in ("keras_fp32", "tflite_fp16", "tflite_int8"):
            for i in range(len(run_ids)):
                for j in range(i + 1, len(run_ids)):
                    r_a, r_b = run_ids[i], run_ids[j]
                    res = exact_mcnemar(
                        y_true_ref, preds_by_key[(r_a, fmt)], preds_by_key[(r_b, fmt)]
                    )
                    mcnemar_rows.append({
                        "comparison": f"{fmt}: {r_a} vs {r_b}",
                        **res,
                    })

    # Write summary CSV & JSON
    csv_fields = [
        "run_id", "arch", "eval_split", "samples", "format", "size_mb",
        "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "worst_class_recall", "ece", "delta_f1_vs_fp32",
        "acc_ci95", "f1_ci95", "p50_ms", "p95_ms", "mean_ms",
    ]
    csv_path = out_dir / "mobile_benchmark_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    if mcnemar_rows:
        mc_path = out_dir / "mcnemar_pairwise.csv"
        with open(mc_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["comparison", "a_right_b_wrong", "a_wrong_b_right", "p_value"]
            )
            writer.writeheader()
            writer.writerows(mcnemar_rows)

    payload = {
        "eval_split": eval_split,
        "num_threads": num_threads,
        "calib_per_class": calib_per_class,
        "summary": summary_rows,
        "mcnemar_pairwise": mcnemar_rows,
    }
    save_json(str(out_dir / "mobile_benchmark_summary.json"), payload)

    if build_handoff:
        _write_handoff_files(handoff_dir, class_names_ref, run_artifacts, summary_rows)
        LOGGER.info("UI handoff bundle written -> %s", handoff_dir)

    return payload


def discover_stage2_runs(runs_dir: Path) -> list[str]:
    """Auto-discover completed Stage 2 candidate runs present in runs_dir."""
    found = [
        rid for rid in DEFAULT_STAGE2_RUNS
        if (runs_dir / rid / "best.keras").is_file() and (runs_dir / rid / "config_used.json").is_file()
    ]
    if not found and runs_dir.is_dir():
        for child in sorted(runs_dir.iterdir()):
            if child.is_dir() and (child / "best.keras").is_file() and (child / "config_used.json").is_file():
                found.append(child.name)
    return found


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Pure FP32 Keras + TFLite PTQ and build UI handoff bundle.")
    parser.add_argument("--runs-dir", type=Path, default=Path("results/runs"))
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--data-root", type=Path, default=Path("data/PlantVillage"))
    parser.add_argument("--eval-split", choices=["val", "test"], default="val")
    parser.add_argument("--out-dir", type=Path, default=Path("results/mobile_export"))
    parser.add_argument("--build-handoff", action="store_true")
    parser.add_argument("--handoff-dir", type=Path, default=Path("results/ui_handoff_bundle"))
    parser.add_argument("--calib-per-class", type=int, default=20)
    parser.add_argument("--bench-warmup", type=int, default=10)
    parser.add_argument("--bench-runs", type=int, default=50)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    tf.get_logger().setLevel("ERROR")
    args = parse_args(argv)
    set_seed(args.seed)
    configure_device("cpu")

    run_ids = args.runs or discover_stage2_runs(args.runs_dir)
    if not run_ids:
        raise RuntimeError(f"No completed Stage 2 runs found in {args.runs_dir}")

    payload = process_runs(
        runs_dir=args.runs_dir,
        run_ids=run_ids,
        data_root=args.data_root,
        eval_split=args.eval_split,
        out_dir=args.out_dir,
        build_handoff=args.build_handoff,
        handoff_dir=args.handoff_dir,
        calib_per_class=args.calib_per_class,
        bench_warmup=args.bench_warmup,
        bench_runs=args.bench_runs,
        num_threads=args.num_threads,
        max_eval_samples=args.max_eval_samples,
        seed=args.seed,
    )

    print(f"\n=== MOBILE & WEB BENCHMARK SUMMARY (split={payload['eval_split']}) ===")
    for row in payload["summary"]:
        print(
            f"  {row['run_id']:22s} | {row['format']:11s} | {row['size_mb']:6.2f} MB | "
            f"acc={row['accuracy']:.4f} | f1={row['f1_macro']:.4f} (dF1={row['delta_f1_vs_fp32']:+.4f}) | "
            f"ECE={row['ece']:.4f} | p50={row['p50_ms']:6.2f} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
