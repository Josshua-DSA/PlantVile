"""Generate notebook/kaggle_experiment_runner.ipynb for Kaggle T4x2 GPU and CPU.

Run:
    python scripts/build_kaggle_notebook.py

The generated notebook runs a clean, noise-free, 1-click reproducible workflow
for 3 mobile-edge architectures (MobileNetV2, EfficientNet-B0, MobileNetV3-Large),
automatic 1x test evaluation, TFLite PTQ (FP32/FP16/INT8), and UI handoff bundling.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "notebook" / "kaggle_experiment_runner.ipynb"


def md(text: str) -> dict:
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": (text.strip() + "\n").splitlines(True),
    }


def code(text: str) -> dict:
    """Create a stripped code cell."""
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": (text.strip() + "\n").splitlines(True),
    }


def build_cells() -> list[dict]:
    """Construct clean, ordered cells for the Kaggle experiment runner notebook."""
    cells = []

    # ---------------------------------------------------------------- Cell 0
    cells.append(md("""
# PlantVile — Clean Mobile-Edge Experiment Runner (Kaggle T4×2 / CPU)

Notebook ini menjalankan eksperimen **PlantVile** secara penuh, bersih (*noise-free*), dan terstandar untuk **3 arsitektur *mobile-edge*** (`MobileNetV2`, `EfficientNet-B0`, dan `MobileNetV3-Large`) dari branch `experiment-arfi`.

### Pengaturan Kaggle Sebelum Menjalankan (*Save & Run All*)
1. **Accelerator:** `GPU T4 x2` (atau `None` untuk audit CPU).
2. **Internet:** `ON` (untuk `git clone` dan bobot ImageNet).
3. **Dataset Input:** Tambahkan `emmarex/plantdisease` (`/kaggle/input/datasets/emmarex/plantdisease/PlantVillage`).

### Ringkasan Tahapan (8 Sel Kode)
| Sel | Tahap | Output Utama |
| :---: | :--- | :--- |
| **1** | Setup & Sinkronisasi Repo | Tabel status hardware, GPU, disk, & commit aktif |
| **2** | Audit Dataset & `--check` | Tabel audit integritas dataset + validasi *pre-flight* |
| **3** | Matriks Eksperimen (`m1`–`m4b`) | Jadwal 6 run terkontrol (Stage 1 + Stage 2 untuk 3 arsitektur) |
| **4** | Training + Evaluasi `val` | Log epoch ringkas (`verbose=2`) + `step_history.csv` |
| **5** | Leaderboard `val` & Kurva GD | Ranking `val`, kurva Stage 1→2 kontinu, overlay, & kurva per step (EMA) |
| **6** | Evaluasi Final 1× (`test`) | Evaluasi otomatis split `test` untuk semua kandidat Stage 2 + CM & Grad-CAM |
| **7** | Ekspor Pure FP32 + TFLite PTQ | Benchmark ukuran (MB), latency (ms), & retensi `FP32`/`FP16`/`INT8` |
| **8** | Pengemasan Unduhan ZIP | `ui_handoff_bundle.zip` (Web + Mobile) & `experiment_reports.zip` |
"""))

    # ---------------------------------------------------------------- Cell 1
    cells.append(md("## 1. Setup Lingkungan & Sinkronisasi Repository"))
    cells.append(code("""import json
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from collections import deque
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"
warnings.filterwarnings("ignore")

import pandas as pd
from IPython.display import Image as IPyImage
from IPython.display import display as ipy_display

REPO_URL = "https://github.com/Josshua-DSA/PlantVile.git"
BRANCH = "experiment-arfi"

DATA_ROOT_CANDIDATES = [
    Path("/kaggle/input/datasets/emmarex/plantdisease/PlantVillage"),
    Path("/kaggle/input/plantdisease/PlantVillage"),
    Path("data/PlantVillage"),
]

IS_KAGGLE = Path("/kaggle/working").is_dir()
BASE_OUT = Path("/kaggle/working") if IS_KAGGLE else Path.cwd() / "kaggle_local_sim"
WORK_DIR = BASE_OUT / "PlantVile"
RUNS_DIR = BASE_OUT / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = next((p for p in DATA_ROOT_CANDIDATES if p.is_dir()), None)
assert DATA_ROOT is not None, f"Dataset tidak ditemukan di: {[str(p) for p in DATA_ROOT_CANDIDATES]}"
DATA_ROOT = DATA_ROOT.resolve()

_NOISE_RE = re.compile(
    r"(^WARNING: All log messages before absl::InitializeLog|"
    r"^WARNING:tensorflow:|"
    r"^[IWE]\\d{4}\\s+\\d{2}:\\d{2}:\\d{2}\\.\\d+|"
    r"oneDNN custom operations|cpu_feature_guard|"
    r"Unable to register cuFFT|Unable to register cuDNN|Unable to register cuBLAS|"
    r"computation_placer|successful NUMA|NUMA node|"
    r"This TensorFlow binary is optimized|XLA service|Loaded runtime CuDNN|"
    r"AllReduce|local_device_option|TensorFlow GPU support is not available)"
)


def run_clean(cmd: list[str]) -> None:
    \"\"\"Run subprocess streaming only meaningful progress lines and preserving full tail on failure.\"\"\"
    env = {
        **os.environ,
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "ABSL_MIN_LOG_LEVEL": "3",
        "TF_ENABLE_ONEDNN_OPTS": "0",
        "CUDA_MODULE_LOADING": "LAZY",
        "PYTHONUNBUFFERED": "1",
        "PYTHONWARNINGS": "ignore",
    }
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    tail = deque(maxlen=40)
    prev_blank = False
    for line in proc.stdout:
        tail.append(line.rstrip())
        clean = line.strip()
        if _NOISE_RE.search(clean):
            continue
        if not clean:
            if prev_blank:
                continue
            prev_blank = True
            print("", flush=True)
            continue
        prev_blank = False
        print(line, end="", flush=True)
    rc = proc.wait()
    if rc != 0:
        print("\\n--- LAST LOG LINES ON ERROR ---")
        for item in tail:
            print(item)
        raise subprocess.CalledProcessError(rc, cmd)


if (WORK_DIR / ".git").is_dir():
    run_clean(["git", "-C", str(WORK_DIR), "fetch", "-q", "origin", BRANCH])
    run_clean(["git", "-C", str(WORK_DIR), "checkout", "-q", BRANCH])
    run_clean(["git", "-C", str(WORK_DIR), "pull", "-q", "origin", BRANCH])
else:
    run_clean(["git", "clone", "-q", "-b", BRANCH, "--single-branch", REPO_URL, str(WORK_DIR)])

os.chdir(WORK_DIR)
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

import keras
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
gpus = tf.config.list_physical_devices("GPU")
HAS_GPU = len(gpus) > 0
CONFIG_PATH = "configs/kaggle_t4x2.yaml" if HAS_GPU else "configs/default.yaml"
commit_id = subprocess.check_output(["git", "log", "-n", "1", "--oneline"], text=True).strip()
disk = shutil.disk_usage(BASE_OUT)

env_table = pd.DataFrame([
    ("Branch / Commit", f"{BRANCH} ({commit_id})"),
    ("Python / TF / Keras", f"{sys.version.split()[0]} / {tf.__version__} / {keras.__version__}"),
    ("Hardware", f"{len(gpus)}x GPU ({gpus[0].name})" if gpus else "CPU-only"),
    ("Config Preset", CONFIG_PATH),
    ("Data Root", str(DATA_ROOT)),
    ("Runs Dir", str(RUNS_DIR)),
    ("Free Disk", f"{disk.free / (1024**3):.2f} GB / {disk.total / (1024**3):.2f} GB"),
], columns=["Setting", "Value"])
ipy_display(env_table)
"""))

    # ---------------------------------------------------------------- Cell 2
    cells.append(md("## 2. Audit Dataset & Validasi *Pre-Flight*"))
    cells.append(code("""audit_out = RUNS_DIR / "dataset_audit.json"
if not audit_out.is_file():
    run_clean([sys.executable, "scripts/audit_dataset.py", "--root", str(DATA_ROOT), "--out", str(audit_out)])

audit_data = json.loads(audit_out.read_text(encoding="utf-8"))
ipy_display(pd.DataFrame([
    ("Classes", audit_data["classes"]),
    ("Valid Images", audit_data["valid_images"]),
    ("Imbalance Ratio", f"{audit_data['imbalance_ratio']}x"),
    ("MD5 Duplicate Groups", audit_data["exact_md5_duplicates"]["groups"]),
    ("PIL Corrupt Count", audit_data["pil_verification"]["corrupt_count"]),
], columns=["Audit Metric", "Value"]))

check_cmd = [
    sys.executable, "main.py", "--check", "--quiet",
    "--config", CONFIG_PATH,
    "--set", f"data.root={DATA_ROOT}",
]
if not HAS_GPU:
    check_cmd += ["--set", "runtime.device=cpu"]
run_clean(check_cmd)
"""))

    # ---------------------------------------------------------------- Cell 3
    cells.append(md("## 3. Matriks Eksperimen Terkontrol (3 Arsitektur *Mobile-Edge*)"))
    cells.append(code("""COMMON_OVERRIDES = [
    "data.img_height=224",
    "data.img_width=224",
    "data.channels=3",
    "data.rescale=true",
    "data.normalize=imagenet",
    "data.deduplicate=true",
    "data.group_same_leaf=true",
    "data.shuffle_seed=42",
    "data.augment=true",
    "data.aug_hflip=true",
    "data.aug_vflip=true",
    "data.aug_rotation=0.08",
    "data.aug_width_shift=0.10",
    "data.aug_height_shift=0.10",
    "data.aug_zoom=0.12",
    "data.aug_shear=0.0",
    "data.aug_brightness=0.20",
    "data.aug_contrast=0.20",
    "data.aug_fill_mode=reflect",
    "model.dense_units=256",
    "model.dropout=0.3",
    "model.batch_norm=true",
    "model.l2_weight_decay=0.0001",
    "train.adamw_weight_decay=0.004",
    "train.log_step_every=10",
    "train.early_stopping=true",
    "train.es_monitor=val_loss",
    "train.es_patience=6",
    "train.es_restore_best=true",
    "train.reduce_lr=true",
    "train.lr_monitor=val_loss",
    "train.lr_factor=0.5",
    "train.lr_patience=3",
    "train.lr_min_lr=1e-06",
]

EXP_REGISTRY = {
    "m1_mobilenetv2_head": {
        "desc": "MobileNetV2 Stage 1: frozen head warmup (10 ep, Adam 1e-3)",
        "overrides": [
            "model.arch=mobilenetv2", "model.freeze_base=true", "model.fine_tune_at=0",
            "train.optimizer=adam", "train.learning_rate=0.001", "train.epochs=10",
            "train.loss=categorical_crossentropy", "train.label_smoothing=0.05",
            "train.class_weight=balanced",
        ],
    },
    "m2_mobilenetv2_ft": {
        "desc": "MobileNetV2 Stage 2: fine-tune >=107 (25 ep, AdamW 1e-4)",
        "resume_from_run": "m1_mobilenetv2_head",
        "overrides": [
            "model.arch=mobilenetv2", "model.freeze_base=true", "model.fine_tune_at=107",
            "train.optimizer=adamw", "train.learning_rate=0.0001", "train.epochs=25",
            "train.loss=categorical_crossentropy", "train.label_smoothing=0.05",
            "train.class_weight=balanced",
        ],
    },
    "m3a_effnetb0_head": {
        "desc": "EfficientNet-B0 Stage 1: frozen head warmup (10 ep, Adam 1e-3)",
        "overrides": [
            "model.arch=efficientnetb0", "model.freeze_base=true", "model.fine_tune_at=0",
            "train.optimizer=adam", "train.learning_rate=0.001", "train.epochs=10",
            "train.loss=categorical_crossentropy", "train.label_smoothing=0.05",
            "train.class_weight=balanced",
        ],
    },
    "m3b_effnetb0_ft": {
        "desc": "EfficientNet-B0 Stage 2: fine-tune >=162 (25 ep, AdamW 1e-4)",
        "resume_from_run": "m3a_effnetb0_head",
        "overrides": [
            "model.arch=efficientnetb0", "model.freeze_base=true", "model.fine_tune_at=162",
            "train.optimizer=adamw", "train.learning_rate=0.0001", "train.epochs=25",
            "train.loss=categorical_crossentropy", "train.label_smoothing=0.05",
            "train.class_weight=balanced",
        ],
    },
    "m4a_mobilenetv3l_head": {
        "desc": "MobileNetV3-Large Stage 1: frozen head warmup (10 ep, Adam 1e-3)",
        "overrides": [
            "model.arch=mobilenetv3_large", "model.freeze_base=true", "model.fine_tune_at=0",
            "train.optimizer=adam", "train.learning_rate=0.001", "train.epochs=10",
            "train.loss=categorical_crossentropy", "train.label_smoothing=0.05",
            "train.class_weight=balanced",
        ],
    },
    "m4b_mobilenetv3l_ft": {
        "desc": "MobileNetV3-Large Stage 2: fine-tune >=124 (25 ep, AdamW 1e-4)",
        "resume_from_run": "m4a_mobilenetv3l_head",
        "overrides": [
            "model.arch=mobilenetv3_large", "model.freeze_base=true", "model.fine_tune_at=124",
            "train.optimizer=adamw", "train.learning_rate=0.0001", "train.epochs=25",
            "train.loss=categorical_crossentropy", "train.label_smoothing=0.05",
            "train.class_weight=balanced",
        ],
    },
}

STAGE2_CANDIDATES = [
    "m2_mobilenetv2_ft",
    "m3b_effnetb0_ft",
    "m4b_mobilenetv3l_ft",
]

RUN_EXPERIMENTS = True
SKIP_IF_DONE = True
STRATEGY_MODE = "auto"

SELECTED_RUNS = [
    "m1_mobilenetv2_head",
    "m2_mobilenetv2_ft",
    "m3a_effnetb0_head",
    "m3b_effnetb0_ft",
    "m4a_mobilenetv3l_head",
    "m4b_mobilenetv3l_ft",
]


def build_run_command(run_id: str, check_resume: bool = False) -> list[str]:
    spec = EXP_REGISTRY[run_id]
    cmd = [
        sys.executable, "main.py", "--yes", "--quiet",
        "--config", CONFIG_PATH,
        "--set", f"data.root={DATA_ROOT}",
        "--set", f"runtime.runs_dir={RUNS_DIR}",
        "--set", f"runtime.run_name={run_id}",
        "--set", "eval.eval_split=val",
        "--set", f"runtime.strategy={STRATEGY_MODE}",
        "--set", "runtime.verbose=2",
    ]
    if not HAS_GPU:
        cmd += ["--set", "runtime.device=cpu"]
    parent_run = spec.get("resume_from_run")
    if parent_run:
        parent_ckpt = RUNS_DIR / parent_run / "best.keras"
        if check_resume:
            assert parent_ckpt.is_file(), f"Warm-start checkpoint belum ada: {parent_ckpt}"
        cmd += ["--set", f"runtime.resume_from={parent_ckpt}"]
    for ov in COMMON_OVERRIDES + spec["overrides"]:
        cmd += ["--set", ov]
    return cmd


schedule_df = pd.DataFrame([
    {
        "run_id": rid,
        "status": "DONE" if (RUNS_DIR / rid / "val_metrics.json").is_file() else "PENDING",
        "warm_start": EXP_REGISTRY[rid].get("resume_from_run", "-"),
        "description": EXP_REGISTRY[rid]["desc"],
    }
    for rid in SELECTED_RUNS
])
ipy_display(schedule_df)
"""))

    # ---------------------------------------------------------------- Cell 4
    cells.append(md("## 4. Eksekusi Training Penuh + Evaluasi `val` (`m1`–`m4b`)"))
    cells.append(code("""if not RUN_EXPERIMENTS:
    print("Training dilewati (RUN_EXPERIMENTS = False).")
else:
    for rid in SELECTED_RUNS:
        done_marker = RUNS_DIR / rid / "val_metrics.json"
        if SKIP_IF_DONE and done_marker.is_file():
            print(f"[SKIP] {rid} sudah selesai.")
            continue
        cmd = build_run_command(rid, check_resume=True)
        print(f"\\n>>> [{rid}] {EXP_REGISTRY[rid]['desc']}")
        t0 = time.time()
        run_clean(cmd)
        print(f"[OK] {rid} selesai dalam {(time.time() - t0) / 60:.2f} menit.")
"""))

    # ---------------------------------------------------------------- Cell 5
    cells.append(md("## 5. Leaderboard Validasi (`val`) & Kurva Gradient Descent"))
    cells.append(code("""import numpy as np

rows = []
for metrics_file in sorted(RUNS_DIR.glob("*/val_metrics.json")):
    rdir = metrics_file.parent
    rid = rdir.name
    m = json.loads(metrics_file.read_text(encoding="utf-8"))
    per_cls = m.get("per_class", [])
    worst_recall = min((c["recall"] for c in per_cls), default=np.nan)
    gap = np.nan
    hist_file = rdir / "history.json"
    if hist_file.is_file():
        h = json.loads(hist_file.read_text(encoding="utf-8"))
        if h.get("accuracy") and h.get("val_accuracy"):
            gap = h["accuracy"][-1] - h["val_accuracy"][-1]
    best_keras = rdir / "best.keras"
    size_mb = best_keras.stat().st_size / (1024 * 1024) if best_keras.is_file() else np.nan
    rows.append({
        "run_id": rid,
        "f1_macro": round(float(m.get("f1_macro", 0.0)), 4),
        "accuracy": round(float(m.get("accuracy", 0.0)), 4),
        "top_3_acc": round(float(m.get("top_k_accuracy", 0.0)), 4),
        "worst_recall": round(float(worst_recall), 4) if not np.isnan(worst_recall) else np.nan,
        "ece": round(float(m.get("calibration", {}).get("ece", np.nan)), 4),
        "misclassified": m.get("num_misclassified", np.nan),
        "train_val_gap": round(float(gap), 4) if not np.isnan(gap) else np.nan,
        "best_keras_mb": round(float(size_mb), 2) if not np.isnan(size_mb) else np.nan,
    })

df_val = pd.DataFrame(rows).sort_values(["f1_macro", "accuracy"], ascending=False).reset_index(drop=True)
df_val.to_csv(RUNS_DIR / "experiment_leaderboard.csv", index=False)
ipy_display(df_val)

gd_out = RUNS_DIR / "gradient_descent_plots"
run_clean([
    sys.executable, "-m", "scripts.plot_gradient_descent",
    "--runs-dir", str(RUNS_DIR),
    "--out-dir", str(gd_out),
])

for png_name in [
    "gd_comparison_stage2.png",
    "gd_combined_mobilenetv2.png",
    "gd_combined_efficientnetb0.png",
    "gd_combined_mobilenetv3_large.png",
    *[f"gd_step_{rid}.png" for rid in STAGE2_CANDIDATES],
]:
    p = gd_out / png_name
    if p.is_file():
        ipy_display(IPyImage(filename=str(p)))
"""))

    # ---------------------------------------------------------------- Cell 6
    cells.append(md("## 6. Evaluasi Final Otomatis 1× pada Split `test` (Seluruh Kandidat Stage 2)"))
    cells.append(code("""test_rows = []
for rid in STAGE2_CANDIDATES:
    rdir = RUNS_DIR / rid
    best_model_path = rdir / "best.keras"
    cfg_used_path = rdir / "config_used.yaml"
    if not best_model_path.is_file():
        continue

    test_json = rdir / "test_metrics.json"
    if not (SKIP_IF_DONE and test_json.is_file()):
        eval_cmd = [
            sys.executable, "main.py", "--quiet",
            "--eval", str(best_model_path),
            "--config", str(cfg_used_path),
            "--set", f"data.root={DATA_ROOT}",
            "--set", "eval.eval_split=test",
        ]
        if not HAS_GPU:
            eval_cmd += ["--set", "runtime.device=cpu"]
        print(f">>> Evaluasi test (1x): {rid}")
        run_clean(eval_cmd)

    tm = json.loads(test_json.read_text(encoding="utf-8"))
    per_cls = tm.get("per_class", [])
    test_rows.append({
        "run_id": rid,
        "accuracy": round(float(tm["accuracy"]), 4),
        "precision_macro": round(float(tm["precision_macro"]), 4),
        "recall_macro": round(float(tm["recall_macro"]), 4),
        "f1_macro": round(float(tm["f1_macro"]), 4),
        "worst_recall": round(float(min(c["recall"] for c in per_cls)), 4) if per_cls else np.nan,
        "ece": round(float(tm["calibration"]["ece"]), 4),
        "misclassified": int(tm.get("num_misclassified", 0)),
    })

df_test = pd.DataFrame(test_rows).sort_values(["f1_macro", "accuracy"], ascending=False).reset_index(drop=True)
df_test.to_csv(RUNS_DIR / "test_leaderboard.csv", index=False)
ipy_display(df_test)

if len(df_test):
    winner_run = df_test.iloc[0]["run_id"]
    print(f"Visualisasi Error Analysis & Grad-CAM split 'test' — {winner_run}:")
    for png_name in ["test_confusion_matrix.png", "test_confidence_calibration.png", "test_gradcam_grid.png"]:
        p = RUNS_DIR / winner_run / png_name
        if p.is_file():
            ipy_display(IPyImage(filename=str(p)))
"""))

    # ---------------------------------------------------------------- Cell 7
    cells.append(md("## 7. Optimasi Mobile TFLite PTQ (`FP32`/`FP16`/`INT8`) & Build UI Handoff Bundle"))
    cells.append(code("""MOBILE_OUT = RUNS_DIR / "mobile_export"
HANDOFF_DIR = BASE_OUT / "ui_handoff_bundle"

export_cmd = [
    sys.executable, "-m", "scripts.export_tflite",
    "--runs-dir", str(RUNS_DIR),
    "--data-root", str(DATA_ROOT),
    "--eval-split", "test",
    "--out-dir", str(MOBILE_OUT),
    "--build-handoff",
    "--handoff-dir", str(HANDOFF_DIR),
]
run_clean(export_cmd)

bench_csv = MOBILE_OUT / "mobile_benchmark_summary.csv"
if bench_csv.is_file():
    df_bench = pd.read_csv(bench_csv)
    cols = [
        "run_id", "format", "size_mb", "accuracy", "f1_macro",
        "delta_f1_vs_fp32", "worst_class_recall", "ece", "p50_ms", "p95_ms",
    ]
    ipy_display(df_bench[cols])
"""))

    # ---------------------------------------------------------------- Cell 8
    cells.append(md("## 8. Pengemasan Deliverable Siap Unduh (`ui_handoff_bundle.zip` & `experiment_reports.zip`)"))
    cells.append(code("""import zipfile

ui_zip = BASE_OUT / "ui_handoff_bundle.zip"
with zipfile.ZipFile(ui_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for fpath in sorted(HANDOFF_DIR.rglob("*")):
        if fpath.is_file():
            zf.write(fpath, arcname=f"ui_handoff_bundle/{fpath.relative_to(HANDOFF_DIR)}")

reports_zip = BASE_OUT / "experiment_reports.zip"
report_exts = {".json", ".csv", ".txt", ".png", ".yaml"}
n_rep = 0
with zipfile.ZipFile(reports_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for fpath in sorted(RUNS_DIR.rglob("*")):
        if fpath.is_file() and fpath.suffix.lower() in report_exts:
            zf.write(fpath, arcname=str(fpath.relative_to(RUNS_DIR)))
            n_rep += 1

disk = shutil.disk_usage(BASE_OUT)
summary_zips = pd.DataFrame([
    ("ui_handoff_bundle.zip", round(ui_zip.stat().st_size / (1024**2), 2), "Paket Web Cloud (.keras + inference_api.py) & Mobile (.tflite)"),
    ("experiment_reports.zip", round(reports_zip.stat().st_size / (1024**2), 2), f"{n_rep} file metrik, tabel, kurva GD, & Grad-CAM"),
    ("Free Disk Remaining", round(disk.free / (1024**3), 2), "Sisa ruang /kaggle/working (GB)"),
], columns=["Artifact", "Size (MB / GB)", "Description"])
ipy_display(summary_zips)
"""))

    return cells


def main() -> None:
    cells = build_cells()
    for idx, cell in enumerate(cells):
        cell["id"] = f"cell-{idx:02d}"

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "mimetype": "text/x-python",
                "file_extension": ".py",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)

    n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"Notebook written: {OUT_PATH} ({len(cells)} cells: {n_md} markdown, {n_code} code)")


if __name__ == "__main__":
    main()
