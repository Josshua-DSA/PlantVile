"""Generate notebook/kaggle_experiment_runner.ipynb for Kaggle T4x2 GPU and CPU.

Run:
    python scripts/build_kaggle_notebook.py

The generated notebook uses subprocess.run(..., check=True) so failures stop
immediately, supports both Kaggle GPU T4x2 and Kaggle CPU, and organizes all
artifacts in /kaggle/working/ for clean UI & report downloads.
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
    """Construct the ordered cells for the Kaggle experiment runner notebook."""
    cells = []

    # ---------------------------------------------------------------- Cell 0
    cells.append(md("""
# PlantVile — Kaggle Experiment Runner (T4×2 GPU & CPU)

Notebook ini menjalankan seluruh alur eksperimen **PlantVile** di Kaggle secara terkontrol dan reproduktif dari repositori GitHub (`branch: experiment-arfi`).

### Checklist Pengaturan Kaggle (Wajib Sebelum Run)
1. **Accelerator:** Pilih **`GPU T4 x2`** (untuk training penuh) atau **`None` (CPU)** (untuk audit / uji ringan).
2. **Internet:** **`ON`** (diperlukan untuk `git clone` dari GitHub dan mengunduh bobot pretrained ImageNet).
3. **Add Input:** Tambahkan dataset PlantVillage (`emmarex/plantdisease`) sehingga ter-mount di `/kaggle/input/datasets/emmarex/plantdisease/PlantVillage`.

### Peta Input & Output per Tahap

| Cell | Tahap | Input Utama | Output Ditampilkan | Disimpan ke `/kaggle/working/` |
| :---: | :--- | :--- | :--- | :--- |
| **1** | Setup & Deteksi Hardware | `REPO_URL`, `BRANCH`, kandidat `DATA_ROOT` | Info GPU (`nvidia-smi`), TF/Keras, sisa disk, config preset | — |
| **2** | Clone / Pull GitHub | `REPO_URL`, `BRANCH`, `WORK_DIR` | Status git & 3 commit terakhir | `/kaggle/working/PlantVile/` |
| **3** | Audit Dataset | `DATA_ROOT` | Jumlah kelas, citra, imbalance, duplikat MD5, corrupt check | `/kaggle/working/runs/dataset_audit.json` |
| **4** | Pre-Flight `--check` & Pilot | `CONFIG_PATH`, `RUN_PILOT` | Validasi config, ringkasan model, shape batch (+ pilot 2 epoch opsional) | `/kaggle/working/runs/pilot_smoke/` *(jika `RUN_PILOT=True`)* |
| **5** | Registry Eksperimen | `EXP_REGISTRY`, `SELECTED_RUNS` | Tabel rencana eksperimen & pratinjau CLI | — |
| **6** | Training + Error Analysis | `SELECTED_RUNS`, `RUN_EXPERIMENTS` | Log epoch, durasi, metrik `val`, top confused pairs | `/kaggle/working/runs/<run_id>/` (`.keras`, CSV, JSON, 6 PNG) |
| **7** | Leaderboard & Multi-Seed | Semua `*/val_metrics.json` | Ranking `val f1_macro`, `Mean ± SD`, grafik perbandingan | `experiment_leaderboard.csv`, `experiment_leaderboard.png` |
| **8** | Inspeksi Visual & Grad-CAM | `INSPECT_RUN` | Kurva, kalibrasi `ECE`, confusion matrix, misclassified, Grad-CAM | — *(inline display)* |
| **9** | Evaluasi Final 1× (`test`) | `FINAL_RUN_NAME`, `RUN_FINAL_TEST` | Metrik final `test`, confusion matrix, kalibrasi, Grad-CAM | `/kaggle/working/runs/<FINAL_RUN_NAME>/reload_*` |
| **10** | Pengemasan Unduhan ZIP | `BUNDLE_RUN_NAME`, `BUILD_ZIPS` | Daftar isi & ukuran MB paket ZIP | `ui_model_bundle_<run>.zip` & `experiment_reports.zip` |
"""))

    # ---------------------------------------------------------------- Cell 1
    cells.append(md("""
## Cell 1 — Setup Global & Deteksi Hardware

- **Input:** `REPO_URL`, `BRANCH`, kandidat `DATA_ROOT` (default Kaggle: `/kaggle/input/datasets/emmarex/plantdisease/PlantVillage`), `WORK_DIR`, `RUNS_DIR`.
- **Output Layar:** Status GPU (`nvidia-smi` + TensorFlow), versi Python/TF/Keras, sisa kuota disk `/kaggle/working`, serta path `DATA_ROOT` dan `CONFIG_PATH` yang otomatis dipilih (`configs/kaggle_t4x2.yaml` untuk GPU, `configs/default.yaml` untuk CPU).
- **Disimpan ke Output Kaggle:** Tidak ada.
"""))

    cells.append(code("""import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["PYTHONUNBUFFERED"] = "1"

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


def run_stream(cmd: list[str]) -> None:
    \"\"\"Run subprocess and stream combined stdout/stderr live into notebook cell output.\"\"\"
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


if shutil.which("nvidia-smi"):
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    )
    print("NVIDIA GPUs :", smi.stdout.strip() or "tidak terbaca")
else:
    print("NVIDIA GPUs : nvidia-smi tidak tersedia (CPU mode)")

import keras
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
HAS_GPU = len(gpus) > 0
CONFIG_PATH = "configs/kaggle_t4x2.yaml" if HAS_GPU else "configs/default.yaml"

disk = shutil.disk_usage(BASE_OUT)
print(f"Python      : {sys.version.split()[0]}")
print(f"TensorFlow  : {tf.__version__} | Keras: {keras.__version__}")
print(f"TF GPUs     : {[g.name for g in gpus] if gpus else 'CPU-only'}")
print(f"CONFIG_PATH : {CONFIG_PATH}")
print(f"DATA_ROOT   : {DATA_ROOT}")
print(f"RUNS_DIR    : {RUNS_DIR}")
print(f"Free Disk   : {disk.free / (1024**3):.2f} GB / {disk.total / (1024**3):.2f} GB")
"""))

    # ---------------------------------------------------------------- Cell 2
    cells.append(md("""
## Cell 2 — Clone / Pull Repository GitHub (`experiment-arfi`)

- **Input:** `REPO_URL`, `BRANCH`, `WORK_DIR`.
- **Output Layar:** Progress `git clone` / `git pull` dan 3 commit terakhir (`git log -n 3 --oneline`) untuk memastikan versi kode yang berjalan.
- **Disimpan ke Output Kaggle:** Direktori source code `/kaggle/working/PlantVile/`.
"""))

    cells.append(code("""if (WORK_DIR / ".git").is_dir():
    print(f"Updating existing repo di {WORK_DIR}...")
    run_stream(["git", "-C", str(WORK_DIR), "fetch", "origin", BRANCH])
    run_stream(["git", "-C", str(WORK_DIR), "checkout", BRANCH])
    run_stream(["git", "-C", str(WORK_DIR), "pull", "origin", BRANCH])
else:
    print(f"Cloning {REPO_URL} (branch={BRANCH}) ke {WORK_DIR}...")
    run_stream(["git", "clone", "-b", BRANCH, "--single-branch", REPO_URL, str(WORK_DIR)])

os.chdir(WORK_DIR)
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

run_stream(["git", "log", "-n", "3", "--oneline"])
"""))

    # ---------------------------------------------------------------- Cell 3
    cells.append(md("""
## Cell 3 — Audit Dataset Kaggle (Read-Only)

- **Input:** `DATA_ROOT` dan `RUNS_DIR`.
- **Output Layar:** Ringkasan jumlah kelas, total citra valid, rasio *imbalance*, subfolder bersarang yang diabaikan (`skipped_subdirs`), file non-citra, grup duplikat MD5, serta hasil verifikasi integritas PIL.
- **Disimpan ke Output Kaggle:** `/kaggle/working/runs/dataset_audit.json`.
"""))

    cells.append(code("""import pandas as pd

audit_out = RUNS_DIR / "dataset_audit.json"
run_stream([sys.executable, "scripts/audit_dataset.py", "--root", str(DATA_ROOT), "--out", str(audit_out)])

audit_data = json.loads(audit_out.read_text(encoding="utf-8"))
summary_rows = [
    ("Resolved Dir", audit_data["resolved_dir"]),
    ("Classes", audit_data["classes"]),
    ("Valid Images", audit_data["valid_images"]),
    ("Total MB", audit_data["total_mb"]),
    ("Imbalance Ratio", f"{audit_data['imbalance_ratio']}x"),
    ("Skipped Subdirs", audit_data["skipped_subdirs"] or "none"),
    ("Non-Image Files", len(audit_data["non_image_files"])),
    ("MD5 Duplicate Groups", f"{audit_data['exact_md5_duplicates']['groups']} (within={audit_data['exact_md5_duplicates']['within_class_groups']}, cross={audit_data['exact_md5_duplicates']['cross_class_groups']})"),
    ("PIL Corrupt Count", audit_data["pil_verification"]["corrupt_count"]),
]
display(pd.DataFrame(summary_rows, columns=["Metric", "Value"]))
"""))

    # ---------------------------------------------------------------- Cell 4
    cells.append(md("""
## Cell 4 — Pre-Flight `--check` & Pilot Smoke Test

- **Input:** `CONFIG_PATH`, `DATA_ROOT`, dan flag `RUN_PILOT = False` (ubah ke `True` bila ingin mencoba mini training 2 epoch pada subset 200 gambar/kelas).
- **Output Layar:** Output validasi `main.py --check` (status config, device/strategy, ringkasan arsitektur model, shape & rentang piksel 1 batch) + durasi pilot jika `RUN_PILOT = True`.
- **Disimpan ke Output Kaggle:** Kosong jika hanya `--check`; `/kaggle/working/runs/pilot_smoke/` jika `RUN_PILOT = True`.
"""))

    cells.append(code("""RUN_PILOT = False  # Ubah ke True untuk uji mini training 2 epoch (200 gambar/kelas)

check_cmd = [
    sys.executable, "main.py", "--check",
    "--config", CONFIG_PATH,
    "--set", f"data.root={DATA_ROOT}",
]
if not HAS_GPU:
    check_cmd += ["--set", "runtime.device=cpu"]

run_stream(check_cmd)

if RUN_PILOT:
    t0 = time.time()
    pilot_cmd = [
        sys.executable, "main.py", "--yes",
        "--config", CONFIG_PATH,
        "--set", f"data.root={DATA_ROOT}",
        "--set", f"runtime.runs_dir={RUNS_DIR}",
        "--set", "runtime.run_name=pilot_smoke",
        "--set", "data.max_per_class=200",
        "--set", "train.epochs=2",
        "--set", "eval.eval_split=val",
        "--set", "runtime.verbose=2",
    ]
    if not HAS_GPU:
        pilot_cmd += ["--set", "runtime.device=cpu"]
    run_stream(pilot_cmd)
    print(f"Pilot selesai dalam {time.time() - t0:.1f} detik.")
"""))

    # ---------------------------------------------------------------- Cell 5
    cells.append(md("""
## Cell 5 — Registry Matriks Eksperimen (`EXP_REGISTRY`)

- **Input:** Daftar `SELECTED_RUNS`, `RUN_EXPERIMENTS = False`, `SKIP_IF_DONE = True`, dan `STRATEGY_MODE = "auto"` (`auto` = `MirroredStrategy` di T4×2; `none` = 1 GPU).
- **Output Layar:** Tabel daftar eksperimen yang dipilih, variabel bebas yang diubah, status (`DONE` / `PENDING`), dan pratinjau perintah CLI.
- **Disimpan ke Output Kaggle:** Tidak ada.
"""))

    cells.append(code("""EXP_REGISTRY = {
    "exp00a_simple_cnn": {
        "desc": "Scratch CNN baseline (no augment)",
        "overrides": ["model.arch=simple_cnn", "data.augment=false", "data.normalize=div255"],
    },
    "exp00b_mobilenet_ctrl": {
        "desc": "Transfer baseline MobileNetV2 (no augment)",
        "overrides": ["model.arch=mobilenetv2", "data.augment=false"],
    },
    "exp01_aug": {
        "desc": "Track A: + Data augmentation",
        "overrides": ["model.arch=mobilenetv2", "data.augment=true"],
    },
    "exp02a_noweight": {
        "desc": "Track B1: class_weight=none",
        "overrides": ["model.arch=mobilenetv2", "data.augment=true", "train.class_weight=none"],
    },
    "exp02b_focal": {
        "desc": "Track B2: loss=focal + class_weight=none",
        "overrides": ["model.arch=mobilenetv2", "data.augment=true", "train.class_weight=none", "train.loss=focal"],
    },
    "exp03_res224": {
        "desc": "Track C: Resolusi input 224x224",
        "overrides": ["model.arch=mobilenetv2", "data.img_height=224", "data.img_width=224"],
    },
    "exp04a_resnet50": {
        "desc": "Track D1: Backbone ResNet50 (224x224)",
        "overrides": ["model.arch=resnet50", "data.img_height=224", "data.img_width=224"],
    },
    "exp04b_effnetb0": {
        "desc": "Track D2: Backbone EfficientNetB0 (224x224)",
        "overrides": ["model.arch=efficientnetb0", "data.img_height=224", "data.img_width=224"],
    },
    "exp05_finetune": {
        "desc": "Track E: Fine-tune MobileNetV2 (>=100) + AdamW",
        "overrides": [
            "model.arch=mobilenetv2", "data.img_height=224", "data.img_width=224",
            "model.freeze_base=true", "model.fine_tune_at=100",
            "train.optimizer=adamw", "train.learning_rate=0.0001", "train.epochs=30",
        ],
    },
    "exp06_seed123": {
        "desc": "Track F: Multi-seed stability (seed=123)",
        "overrides": ["model.arch=mobilenetv2", "data.img_height=224", "data.img_width=224", "runtime.seed=123"],
    },
    "exp06_seed2024": {
        "desc": "Track F: Multi-seed stability (seed=2024)",
        "overrides": ["model.arch=mobilenetv2", "data.img_height=224", "data.img_width=224", "runtime.seed=2024"],
    },
}

# --- Kontrol Eksekusi ---
RUN_EXPERIMENTS = False  # <-- Ubah ke True untuk menjalankan training
SKIP_IF_DONE = True      # <-- Lewati run yang sudah punya val_metrics.json
STRATEGY_MODE = "auto"   # "auto" (MirroredStrategy T4x2) | "none" (1 GPU)
SELECTED_RUNS = [
    "exp00a_simple_cnn",
    "exp00b_mobilenet_ctrl",
    "exp01_aug",
]


def build_run_command(run_id: str) -> list[str]:
    spec = EXP_REGISTRY[run_id]
    cmd = [
        sys.executable, "main.py", "--yes",
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
    for ov in spec["overrides"]:
        cmd += ["--set", ov]
    return cmd


preview_rows = []
for rid in SELECTED_RUNS:
    assert rid in EXP_REGISTRY, f"Run ID tidak dikenal: {rid}"
    done = (RUNS_DIR / rid / "val_metrics.json").is_file()
    preview_rows.append({
        "run_id": rid,
        "status": "DONE" if done else "PENDING",
        "description": EXP_REGISTRY[rid]["desc"],
        "overrides": ", ".join(EXP_REGISTRY[rid]["overrides"]),
    })

display(pd.DataFrame(preview_rows))
print("RUN_EXPERIMENTS =", RUN_EXPERIMENTS)
"""))

    # ---------------------------------------------------------------- Cell 6
    cells.append(md("""
## Cell 6 — Eksekusi Training + Evaluasi `val` + Error Analysis + Grad-CAM

- **Input:** `SELECTED_RUNS`, `EXP_REGISTRY`, `CONFIG_PATH`, `DATA_ROOT`, `RUNS_DIR` (otomatis mengunci `eval.eval_split=val`).
- **Output Layar:** Progress tiap epoch, ringkasan metrik `val`, `ECE`, dan top-10 *confused pairs* beserta waktu eksekusi per eksperimen.
- **Disimpan ke Output Kaggle (`/kaggle/working/runs/<run_id>/`):**
  - Model & Config: `best.keras`, `final.keras`, `config_used.yaml`, `config_used.json`, `class_names.json`, `device.json`, `history.csv`, `history.json`
  - Metrik & Error Analysis: `val_metrics.json`, `val_per_class.json/csv`, `val_classification_report.txt`, `val_confused_pairs.json/csv`, `val_misclassified.csv`
  - Visualisasi PNG: `val_loss_accuracy.png`, `val_confusion_matrix.png`, `val_sample_predictions.png`, `val_confidence_calibration.png`, `val_misclassified_grid.png`, `val_gradcam_grid.png`
"""))

    cells.append(code("""if not RUN_EXPERIMENTS:
    print("Training dilewati (RUN_EXPERIMENTS = False). Ubah ke True di Cell 5 untuk memulai.")
else:
    for rid in SELECTED_RUNS:
        done_marker = RUNS_DIR / rid / "val_metrics.json"
        if SKIP_IF_DONE and done_marker.is_file():
            print(f"[SKIP] {rid} sudah memiliki {done_marker}")
            continue
        cmd = build_run_command(rid)
        print(f"\\n{'=' * 72}\\n[RUN] {rid} — {EXP_REGISTRY[rid]['desc']}\\n{'=' * 72}")
        t0 = time.time()
        run_stream(cmd)
        print(f"[DONE] {rid} selesai dalam {(time.time() - t0) / 60:.2f} menit.")
"""))

    # ---------------------------------------------------------------- Cell 7
    cells.append(md("""
## Cell 7 — Leaderboard & Agregasi Multi-Seed (`val`)

- **Input:** Seluruh file `val_metrics.json`, `history.json`, dan `best.keras` di dalam `RUNS_DIR`.
- **Output Layar:** Tabel ranking eksperimen berdasarkan `val f1_macro` (`accuracy`, `top_k_accuracy`, *worst-class recall*, `ECE`, *train–val gap*, ukuran `best.keras`), ringkasan `Mean ± SD` jika ada run `exp06_*`, serta grafik perbandingan.
- **Disimpan ke Output Kaggle:** `/kaggle/working/runs/experiment_leaderboard.csv` dan `/kaggle/working/runs/experiment_leaderboard.png`.
"""))

    cells.append(code("""import matplotlib.pyplot as plt
import numpy as np

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
        if "accuracy" in h and "val_accuracy" in h and h["accuracy"]:
            gap = h["accuracy"][-1] - h["val_accuracy"][-1]

    best_keras = rdir / "best.keras"
    size_mb = best_keras.stat().st_size / (1024 * 1024) if best_keras.is_file() else np.nan

    rows.append({
        "run_id": rid,
        "f1_macro": round(float(m.get("f1_macro", 0.0)), 4),
        "accuracy": round(float(m.get("accuracy", 0.0)), 4),
        "top_k_acc": round(float(m.get("top_k_accuracy", 0.0)), 4),
        "worst_recall": round(float(worst_recall), 4) if not np.isnan(worst_recall) else np.nan,
        "ece": round(float(m.get("calibration", {}).get("ece", np.nan)), 4),
        "misclassified": m.get("num_misclassified", np.nan),
        "train_val_gap": round(float(gap), 4) if not np.isnan(gap) else np.nan,
        "best_keras_mb": round(float(size_mb), 2) if not np.isnan(size_mb) else np.nan,
    })

if not rows:
    df_board = pd.DataFrame()
    print("Belum ada hasil val_metrics.json di", RUNS_DIR)
else:
    df_board = pd.DataFrame(rows).sort_values(["f1_macro", "accuracy"], ascending=False).reset_index(drop=True)
    df_board.to_csv(RUNS_DIR / "experiment_leaderboard.csv", index=False)
    display(df_board)

    ms_rows = df_board[df_board["run_id"].str.startswith("exp06_")]
    if len(ms_rows) >= 2:
        print(
            f"\\nMulti-Seed Summary ({len(ms_rows)} runs): "
            f"f1_macro = {ms_rows['f1_macro'].mean():.4f} ± {ms_rows['f1_macro'].std(ddof=1):.4f} | "
            f"accuracy = {ms_rows['accuracy'].mean():.4f} ± {ms_rows['accuracy'].std(ddof=1):.4f}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, 0.55 * len(df_board) + 1.5)))
    plot_df = df_board.iloc[::-1]
    axes[0].barh(plot_df["run_id"], plot_df["f1_macro"], color="#2f7d5d")
    axes[0].set_title("Validation Macro F1 per Run")
    axes[0].set_xlabel("val f1_macro")
    axes[0].set_xlim(0, 1.05)

    axes[1].barh(plot_df["run_id"], plot_df["ece"].fillna(0.0), color="#c44e52")
    axes[1].set_title("Expected Calibration Error (ECE, lebih kecil lebih baik)")
    axes[1].set_xlabel("ECE (10 bins)")

    plt.tight_layout()
    board_png = RUNS_DIR / "experiment_leaderboard.png"
    plt.savefig(board_png, dpi=140, bbox_inches="tight")
    plt.show()
    plt.close(fig)
"""))

    # ---------------------------------------------------------------- Cell 8
    cells.append(md("""
## Cell 8 — Inspeksi Visual Error Analysis & Grad-CAM

- **Input:** `INSPECT_RUN` (biarkan kosong `""` untuk otomatis memilih peringkat 1 leaderboard, atau isi dengan nama run tertentu).
- **Output Layar:** Menampilkan langsung grafik kurva training, *reliability diagram*, *confusion matrix*, tabel top-10 *confused pairs* & *high-confidence misclassified*, grid salah prediksi, dan grid Grad-CAM.
- **Disimpan ke Output Kaggle:** Tidak ada (membaca artefak dari Cell 6).
"""))

    cells.append(code("""INSPECT_RUN = ""  # kosong = otomatis run terbaik di leaderboard

board_df = globals().get("df_board", None)
target_run = INSPECT_RUN or (board_df.iloc[0]["run_id"] if board_df is not None and len(board_df) else "")
if not target_run:
    print("Belum ada run untuk diinspeksi.")
else:
    rdir = RUNS_DIR / target_run
    print(f"=== Inspeksi Visual: {target_run} ===")

    for csv_name, title in [
        ("val_confused_pairs.csv", "Top-10 Confused Pairs"),
        ("val_misclassified.csv", "Top-10 High-Confidence Misclassified"),
    ]:
        cpath = rdir / csv_name
        if cpath.is_file():
            print(f"\\n--- {title} ---")
            ipy_display(pd.read_csv(cpath).head(10))

    for png_name in [
        "val_loss_accuracy.png",
        "val_confidence_calibration.png",
        "val_confusion_matrix.png",
        "val_misclassified_grid.png",
        "val_gradcam_grid.png",
    ]:
        ppath = rdir / png_name
        if ppath.is_file():
            print(f"\\n>>> {png_name}")
            ipy_display(IPyImage(filename=str(ppath)))
"""))

    # ---------------------------------------------------------------- Cell 9
    cells.append(md("""
## Cell 9 — Evaluasi Final 1× pada Split `test`

> **Aturan Protokol:** Jalankan cell ini **hanya satu kali** setelah seluruh eksperimen dan pemilihan model di split `val` selesai.

- **Input:** `RUN_FINAL_TEST = False` (ubah ke `True` jika siap) dan `FINAL_RUN_NAME = "<run_id_terpilih>"`.
- **Output Layar:** Metrik final pada split `test` (`accuracy`, `top_k_accuracy`, `f1_macro`, `ECE`, *worst-class recall*) + plot final.
- **Disimpan ke Output Kaggle (`/kaggle/working/runs/<FINAL_RUN_NAME>/`):** `reload_metrics.json`, `reload_per_class.{json,csv}`, `reload_classification_report.txt`, `reload_confused_pairs.{json,csv}`, `reload_misclassified.csv`, serta `reload_*.png`.
"""))

    cells.append(code("""RUN_FINAL_TEST = False   # <-- Ubah ke True hanya untuk 1x evaluasi test akhir
FINAL_RUN_NAME = ""      # <-- Isi eksplisit, misal: "exp05_finetune"

if not RUN_FINAL_TEST:
    print("Evaluasi final test dilewati (RUN_FINAL_TEST = False).")
else:
    assert FINAL_RUN_NAME, "Isi FINAL_RUN_NAME terlebih dahulu sebelum menjalankan evaluasi test!"
    final_dir = RUNS_DIR / FINAL_RUN_NAME
    best_model_path = final_dir / "best.keras"
    cfg_used_path = final_dir / "config_used.yaml"
    assert best_model_path.is_file() and cfg_used_path.is_file(), f"Artefak tidak lengkap di {final_dir}"

    eval_cmd = [
        sys.executable, "main.py",
        "--eval", str(best_model_path),
        "--config", str(cfg_used_path),
        "--set", f"data.root={DATA_ROOT}",
        "--set", "eval.eval_split=test",
    ]
    if not HAS_GPU:
        eval_cmd += ["--set", "runtime.device=cpu"]

    run_stream(eval_cmd)

    reload_metrics = json.loads((final_dir / "reload_metrics.json").read_text(encoding="utf-8"))
    print(
        f"\\n[FINAL TEST] {FINAL_RUN_NAME} -> "
        f"accuracy={reload_metrics['accuracy']:.4f} | "
        f"f1_macro={reload_metrics['f1_macro']:.4f} | "
        f"top_{reload_metrics['top_k']}={reload_metrics['top_k_accuracy']:.4f} | "
        f"ECE={reload_metrics['calibration']['ece']:.4f}"
    )
    for png_name in ["reload_confusion_matrix.png", "reload_confidence_calibration.png", "reload_gradcam_grid.png"]:
        ppath = final_dir / png_name
        if ppath.is_file():
            ipy_display(IPyImage(filename=str(ppath)))
"""))

    # ---------------------------------------------------------------- Cell 10
    cells.append(md("""
## Cell 10 — Pengemasan Artefak Siap Unduh (UI Bundle & Laporan)

- **Input:** `BUNDLE_RUN_NAME` (default otomatis dari `FINAL_RUN_NAME` atau peringkat 1 leaderboard), `BUILD_ZIPS = True`, dan `PRUNE_EXTRA_FINAL_KERAS = False`.
- **Output Layar:** Daftar isi, ukuran (MB) kedua arsip ZIP, dan sisa ruang `/kaggle/working`.
- **Disimpan ke Output Kaggle (`/kaggle/working/`):**
  1. `ui_model_bundle_<run>.zip`: paket minimal siap pakai aplikasi/UI (`best.keras`, `class_names.json`, `config_used.json`, `config_used.yaml`, `val_metrics.json`, dan `reload_metrics.json` bila ada).
  2. `experiment_reports.zip`: seluruh laporan audit, leaderboard, JSON, CSV, TXT, dan PNG grafik eksperimen (tanpa `.keras`/`.h5` agar ringan diunduh).
"""))

    cells.append(code("""import zipfile

BUILD_ZIPS = True
PRUNE_EXTRA_FINAL_KERAS = False  # True = hapus final.keras milik run lain jika disk mepet
board_df = globals().get("df_board", None)
BUNDLE_RUN_NAME = globals().get("FINAL_RUN_NAME", "") or (
    board_df.iloc[0]["run_id"] if board_df is not None and len(board_df) else ""
)

if PRUNE_EXTRA_FINAL_KERAS and BUNDLE_RUN_NAME:
    for fk in RUNS_DIR.glob("*/final.keras"):
        if fk.parent.name != BUNDLE_RUN_NAME:
            fk.unlink()
            print("Removed extra:", fk)

if not BUILD_ZIPS:
    print("Pengemasan ZIP dilewati (BUILD_ZIPS = False).")
else:
    if BUNDLE_RUN_NAME:
        bdir = RUNS_DIR / BUNDLE_RUN_NAME
        ui_zip = BASE_OUT / f"ui_model_bundle_{BUNDLE_RUN_NAME}.zip"
        ui_files = [
            "best.keras", "class_names.json", "config_used.json",
            "config_used.yaml", "val_metrics.json", "reload_metrics.json",
        ]
        with zipfile.ZipFile(ui_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fname in ui_files:
                fpath = bdir / fname
                if fpath.is_file():
                    zf.write(fpath, arcname=f"{BUNDLE_RUN_NAME}/{fname}")
        print(f"UI bundle ditulis : {ui_zip} ({ui_zip.stat().st_size / (1024**2):.2f} MB)")

    reports_zip = BASE_OUT / "experiment_reports.zip"
    report_exts = {".json", ".csv", ".txt", ".png", ".yaml"}
    n_files = 0
    with zipfile.ZipFile(reports_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fpath in sorted(RUNS_DIR.rglob("*")):
            if fpath.is_file() and fpath.suffix.lower() in report_exts:
                zf.write(fpath, arcname=str(fpath.relative_to(RUNS_DIR)))
                n_files += 1
    print(f"Reports ZIP ditulis: {reports_zip} ({reports_zip.stat().st_size / (1024**2):.2f} MB, {n_files} files)")

disk = shutil.disk_usage(BASE_OUT)
print(f"Sisa disk {BASE_OUT}: {disk.free / (1024**3):.2f} GB")
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
