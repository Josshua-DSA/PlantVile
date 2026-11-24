# PlantVile — Plant Disease Classification (Mobile-Edge CNN, TFLite PTQ & Grad-CAM)

Proyek UAS Neuro Computing untuk klasifikasi **15 kelas penyakit daun tanaman** (Tomato, Pepper, Potato) pada dataset **PlantVillage** (`20.638` citra valid → **`20.624` citra** setelah deduplikasi MD5 *in-memory*, split *anti-leakage* `group_same_leaf=true` 70/15/15).

Eksperimen difokuskan pada 3 arsitektur CNN ringan (*mobile-edge*): **`MobileNetV2`**, **`EfficientNet-B0`**, dan **`MobileNetV3-Large`**, optimasi **TensorFlow Lite Post-Training Quantization (`FP32`, `FP16`, `INT8`)**, serta kesiapan integrasi ganda untuk **Backend Web Cloud (`.keras` + Grad-CAM)** dan ***On-Device* Mobile (`.tflite`)**.

* **Dataset:** [PlantVillage (`emmarex/plantdisease`)](https://www.kaggle.com/datasets/emmarex/plantdisease)
* **Framework:** TensorFlow `2.21.0` / Keras `3.15.1`
* **Protokol Eksperimen:** [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)
* **Laporan Audit & EDA:** [docs/DATA_AUDIT_AND_EDA.md](docs/DATA_AUDIT_AND_EDA.md)

---

## 1. Struktur Proyek

```text
PlantVile/
├── main.py                              # Entrypoint CLI (wizard / YAML / --check / --eval / --quiet)
├── EXPERIMENT_PROTOCOL.md               # Protokol eksperimen 3 arsitektur mobile-edge & TFLite PTQ
├── requirements.txt                     # Dependensi runtime & evaluasi statistik
├── configs/
│   ├── default.yaml                     # Konfigurasi default CPU lokal
│   └── kaggle_t4x2.yaml                 # Preset GPU Kaggle T4×2 (mixed_float16, batch_size=64)
├── docs/
│   └── DATA_AUDIT_AND_EDA.md            # Laporan audit integritas & EDA 15 kelas
├── src/
│   ├── config.py                        # SPEC tunggal parameter, YAML/CLI override, & validasi
│   ├── data.py                          # Scan dataset, MD5 dedup, group_same_leaf split, tf.data
│   ├── model.py                         # simple_cnn + 5 backbone (VGG16, ResNet50, MobileNetV2, MobileNetV3-Large, EfficientNet-B0)
│   ├── train.py                         # Loop training, callbacks, & StepHistoryLogger (per step CSV)
│   ├── evaluate.py                      # Metrik, ECE kalibrasi, confused pairs, Grad-CAM, & kurva GD
│   ├── gradcam.py                       # Modul reusable Grad-CAM overlay
│   └── utils.py                         # Seed, device/MirroredStrategy, IO
├── scripts/
│   ├── audit_dataset.py                 # Audit read-only (integritas PIL, MD5 duplikat, imbalance)
│   ├── build_eda_notebook.py            # Generator notebook/data_audit_eda.ipynb
│   ├── build_kaggle_notebook.py         # Generator notebook/kaggle_experiment_runner.ipynb
│   ├── plot_gradient_descent.py         # Plot gabungan Stage 1→2, overlay Stage 2, & kurva per step
│   ├── export_tflite.py                 # Ekspor pure FP32 .keras, TFLite FP32/FP16/INT8, McNemar/CI, & UI bundle
│   └── smoke_test.py                    # End-to-end CPU smoke test terisolasi
├── notebook/
│   ├── data_audit_eda.ipynb             # Notebook audit data & EDA
│   └── kaggle_experiment_runner.ipynb   # Notebook bersih 1-klik Kaggle (m1–m4b + test eval + PTQ + ZIP)
├── tests/
│   └── test_error_analysis.py           # Unit & regression tests (pytest)
├── data/                                # Dataset lokal (diabaikan Git)
└── results/                             # Output eksperimen & UI handoff bundle (diabaikan Git)
```

---

## 2. Cara Menjalankan di Kaggle (Direkomendasikan — 1 Klik *Save & Run All*)

1. Buka Kaggle Notebook baru, set **Accelerator = `GPU T4 x2`** dan **Internet = `ON`**.
2. Tambahkan input dataset **`emmarex/plantdisease`**.
3. Impor atau jalankan **`notebook/kaggle_experiment_runner.ipynb`** (branch `experiment-arfi`).
4. Klik ***Save & Run All*** — notebook akan otomatis menjalankan:
   * Audit dataset & *pre-flight check* (`--check --quiet`),
   * Training terkontrol 2-stage untuk `MobileNetV2` (`m1`/`m2`), `EfficientNet-B0` (`m3a`/`m3b`), dan `MobileNetV3-Large` (`m4a`/`m4b`) dengan pencatatan `step_history.csv`,
   * Leaderboard `val` + grafik *gradient descent* gabungan & per step,
   * Evaluasi `test` otomatis 1× untuk seluruh kandidat Stage 2 + Confusion Matrix, kalibrasi `ECE`, & Grad-CAM,
   * Ekspor Pure FP32 `.keras`, TFLite PTQ (`FP32`, `FP16`, `INT8`), uji McNemar & Bootstrap CI, serta pengemasan **`ui_handoff_bundle.zip`** dan **`experiment_reports.zip`**.

---

## 3. Cara Menjalankan via CLI Lokal

```powershell
# Aktivasi virtual environment (Windows) atau sesuaikan python environment
$PY = ".\.venv\Scripts\python.exe"

# 1) Validasi config + dataset + model (tanpa training)
& $PY main.py --check --config configs/default.yaml --set data.root=data/PlantVillage

# 2) Lihat seluruh parameter yang didukung
& $PY main.py --list-params

# 3) Training langsung dari YAML + override CLI (--quiet untuk log ringkas)
& $PY main.py --yes --quiet --config configs/default.yaml `
  --set data.root=data/PlantVillage --set model.arch=mobilenetv3_large `
  --set data.img_height=224 --set data.img_width=224 `
  --set train.optimizer=adamw --set train.learning_rate=0.0001 --set train.epochs=25

# 4) Evaluasi checkpoint tersimpan (otomatis memuat config_used.yaml di folder model)
& $PY main.py --eval results/runs/m2_mobilenetv2_ft/best.keras --set eval.eval_split=val

# 5) Buat grafik Gradient Descent gabungan & per step
& $PY -m scripts.plot_gradient_descent --runs-dir results/runs --out-dir results/gradient_descent_plots

# 6) Ekspor Pure FP32 .keras, TFLite (FP32/FP16/INT8), benchmark CPU, & UI Handoff Bundle
& $PY -m scripts.export_tflite --runs-dir results/runs --data-root data/PlantVillage --eval-split val --build-handoff
```

---

## 4. Verifikasi & Pengujian Lokal

```powershell
# Unit & regression tests
.\.venv\Scripts\python.exe -m pytest tests\test_error_analysis.py -v

# End-to-end CPU smoke test (menguji audit, check, mini-train, step_history, Grad-CAM, & --eval)
.\.venv\Scripts\python.exe scripts\smoke_test.py
```

---

## 5. Paket Serah Terima UI (`results/ui_handoff_bundle/`)

Setelah `scripts/export_tflite.py --build-handoff` dijalankan (atau diekstrak dari `ui_handoff_bundle.zip` Kaggle), direktori handoff berisi:
* `inference_api.py` — API mandiri `PlantDiseasePredictor(bundle_dir, backend="keras"|"tflite", quant="fp16")` (siap diimpor oleh FastAPI/Flask/Streamlit tanpa dependensi `src/`).
* `models/<run_id>/*_infer_fp32.keras` — model *pure* FP32 untuk backend web cloud + visualisasi **Grad-CAM**.
* `models/<run_id>/*_{fp32,fp16,int8}.tflite` — model ringan terkuantisasi untuk server hemat RAM maupun aplikasi mobile *on-device*.
* `labels.txt`, `class_names.json`, `model_metadata.json`, `UI_INTEGRATION_README.md`, dan `android_benchmark_guide.md`.
