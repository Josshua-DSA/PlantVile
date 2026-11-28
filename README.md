# PlantVile — Plant Disease Classification (Mobile-Edge CNN, TFLite PTQ & Grad-CAM)

Proyek UAS Neuro Computing untuk klasifikasi **15 kelas penyakit daun tanaman** (Tomato, Pepper, Potato) pada dataset **PlantVillage** (`20.638` citra valid → **`20.624` citra unik** setelah deduplikasi MD5 *in-memory*, split *anti-leakage* `group_same_leaf=true` 70/15/15: `14.434` train / `3.095` val / `3.095` test).

Eksperimen membandingkan 3 arsitektur CNN ringan (*mobile-edge*): **`MobileNetV2`**, **`EfficientNet-B0`**, dan **`MobileNetV3-Large`**, disertai optimasi **TensorFlow Lite Post-Training Quantization (`FP32`, `FP16`, `INT8`)** dan visualisasi **Grad-CAM**. Fokus pengembangan aplikasi saat ini adalah **Web / Localhost** (`.keras` FP32 + Grad-CAM atau `.tflite` FP16 hemat RAM), sedangkan pengembangan ***on-device* Mobile (`.tflite`) bersifat opsional / tidak wajib**.

* **Dataset:** [PlantVillage (`emmarex/plantdisease`)](https://www.kaggle.com/datasets/emmarex/plantdisease)
* **Framework:** `TensorFlow 2.21.0` / `Keras 3.15.1`
* **Protokol Eksperimen:** [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)
* **Konfigurasi Eksperimen:** [docs/EXPERIMENT_CONFIGURATION.md](docs/EXPERIMENT_CONFIGURATION.md)
* **Laporan Audit & EDA:** [docs/DATA_AUDIT_AND_EDA.md](docs/DATA_AUDIT_AND_EDA.md)
* **Laporan Evaluasi & Seleksi Model:** [docs/MODEL_EVALUATION_AND_SELECTION.md](docs/MODEL_EVALUATION_AND_SELECTION.md)
* **Paket Deliverable & Panduan Inference:** [results/README.md](results/README.md)

---

## 1. Ringkasan Hasil Evaluasi & Pemilihan Model (`test`, `n = 3.095`)

| Skenario Penggunaan | Model & Format | Ukuran | `test F1-Macro` [`95% CI`] | Latency CPU (`p50`) | Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Web / Localhost + Grad-CAM** *(rekomendasi utama)* | `m4b_mobilenetv3l_ft` (`keras_fp32`) | `13,04 MB` | **`0,9940`** `[0,9912, 0,9965]` | `23,80 ms` | `Deployment Candidate` |
| **Web Hemat RAM** *(tanpa Grad-CAM, opsional Mobile FP16)* | `m4b_mobilenetv3l_ft` (`tflite_fp16`) | `6,24 MB` | **`0,9943`** `[0,9916, 0,9968]` | **`4,13 ms`** | `Deployment Candidate` |
| **Opsional Edge Ultra-Ringan (`INT8`)** | `m2_mobilenetv2_ft` (`tflite_int8`) | **`2,93 MB`** | `0,9864` `[0,9820, 0,9904]` | `5,82 ms` | `Deployment Candidate` |

* **Catatan `m3b_effnetb0_ft`:** Berstatus `Meets Evaluation Criteria` (`keras_fp32` `F1-Macro = 0,9917`; secara statistik tidak berbeda signifikan dari `m4b` pada uji Exact McNemar `p = 0,5716`), namun memiliki ukuran model lebih besar dan latensi inferensi lebih tinggi dibanding `m4b`.
* **Batasan Domain:** Seluruh metrik di atas diukur pada citra laboratorium PlantVillage 15 kelas; performa pada foto lapangan (*in-situ* dengan pencahayaan dan latar alami) belum diukur.

> [!CAUTION]
> **JANGAN GUNAKAN** `m4b_mobilenetv3l_ft_int8.tflite` (`F1 = 0,4428`, status `Not Accepted`) maupun `m3b_effnetb0_ft_int8.tflite` (`F1 = 0,9467`, status `Not Accepted`) karena mengalami degradasi parah saat kuantisasi INT8. Jika membutuhkan format `INT8`, gunakan **hanya `m2_mobilenetv2_ft_int8.tflite`**.

---

## 2. Struktur Proyek

```text
PlantVile/
├── main.py                                   # Entrypoint CLI (wizard / YAML / --check / --eval / --quiet)
├── EXPERIMENT_PROTOCOL.md                    # Protokol eksperimen 3 arsitektur mobile-edge & TFLite PTQ
├── requirements.txt                          # Dependensi runtime & evaluasi statistik
├── configs/
│   ├── default.yaml                          # Konfigurasi default CPU lokal
│   └── kaggle_t4x2.yaml                      # Preset GPU Kaggle T4×2 (mixed_float16, batch_size=64)
├── docs/
│   ├── DATA_AUDIT_AND_EDA.md                 # Laporan audit integritas & EDA 15 kelas
│   ├── EXPERIMENT_CONFIGURATION.md           # Dokumentasi lengkap konfigurasi eksperimen CV
│   └── MODEL_EVALUATION_AND_SELECTION.md     # Laporan evaluasi final test set & keputusan seleksi model
├── src/
│   ├── config.py                             # SPEC tunggal parameter, YAML/CLI override, & validasi
│   ├── data.py                               # Scan dataset, MD5 dedup, group_same_leaf split, tf.data
│   ├── model.py                              # simple_cnn + 5 backbone (VGG16, ResNet50, MobileNetV2, MobileNetV3-Large, EfficientNet-B0)
│   ├── train.py                              # Loop training, callbacks, & StepHistoryLogger (per step CSV)
│   ├── evaluate.py                           # Metrik, ECE kalibrasi, confused pairs, Grad-CAM, & kurva GD
│   ├── gradcam.py                            # Modul reusable Grad-CAM overlay
│   └── utils.py                              # Seed, device/MirroredStrategy, IO
├── scripts/
│   ├── audit_dataset.py                      # Audit read-only (integritas PIL, MD5 duplikat, imbalance)
│   ├── build_eda_notebook.py                 # Generator notebook/data_audit_eda.ipynb
│   ├── build_kaggle_notebook.py              # Generator notebook/kaggle_experiment_runner.ipynb
│   ├── build_notebook.py                     # Generator notebook/plantvillage_cnn.ipynb
│   ├── plot_gradient_descent.py              # Plot gabungan Stage 1→2, overlay Stage 2, & kurva per step
│   ├── export_tflite.py                      # Ekspor pure FP32 .keras, TFLite FP32/FP16/INT8, McNemar/CI, & bundle
│   └── smoke_test.py                         # End-to-end CPU smoke test terisolasi
├── notebook/
│   ├── data_audit_eda.ipynb                  # Notebook audit data & EDA
│   ├── kaggle_experiment_runner.ipynb        # Notebook 1-klik Kaggle (m1–m4b + test eval + PTQ + ZIP)
│   └── plantvillage_cnn.ipynb                # Notebook eksperimen CNN interaktif
├── tests/
│   └── test_error_analysis.py                # Unit & regression tests (pytest)
├── data/                                     # Dataset lokal (diabaikan Git)
└── results/                                  # Paket deliverable & dokumentasi hasil
    ├── README.md                             # [Dilacak Git] Panduan paket deliverable & batasan scope
    ├── artifacts/
    │   ├── README.md                         # [Dilacak Git] Ringkasan artefak & tabel pilihan cepat
    │   ├── INFERENCE_PIPELINE.md             # [Dilacak Git] Spesifikasi pipeline inferensi 7 langkah
    │   ├── MODEL_CARD.md                     # [Dilacak Git] Metrik per spesies, kalibrasi, & disclaimer UI
    │   ├── MOBILE_DEPLOYMENT.md              # [Dilacak Git] Panduan mobile (opsional / tidak wajib)
    │   ├── labels.txt                        # [Dilacak Git] Daftar 15 label kelas (indeks 0–14)
    │   ├── class_names.json                  # [Dilacak Git] Daftar nama kelas format JSON
    │   ├── model_metadata.json               # [Dilacak Git] Parameter input shape, normalisasi, & Grad-CAM
    │   └── models/                           # [Diabaikan Git] Bobot .keras & .tflite
    ├── reports/
    │   ├── REPORTS_INDEX.md                  # [Dilacak Git] Peta indeks laporan eksperimen
    │   └── ...                               # [Diabaikan Git] Grafik PNG, CSV metrik, & JSON detail
    ├── runs/                                 # [Diabaikan Git] Checkpoint training mentah internal
    └── plantvile_deliverable.zip             # [Diabaikan Git] Arsip deliverable lengkap untuk distribusi tim
```

---

## 3. Paket Deliverable & Panduan Integrasi (`results/`)

Pengembangan aplikasi difokuskan pada **Web / Localhost** dengan pengembang backend/UI menulis kode implementasi secara mandiri di repositori aplikasi (tanpa meng-`import` skrip apa pun dari folder `results/`). Pengembangan **Mobile bersifat opsional / tidak wajib**.

Dokumentasi dan metadata teks ringan berikut dapat dibaca langsung melalui GitHub tanpa mengunduh arsip besar:
* [results/README.md](results/README.md) — Peta utama paket deliverable, ringkasan pilihan model, & batasan *scope*.
* [results/artifacts/README.md](results/artifacts/README.md) — Ringkasan isi artefak model & peringatan kuantisasi INT8.
* [results/artifacts/INFERENCE_PIPELINE.md](results/artifacts/INFERENCE_PIPELINE.md) — Spesifikasi *end-to-end inference pipeline* 7 langkah (*framework-agnostic*) & pemetaan 15 label kelas UI.
* [results/artifacts/MODEL_CARD.md](results/artifacts/MODEL_CARD.md) — Rincian metrik evaluasi `test`, analisis iris, & teks *disclaimer* UI.
* [results/artifacts/MOBILE_DEPLOYMENT.md](results/artifacts/MOBILE_DEPLOYMENT.md) — *(Opsional / Tidak Wajib)* Kontrak tensor mobile & panduan ADB `benchmark_model`.
* [results/reports/REPORTS_INDEX.md](results/reports/REPORTS_INDEX.md) — Peta seluruh grafik PNG dan tabel CSV laporan eksperimen.

Metadata ringan ([`labels.txt`](results/artifacts/labels.txt), [`class_names.json`](results/artifacts/class_names.json), [`model_metadata.json`](results/artifacts/model_metadata.json)) ikut dilacak di Git, sedangkan bobot model (`.keras`, `.tflite`), laporan visual lengkap, dan arsip distribusi `results/plantvile_deliverable.zip` dibagikan secara terpisah di luar Git.

---

## 4. Cara Menjalankan di Kaggle (Reproduksi Eksperimen — 1 Klik *Save & Run All*)

1. Buka Kaggle Notebook baru, set **Accelerator = `GPU T4 x2`** dan **Internet = `ON`**.
2. Tambahkan input dataset **`emmarex/plantdisease`**.
3. Impor atau jalankan **`notebook/kaggle_experiment_runner.ipynb`** (branch `main`).
4. Klik ***Save & Run All*** — notebook akan otomatis menjalankan:
   * Audit dataset & *pre-flight check* (`--check --quiet`),
   * Training terkontrol 2-stage untuk `MobileNetV2` (`m1`/`m2`), `EfficientNet-B0` (`m3a`/`m3b`), dan `MobileNetV3-Large` (`m4a`/`m4b`) dengan pencatatan `step_history.csv`,
   * Leaderboard `val` + grafik *gradient descent* gabungan & per step,
   * Evaluasi `test` otomatis 1× untuk seluruh kandidat Stage 2 + Confusion Matrix, kalibrasi `ECE`, & Grad-CAM,
   * Ekspor Pure FP32 `.keras`, TFLite PTQ (`FP32`, `FP16`, `INT8`), uji McNemar & Bootstrap CI, serta pengemasan arsip mentah Kaggle (`ui_handoff_bundle.zip` dan `experiment_reports.zip`) yang kemudian disusun ke dalam `results/artifacts/` dan `results/reports/`.

---

## 5. Cara Menjalankan via CLI Lokal

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

# 6) Ekspor Pure FP32 .keras, TFLite (FP32/FP16/INT8), benchmark CPU, & paket artefak
& $PY -m scripts.export_tflite --runs-dir results/runs --data-root data/PlantVillage --eval-split val --build-handoff
```

---

## 6. Verifikasi & Pengujian Lokal

```powershell
# Unit & regression tests
.\.venv\Scripts\python.exe -m pytest tests\test_error_analysis.py -v

# End-to-end CPU smoke test (menguji audit, check, mini-train, step_history, Grad-CAM, & --eval)
.\.venv\Scripts\python.exe scripts\smoke_test.py
```
