# Experiment Protocol — PlantVile (Mobile-Edge CNN, TFLite PTQ & Dual-Target Deployment)

Dokumen ini adalah protokol eksperimen resmi untuk proyek **PlantVile** yang memadukan temuan audit/EDA dataset lokal ([docs/DATA_AUDIT_AND_EDA.md](docs/DATA_AUDIT_AND_EDA.md)), ruang lingkup model ringan (*mobile-edge*), optimasi kuantisasi TFLite PTQ (`FP16`/`INT8`), serta kesiapan *deployment* ganda (**Web + Backend Cloud** dan ***On-Device* Smartphone**).

> **Aturan Utama:** Seluruh seleksi, analisis gradien, dan penentuan konfigurasi berjalan pada split **`val`** (`--set eval.eval_split=val`). Split **`test`** dibekukan hingga tahap akhir dan dievaluasi **tepat 1 kali** pada seluruh kandidat final Stage 2 (`m2`, `m3b`, `m4b`) tanpa pelatihan atau *tuning* ulang sesudahnya.

---

## 1. Landasan Bukti Data & Ruang Lingkup Eksperimen (`[Fakta Terverifikasi]`)

1. **Skala & Komposisi Dataset:** 15 kelas (Tomato `77,6%`, Pepper `12,0%`, Potato `10,4%`), `20.638` citra valid resolusi `256×256`, menjadi **`20.624` citra** setelah deduplikasi MD5 (`14` duplikat *within-class* dibuang *in-memory*).
2. **Ketimpangan Kelas:** Rasio *max/min* **21,1×** (`3.208` pada `Tomato__Tomato_YellowLeaf__Curl_Virus` vs `152` pada `Potato___healthy`, hanya `23` sampel di `val`/`test`).
3. **Anti-Leakage Split:** `deduplicate=true` + `group_same_leaf=true` (`seed=42`) menjaga seluruh `101` grup `Leaf X.Y` (`290` file) utuh per split (`14.434` train / `3.095` val / `3.095` test; `OVERLAPS: 0 0 0`).
4. **Target Kandidat & Evaluasi Komprehensif:**
   - **3 Arsitektur *Mobile-Edge*:** `MobileNetV2` (baseline efisiensi), `EfficientNet-B0`, dan `MobileNetV3-Large`.
   - **Optimasi Deployment:** Pure FP32 `.keras` (untuk Backend Web + Grad-CAM) dan TensorFlow Lite PTQ (`FP32`, `FP16`, dan `INT8` dengan kalibrasi *stratified* **300 citra murni dari split `train`**).
   - **Metrik Komprehensif:** Klasifikasi (`Accuracy`, `Precision Macro`, `Recall Macro`, `F1-Macro`, *Worst-Class Recall*, `ECE`, *Bootstrap 95% CI*, *Exact McNemar p-value*) + kelayakan perangkat (`ukuran file MB` & *single-image latency p50/p95 ms*).

---

## 2. Hipotesis Eksperimen Terkontrol (`[Hipotesis]`)

| ID | Pernyataan Hipotesis | Desain Pengujian Terkontrol | Bukti & Kriteria Evaluasi |
| :--- | :--- | :--- | :--- |
| **H1** | **2-Stage Fine-Tuning vs Frozen Base:** Membuka blok residual atas setelah *head warmup* 10 epoch (Stage 2, 25 epoch, `AdamW`, `lr=1e-4`) meningkatkan `val f1_macro` dan menekan kekeliruan antar-penyakit Tomat dibanding Stage 1 *frozen head*. | Bandingkan `m1↔m2`, `m3a↔m3b`, `m4a↔m4b` + kurva GD gabungan Stage 1→2. | `val f1_macro`, *slope* `val_loss`, & `val_confused_pairs.csv`. |
| **H2** | **Perbandingan Terkontrol 3 Arsitektur Mobile-Edge:** Dengan seluruh variabel pengganggu identik (`224×224`, 25 ep Stage 2, `AdamW wd=0.004`, `ls=0.05`, `balanced`), `MobileNetV3-Large`, `EfficientNet-B0`, dan `MobileNetV2` diuji signifikansi statistik (`Exact McNemar`) beserta *trade-off* ukuran MB dan latency CPU. | Bandingkan `m2_mobilenetv2_ft`, `m3b_effnetb0_ft`, dan `m4b_mobilenetv3l_ft` pada `val` lalu 1× `test`. | `f1_macro`, *Exact McNemar p-value*, *Bootstrap 95% CI*, ukuran MB, p50 ms. |
| **H3** | **Efek Kuantisasi TFLite PTQ (`FP16` / `INT8`):** Konversi `.tflite` `FP16` mempertahankan F1 mendekati `.keras` FP32 ($\Delta\text{F1} \le 0{,}005$) sekaligus memangkas ukuran file ~50%, sedangkan `INT8` (kalibrasi 300 citra `train`) menekan ukuran ~70–75% dengan potensi degradasi minor pada kelas bertekstur halus. | Evaluasi `export_tflite.py` pada varian `keras_fp32`, `tflite_fp32`, `tflite_fp16`, `tflite_int8`. | `ΔF1 vs FP32`, *Exact McNemar*, ukuran `.tflite` (MB), CPU latency (ms). |
| **H4** | **Validitas Spasial Grad-CAM & Kalibrasi Probabilitas:** Augmentasi fotometris + geometris memusatkan aktivasi **Grad-CAM** pada lesi daun (bukan latar abu-abu), dan *post-hoc Temperature Scaling* pada logit `val` menekan **ECE** tanpa mengubah akurasi. | Audit `gradcam_grid` + 5-fold OOF Temperature Scaling pada split `val`. | `border_16px_energy_share`, `ECE` before→after OOF. |

---

## 3. Matriks Eksperimen Final (`m1`–`m4b`, Identik & Bebas Confounder)

| Run ID | Arsitektur | Stage & Parent (`resume_from`) | Optimizer & Budget |
| :--- | :--- | :--- | :--- |
| `m1_mobilenetv2_head` | `MobileNetV2` | Stage 1 (`freeze_base=true`, `fine_tune_at=0`) | `adam`, `lr=1e-3`, 10 ep, `ls=0.05`, `balanced` |
| `m2_mobilenetv2_ft` | `MobileNetV2` | Stage 2 ← `m1/best.keras` (`fine_tune_at=107`) | `adamw` (`wd=0.004`), `lr=1e-4`, 25 ep |
| `m3a_effnetb0_head` | `EfficientNet-B0` | Stage 1 (`freeze_base=true`, `fine_tune_at=0`) | `adam`, `lr=1e-3`, 10 ep, `ls=0.05`, `balanced` |
| `m3b_effnetb0_ft` | `EfficientNet-B0` | Stage 2 ← `m3a/best.keras` (`fine_tune_at=162`) | `adamw` (`wd=0.004`), `lr=1e-4`, 25 ep |
| `m4a_mobilenetv3l_head` | `MobileNetV3-Large` | Stage 1 (`freeze_base=true`, `fine_tune_at=0`) | `adam`, `lr=1e-3`, 10 ep, `ls=0.05`, `balanced` |
| `m4b_mobilenetv3l_ft` | `MobileNetV3-Large` | Stage 2 ← `m4a/best.keras` (`fine_tune_at=124`) | `adamw` (`wd=0.004`), `lr=1e-4`, 25 ep |

> Seluruh 6 run mencatat `step_history.csv` setiap 10 step (`train.log_step_every=10`) sebagai pengamat *read-only* untuk visualisasi kurva *mini-batch gradient descent* per step (EMA) dan per epoch kontinu Stage 1→2.

---

## 4. Batasan Penelitian (*Limitations*)

1. **Bias Latar Laboratorium (`M4`):** Citra PlantVillage diambil pada latar studio terkendali (*uniform background*). Model tidak diuji pada foto lapangan *in-situ* dengan oklusi kompleks; aplikasi lapangan di luar distribusi memerlukan adaptasi domain atau segmentasi daun terlebih dahulu.
2. **Duplikat Hampir-Identik & Cakupan Regex `Leaf X.Y` (`M5`):** Audit mengidentifikasi 1 pasang near-duplicate antar-foto (`JR_HL 8552 ↔ 8553`) serta pola grup `Leaf X.Y` baru mencakup 9,4% kelas (dominan `Tomato_healthy`). Foto lain dari tanaman yang sama dengan penamaan berbeda tidak terdeteksi oleh aturan regex saat ini.
3. **Anggaran Terkontrol 25 Epoch (`M8`):** Seluruh arsitektur menggunakan jadwal `epochs=25` dan `lr=1e-4` yang seragam demi perbandingan bebas *confounder*; masing-masing arsitektur tidak dilakukan pencarian *hyperparameter* independen.
4. **Proksi Latency CPU (`M3`):** Pengukuran lokal/Kaggle mencerminkan latensi CPU server web/laptop (`num_threads=4`). Untuk latensi *on-device* aktual pada chipset Android (CPU/NNAPI/Delegate), paket deliverable menyertakan `android_benchmark_guide.md` (`android_aarch64_benchmark_model` via ADB).
