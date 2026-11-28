# Laporan Evaluasi Model & Seleksi Deployment — PlantVile

> **Tanggal Evaluasi:** 2026-11-25  
> **Lingkup Artefak:** `results/experiment_reports/` (`m1`–`m4b`, `mobile_export/`, `gradient_descent_plots/`) dan `results/ui_handoff_bundle/model_metadata.json`  
> **Standar Pelaporan:** Mengikuti kerangka `/deep-learning-model-evaluation` — seluruh metrik yang dilaporkan berasal langsung dari artefak hasil eksekusi (`[Fakta Terverifikasi]`); dugaan mekanisme penyebab diberi label `[Hipotesis]`, dan ambang penerimaan yang belum ditetapkan pemangku kepentingan diberi label **`[Proposed]` / `[Asumsi]`**.

---

## 1. Executive Summary

- **Kandidat yang Dievaluasi:** 3 arsitektur *mobile-edge* Stage 2 (*partial fine-tuning*) — `m2_mobilenetv2_ft` (`MobileNetV2`), `m3b_effnetb0_ft` (`EfficientNet-B0`), dan `m4b_mobilenetv3l_ft` (`MobileNetV3-Large`) — dibandingkan terhadap *baseline* Stage 1 *frozen-head* (`m1`, `m3a`, `m4a`) pada 4 format *deployment* (`keras_fp32`, `tflite_fp32`, `tflite_fp16`, `tflite_int8`).
- **Status Keputusan Formal** *(berlaku khusus pada domain citra laboratorium PlantVillage 15 kelas; kesiapan foto lapangan in-situ belum diuji)*:
  1. **Target Utama Web Cloud + Grad-CAM (`keras_fp32`) & Mobile Presisi Tinggi (`tflite_fp16`):** **`m4b_mobilenetv3l_ft` (`MobileNetV3-Large`) → `Deployment Candidate`**.  
     Mencapai `test F1-Macro = 0.9940` (`95% CI: [0.9912, 0.9965]`, `accuracy = 0.9935`, `13.04 MB`, `p50 = 23.80 ms`) pada `keras_fp32`, serta `test F1-Macro = 0.9943` (`95% CI: [0.9916, 0.9968]`, `6.24 MB`, `p50 = 4.13 ms`) pada `tflite_fp16`.
  2. **Target Edge Ultra-Ringan / Wajib INT8 (`tflite_int8`):** **`m2_mobilenetv2_ft` (`MobileNetV2`) → `Deployment Candidate`**.  
     Satu-satunya kandidat yang mempertahankan akurasi pasca-kuantisasi penuh INT8 tanpa degradasi bermakna (`test F1-Macro = 0.9864`, `95% CI: [0.9820, 0.9904]`, `ΔF1 = -0.0003`, `McNemar p = 1.0`, ukuran **`2.93 MB`**, `p50 = 5.82 ms`).
  3. **`m3b_effnetb0_ft` (`keras_fp32` & `tflite_fp16`) → `Meets Evaluation Criteria`**.  
     Secara statistik tidak berbeda signifikan dari `m4b` pada split `test` (`McNemar p = 0.5716` pada `keras_fp32`; `p = 0.4421` pada `tflite_fp16`), tetapi terdominasi secara operasional (`1,34×` lebih besar dan `1,61×–2,31×` lebih lambat dibanding `m4b`).
  4. **`m4b_mobilenetv3l_ft` (`tflite_int8`) & `m3b_effnetb0_ft` (`tflite_int8`) → `Not Accepted`**.  
     Gagal pada gerbang retensi kuantisasi (`m4b` INT8 runtuh ke `F1 = 0.4428`, `worst_class_recall = 0.0`, `p50 = 63.80 ms`; `m3b` INT8 turun `-0.0451` ke `F1 = 0.9467`, `McNemar p = 3.03e-39`).

---

## 2. Decision Context

- **Definisi Tugas:** Klasifikasi *single-label multi-class* 15 kelas penyakit & kondisi sehat daun pada 3 spesies tanaman (Tomat 10 kelas, Paprika/Cabai 2 kelas, Kentang 3 kelas).
- **Target Operasi Ganda:**
  - *Jangka Pendek:* Aplikasi Web dengan backend Python di cloud/server lokal yang melayani inferensi cepat dan **visualisasi Grad-CAM** (`keras_fp32`).
  - *Ekspansi / Edge:* Inferensi *on-device* smartphone atau server hemat RAM (`tflite_fp16` / `tflite_int8`).
- **Konsekuensi Kesalahan:**
  - *False Negative* pada penyakit agresif (mis. `Tomato_Late_blight`, `Tomato_Early_blight`) berisiko menunda penanganan infeksi.
  - *False Positive* pada daun sehat menyebabkan penyemprotan pestisida/fungisida yang tidak perlu.
  - Salah klasifikasi antar-penyakit Tomat (mis. jamur vs bakteri vs hama tungau) berisiko salah memilih perlakuan pengendalian.
- **Pemilihan Metrik Utama:** **`F1-Macro`** dipilih sebagai metrik primer karena memberi bobot setara pada seluruh 15 kelas meskipun distribusi dataset sangat timpang (**21,1×** antara kelas terbesar dan terkecil). Metrik sekunder meliputi `Worst-Class Recall` (penjaga kelas minoritas), `Accuracy`, `Top-3 Accuracy`, `ECE`, `Bootstrap 95% CI`, `Exact McNemar p-value`, ukuran model (MB), dan latency CPU (`p50`/`p95` ms).

---

## 3. Experimental Protocol & Integrity Audit

- **Dataset & Pembersihan (`[Fakta Terverifikasi]`):** PlantVillage `color` (`20.638` citra valid, `0` citra korup berdasarkan audit PIL) → **`20.624` citra unik** setelah deduplikasi MD5 *in-memory* (`14` duplikat *within-class* dibuang; `0` lintas-kelas).
- **Strategi Split & Anti-Leakage:** `14.434` train (`70%`) / `3.095` val (`15%`) / `3.095` test (`15%`), *stratified per class*, `group_same_leaf=True` (`101` grup `Leaf X.Y` / `290` file dijaga utuh per split), `seed=42`, dengan **`0` tumpang tindih** antar-split.
- **Konsistensi Preprocessing & Komputasi:** Seluruh model dievaluasi pada input `224×224×3` (`bilinear` resize) dengan `preprocess_input` spesifik masing-masing backbone, `batch_size=64`, `label_smoothing=0.05`, `class_weight=balanced` (dari `train`), `l2=1e-4`, `MirroredStrategy (2 GPUs)`, `mixed_float16`, dan kalibrasi INT8 PTQ menggunakan `300` citra *stratified* eksklusif dari `train`.
- **Protokol Seleksi & Test:** Seleksi checkpoint (`best.keras`), analisis kurva training, dan fitting *5-fold OOF Temperature Scaling* hanya memakai `val`. Evaluasi split `test` (`n=3.095`) dijalankan tepat 1× di akhir.

### Temuan Audit Protokol (*Caveats* yang Wajib Diungkapkan) `[Fakta Terverifikasi]`
1. **Perbedaan Provenance Run `m1`–`m3b` vs `m4a`–`m4b`:**
   - Pada `config_used.json`, `m1`–`m3b` dijalankan sebelum penambahan parameter eksplisit `adamw_weight_decay` (`None`), `StepHistoryLogger` (`step_history.csv=False`), dan `CoreModelCheckpoint` (`best.keras` menyimpan state optimizer: `m2=42.71 MB`, `m3b=70.43 MB` vs `m4b=13.05 MB`).
   - Pada Keras `AdamW`, nilai default `weight_decay` adalah `0.004` — identik secara matematis dengan `adamw_weight_decay=0.004` pada `m4b`, dan `best.keras` tetap diekspor ulang menjadi `infer_fp32.keras` bebas optimizer. Namun kurva *per-step* (`step_history.csv`) hanya tersedia untuk `m4a` dan `m4b`.
2. **Single Seed & Single Partition:** Seluruh eksperimen berjalan pada `seed=42` tunggal. Variansi antar-*seed* dan antar-*fold* split adalah **`not measured`**. Ketidakpastian statistik pada laporan ini diestimasi melalui *Bootstrap 95% CI* (1.000 sampel ulang pada `test`) dan uji berpasangan *Exact McNemar*.
3. **Pergeseran Kecil Cast FP16 → FP32 pada `m3b_effnetb0_ft`:**
   - Pada `test_leaderboard.csv` (`best.keras` `mixed_float16`): `accuracy = 0.9926`, `f1_macro = 0.9934` (`23` error).
   - Setelah dikonversi ke `infer_fp32.keras` (`mobile_benchmark_summary.csv`): `accuracy = 0.992246`, `f1_macro = 0.991729` (`ΔF1 = -0.0017`, `24` error), sementara `m2` dan `m4b` menunjukkan pergeseran `0.0000`.
4. **Ketidaksinkronan Metrik Monitor Checkpoint:** `checkpoint_monitor` memantau `val_accuracy`, sedangkan `EarlyStopping` dan `ReduceLROnPlateau` memantau `val_loss`. Hal ini menyebabkan `m3b` memilih checkpoint `epoch 18` (`val_acc = 0.9932`), padahal `val_loss`-nya masih terus turun hingga `epoch 25` (`0.3859`). Sebaliknya `m4b` mencapai puncak `val_accuracy` dan `val_loss` terendah bersamaan pada `epoch 25`.

---

## 4. Predictive Performance

### Tabel A — Perbandingan Validation (`val`, `n = 3.095`): Efek 2-Stage Fine-Tuning
*(Sumber: `results/experiment_reports/experiment_leaderboard.csv`)*

| Run ID | Stage | `f1_macro` | `accuracy` | `top_3_acc` | `worst_recall` | `ece` | Misclassified | `train_val_gap` | Gain F1 (Stage 1→2) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`m4b_mobilenetv3l_ft`** | Stage 2 FT | **0.9941** | **0.9932** | 0.9990 | 0.9733 | **0.0385** | **21** | +0.0061 | **+0.0591** |
| **`m3b_effnetb0_ft`** | Stage 2 FT | 0.9929 | **0.9932** | 0.9987 | **0.9800** | 0.0436 | **21** | +0.0057 | **+0.0751** |
| **`m2_mobilenetv2_ft`** | Stage 2 FT | 0.9865 | 0.9864 | **0.9994** | 0.9400 | 0.0389 | 42 | +0.0149 | **+0.1004** |
| `m4a_mobilenetv3l_head` | Stage 1 Head | 0.9350 | 0.9389 | 0.9955 | 0.7667 | 0.0740 | 189 | +0.0029 | — |
| `m3a_effnetb0_head` | Stage 1 Head | 0.9178 | 0.9351 | 0.9935 | 0.7867 | 0.0657 | 201 | +0.0001 | — |
| `m1_mobilenetv2_head` | Stage 1 Head | 0.8861 | 0.9018 | 0.9871 | 0.4933 | 0.0781 | 304 | -0.0050 | — |

### Tabel B — Evaluasi Final Split `test` (`n = 3.095`, Kandidat Stage 2 `keras_fp32`)
*(Sumber: `test_leaderboard.csv` untuk `best.keras` dan `mobile_benchmark_summary.csv` untuk `infer_fp32.keras` + Bootstrap 95% CI)*

| Model (`infer_fp32.keras`) | `f1_macro` [95% CI] | `accuracy` [95% CI] | `precision_macro` | `recall_macro` | `worst_recall` | `ece` | Misclassified (`best.keras`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`m4b_mobilenetv3l_ft`** | **0.9940** `[0.9912, 0.9965]` | **0.9935** `[0.9906, 0.9961]` | **0.9942** | **0.9937** | 0.9600 | 0.0407 | **20** |
| **`m3b_effnetb0_ft`** | 0.9917 `[0.9868, 0.9954]` *(raw: 0.9934)* | 0.9922 `[0.9890, 0.9952]` *(raw: 0.9926)* | 0.9903 | 0.9932 | **0.9684** | 0.0452 | 23 *(raw)* |
| **`m2_mobilenetv2_ft`** | 0.9867 `[0.9825, 0.9905]` | 0.9858 `[0.9816, 0.9897]` | 0.9863 | 0.9872 | 0.9400 | **0.0371** | 44 |

### Tabel C — Uji Signifikansi Statistik Exact McNemar (`test`, `n = 3.095`)
*(Sumber: `results/experiment_reports/mobile_export/mcnemar_pairwise.csv`)*

| Perbandingan | `A benar, B salah` | `A salah, B benar` | `p-value` | Interpretasi Statistik ($\alpha = 0{,}05$) |
| :--- | :---: | :---: | :---: | :--- |
| `keras_fp32`: `m2` vs `m3b` | 14 | 34 | `0.0055` | `m3b` signifikan lebih akurat dari `m2` |
| `keras_fp32`: `m2` vs `m4b` | 13 | 37 | `0.0009` | `m4b` signifikan lebih akurat dari `m2` |
| `keras_fp32`: `m3b` vs `m4b` | 12 | 16 | `0.5716` | Tidak berbeda signifikan (selisih dalam variasi acak) |
| `tflite_fp16`: `m3b` vs `m4b` | 11 | 16 | `0.4421` | Tidak berbeda signifikan secara akurasi pada FP16 |
| `m2`: `keras_fp32` vs `tflite_int8` | 6 | 6 | `1.0000` | Retensi sempurna tanpa degradasi signifikan |
| `m3b`: `keras_fp32` vs `tflite_int8` | 145 | 3 | `3.03e-39` | Degradasi sangat signifikan |
| `m4b`: `keras_fp32` vs `tflite_int8` | 1.739 | 7 | `0.0000` | Keruntuhan akurasi ekstrem pada INT8 |

---

## 5. Class-Wise and Slice Performance (`test`, `n = 3.095`)

### Tabel Per-Kelas Lengkap (`test_per_class.csv`)

| Kelas | Support | `m2` F1 / Recall | `m3b` F1 / Recall | `m4b` F1 / Recall |
| :--- | :---: | :---: | :---: | :---: |
| `Pepper__bell___Bacterial_spot` | 150 | 0.9835 / 0.9933 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| `Pepper__bell___healthy` | 222 | 0.9932 / 0.9865 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| `Potato___Early_blight` | 150 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| `Potato___Late_blight` | 150 | 0.9967 / 1.0000 | 1.0000 / 1.0000 | 0.9967 / 1.0000 |
| `Potato___healthy` | 23 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| `Tomato_Bacterial_spot` | 319 | 0.9860 / 0.9969 | 0.9938 / 0.9969 | 0.9953 / 0.9937 |
| `Tomato_Early_blight` | 150 | 0.9527 / **0.9400** | 0.9801 / 0.9867 | 0.9730 / **0.9600** |
| `Tomato_Late_blight` | 285 | 0.9768 / 0.9614 | 0.9805 / **0.9684** | 0.9841 / 0.9789 |
| `Tomato_Leaf_Mold` | 143 | 0.9965 / 1.0000 | 0.9930 / 0.9930 | 0.9965 / 0.9930 |
| `Tomato_Septoria_leaf_spot` | 266 | 0.9829 / 0.9737 | 0.9925 / 0.9962 | 0.9870 / 0.9962 |
| `Tomato_Spider_mites_Two_spotted_spider_mite` | 251 | 0.9819 / 0.9721 | 0.9940 / 0.9960 | 0.9960 / 1.0000 |
| `Tomato__Target_Spot` | 211 | 0.9633 / 0.9953 | 0.9833 / 0.9763 | 0.9882 / 0.9905 |
| `Tomato__Tomato_YellowLeaf__Curl_Virus` | 481 | 0.9969 / 0.9938 | 0.9979 / 0.9958 | 0.9990 / 0.9979 |
| `Tomato__Tomato_mosaic_virus` | 56 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| `Tomato_healthy` | 238 | 0.9896 / 0.9958 | 0.9854 / 0.9958 | 0.9937 / 0.9958 |

### Tabel Slice Semantik (Spesies Tanaman, Status Kesehatan, & Kelas Minoritas)

| Slice | Support (`n`) | `m2` Err / Acc / Mean F1 | `m3b` Err / Acc / Mean F1 | `m4b` Err / Acc / Mean F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Tomato (10 kelas)** | `2.400` | 40 / 0.9833 / 0.9827 | 23 / 0.9904 / 0.9900 | **20** / **0.9917** / **0.9913** |
| **Pepper (2 kelas)** | `372` | 4 / 0.9892 / 0.9884 | **0** / **1.0000** / **1.0000** | **0** / **1.0000** / **1.0000** |
| **Potato (3 kelas)** | `323` | **0** / **1.0000** / 0.9989 | **0** / **1.0000** / **1.0000** | **0** / **1.0000** / 0.9989 |
| **Healthy (3 kelas)** | `483` | 4 / 0.9917 / 0.9943 | 1 / 0.9979 / 0.9951 | **1** / **0.9979** / **0.9979** |
| **Diseased (12 kelas)** | `2.612` | 40 / 0.9847 / 0.9848 | 22 / 0.9916 / 0.9929 | **19** / **0.9927** / **0.9930** |
| **Low-support (`n < 100`)** (`Potato___healthy`, `Tomato_mosaic_virus`) | `79` | 0 / 1.0000 / 1.0000 | 0 / 1.0000 / 1.0000 | 0 / 1.0000 / 1.0000 |

- **Temuan Kritis Slice:** **100% kesalahan `m4b` (`20/20`) dan `m3b` (`23/23`) terisolasi di dalam sub-grup Tomat** (dominan pada `Tomato_Early_blight` dan `Tomato_Late_blight`). Kelas minoritas berdukungan rendah (`Potato___healthy` `n=23`, `Tomato__Tomato_mosaic_virus` `n=56`) mencatat `100%` akurasi pada ketiga model, tetapi jumlah sampel kecil membatasi kepastian statistiknya di luar dataset ini.

---

## 6. Generalization and Training Dynamics

*(Sumber: `history.json` dan `gradient_descent_plots/gd_summary.json`; batas bawah entropi *label smoothing* $H(q) = 0{,}311737$)*

| Run ID | Best `val_acc` (@Epoch) | Best `val_loss` (@Epoch) | `gap_acc` (train−val) | `gap_loss` (val−train) | `last5_val_loss_slope` | `last5_diff_std` | Jadwal LR (Start → End) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `m1_mobilenetv2_head` | 0.9024 @ep10 | 0.6943 @ep10 | -0.0050 | +0.0027 | — | — | `1.0e-3 → 1.0e-3` |
| `m2_mobilenetv2_ft` | 0.9868 @ep23 | 0.4151 @ep23 | +0.0090 | +0.0159 | -0.0070 / ep | 0.0238 | `1.0e-4 → 5.0e-5` |
| `m3a_effnetb0_head` | 0.9344 @ep9 | 0.6064 @ep9 | -0.0092 | -0.0282 | — | — | `1.0e-3 → 1.0e-3` |
| `m3b_effnetb0_ft` | 0.9932 @ep18 | 0.3859 @ep25 | +0.0035 | +0.0004 | -0.0053 / ep | **0.0074** | `1.0e-4 → 1.0e-4` |
| `m4a_mobilenetv3l_head` | 0.9389 @ep10 | 0.5884 @ep10 | +0.0029 | +0.0155 | — | — | `1.0e-3 → 1.0e-3` |
| **`m4b_mobilenetv3l_ft`** | **0.9932 @ep25** | **0.3668 @ep25** | +0.0061 | +0.0088 | -0.0042 / ep | 0.0078 | `1.0e-4 → 5.0e-5` |

- **Interpretasi Dinamika Training (`[Fakta Terverifikasi]`):**
  - Tidak ada run yang dihentikan lebih awal oleh `EarlyStopping`; ketiga model Stage 2 menyelesaikan 25 epoch penuh tanpa gejala *overfitting* merusak (`gap_acc <= +0.0149` pada puncak masing-masing).
  - **`m4b_mobilenetv3l_ft`** mencapai `val_loss` terendah (`0.3668`, paling dekat ke lantai teoretis `0.3117`), `last5_diff_std = 0.0078` yang stabil, dan kurva *mini-batch* per step (5.650 step total, `final_ema_loss = 0.36598`) menurun mulus mengikuti `val_loss`.
  - **`m2_mobilenetv2_ft`** menunjukkan osilasi `val_loss` terbesar (`last5_diff_std = 0.0238`, 3× lebih volatil dibanding `m3b`/`m4b`) sebelum `ReduceLROnPlateau` menurunkan LR ke `5.0e-5` dan menstabilkan epoch 22–25.

---

## 7. Error Analysis (`test`, `n = 3.095`)

Total kesalahan pada `best.keras`: `m4b = 20` (`4` *high-confidence* `>= 0.80`), `m3b = 23` (`2` *high-confidence*), `m2 = 44` (`4` *high-confidence*). Terdapat **6 sampel yang gagal serentak di seluruh 3 model**, dan **8 sampel bersama antara `m3b` dan `m4b`**.

### Klaster Kegagalan Terstruktur

1. **`EA-001` — Dugaan Kuat *Label Noise* pada Ground Truth (Kepercayaan Diagnostik: Tinggi `[Hipotesis]`)**
   - **Observasi:** Sampel `cb3bd866-dd84-4f7e-9226-b4a2a2c7f2ae___GH_HL Leaf 173.2.JPG` berlabel folder `Tomato_Late_blight`, tetapi diprediksi sebagai `Tomato_healthy` dengan *confidence* ekstrem oleh **seluruh 3 model** (`m4b: 0.971`, `m2: 0.965`, `m3b: 0.953`, `p_true <= 0.010`).
   - **Bukti Pendukung:** Prefix nama file PlantVillage adalah **`GH_HL Leaf`** (*Greenhouse Healthy Leaf* — penamaan resmi sampel daun sehat), dan inspeksi visual pada `test_gradcam_grid.png` (`m4b`, baris 1 kolom 1–2) memperlihatkan daun hijau mulus tanpa lesi nekrotik, dengan aktivasi Grad-CAM penuh pada lamina daun sehat.
2. **`EA-002` — Kemiripan Morfologis Lesi Bercak/Nekrosis Intra-Tomat (Kepercayaan Diagnostik: Tinggi `[Fakta Terverifikasi]` + `[Hipotesis]`)**
   - **Observasi:** Pasangan tertukar dominan di seluruh model adalah `Tomato_Early_blight <-> Tomato_Septoria_leaf_spot` (`4` kasus pada `m4b`), `Tomato_Late_blight -> Tomato_Early_blight` (`2–4` kasus), dan `Tomato_Spider_mites <-> Tomato__Target_Spot` (`7` kasus pada `m2`).
   - **Sampel Representatif:** `03a15d38...___RS_Late.B 5207.JPG`, `766771e1...___RS_Late.B 6881.JPG`, `535c3c4f...___RS_Erly.B 7770.JPG`, `e9c43f84...___Com.G_TgS_FL 9965.JPG`, `e81dfd96...___RS_Erly.B 7407.JPG`, `fb09d588...___GCREC_Bact.Sp 3557.JPG`.
   - **Bukti Pendukung:** Inspeksi `m4b/test_gradcam_grid.png` menunjukkan Grad-CAM terpusat tepat pada area lesi daun (bukan latar abu-abu), mengindikasikan model sudah mengekstraksi wilayah gejala yang relevan, namun fitur bercak tahap lanjut saling tumpang tindih antar-patogen Tomat. *(Pengecualian kecil: baris 4 kolom 1–2 `Target_Spot -> healthy (0.70)` menunjukkan aktivasi menyebar hingga latar bawah-kanan).*
3. **`EA-003` — Sampel di Luar Pola Sumber Standar (Kepercayaan Diagnostik: Sedang `[Hipotesis]`)**
   - **Observasi:** `7de93009-6180-4027-8779-76947cda3e2f___2700323949_95aa2eaa01_o.jpg` (berlabel `Tomato_healthy`, penamaan non-standar Flickr `_o.jpg`) gagal di seluruh 3 model dengan *confidence* rendah (`m4b->Late_blight 0.44`, `m3b->Late_blight 0.66`, `m2->Pepper_Bacterial_spot 0.62`).

---

## 8. Robustness, Calibration, and Operational Efficiency

### A. Kalibrasi Probabilitas (`[Fakta Terverifikasi]`)
- **ECE Mentah (Test):** Seluruh model Stage 2 menunjukkan `ECE = 0.037–0.045`, dengan kurva reliabilitas pada `test_confidence_calibration.png` berada **di atas garis diagonal** (*sedikit underconfident* sebagai efek `label_smoothing=0.05` yang memangkas puncak softmax ke ~0.95).
- **Temperature Scaling (`5-fold OOF` pada `val`):** Optimasi parameter temperatur $T < 1$ secara konsisten menurunkan `val ECE` hingga **92,4%–94,8%** tanpa mengubah akurasi *argmax*:
  - `m4b_mobilenetv3l_ft`: $T = 0{,}6429$ → `val ECE`: `0.0385 → 0.0020` (-94,8%)
  - `m3b_effnetb0_ft`: $T = 0{,}5807$ → `val ECE`: `0.0435 → 0.0023` (-94,6%)
  - `m2_mobilenetv2_ft`: $T = 0{,}6073$ → `val ECE`: `0.0389 → 0.0029` (-92,4%)
- *Catatan:* Nilai `test ECE` pasca-*Temperature Scaling* adalah **`not measured`** (fit `T` hanya dievaluasi OOF di `val` agar menjaga prinsip anti-overuse `test`).

### B. Benchmark Efisiensi & Kuantisasi PTQ (Split `test`, CPU `num_threads=4`)
*(Sumber: `results/experiment_reports/mobile_export/mobile_benchmark_summary.csv`)*

| Run ID | Format | Size (MB) | `accuracy` | `f1_macro` | `ΔF1 vs FP32` | `worst_recall` | `ece` | `p50_ms` | `p95_ms` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`m4b_mobilenetv3l_ft`** | `keras_fp32` | 13.04 | 0.9935 | 0.9940 | 0.0000 | 0.9600 | 0.0407 | 23.80 | 25.04 |
| `m4b_mobilenetv3l_ft` | `tflite_fp32` | 12.32 | 0.9935 | 0.9940 | 0.0000 | 0.9600 | 0.0407 | 3.90 | 4.24 |
| **`m4b_mobilenetv3l_ft`** | **`tflite_fp16`** | **6.24** | **0.9939** | **0.9943** | **+0.0004** | **0.9667** | 0.0406 | **4.13** | 6.53 |
| `m4b_mobilenetv3l_ft` | `tflite_int8` | 3.60 | 0.4339 | 0.4428 | -0.5512 | 0.0000 | 0.2270 | 63.80 | 76.87 |
| `m3b_effnetb0_ft` | `keras_fp32` | 17.52 | 0.9922 | 0.9917 | 0.0000 | 0.9684 | 0.0452 | 38.38 | 40.52 |
| `m3b_effnetb0_ft` | `tflite_fp32` | 16.58 | 0.9922 | 0.9917 | 0.0000 | 0.9684 | 0.0452 | 9.60 | 10.24 |
| `m3b_effnetb0_ft` | `tflite_fp16` | 8.39 | 0.9922 | 0.9917 | 0.0000 | 0.9684 | 0.0452 | 9.55 | 10.03 |
| `m3b_effnetb0_ft` | `tflite_int8` | 5.03 | 0.9464 | 0.9467 | -0.0451 | 0.8167 | 0.0741 | 9.31 | 10.00 |
| `m2_mobilenetv2_ft` | `keras_fp32` | 10.45 | 0.9858 | 0.9867 | 0.0000 | 0.9400 | 0.0371 | 24.88 | 26.63 |
| `m2_mobilenetv2_ft` | `tflite_fp32` | 9.74 | 0.9858 | 0.9867 | 0.0000 | 0.9400 | 0.0371 | 4.13 | 4.82 |
| `m2_mobilenetv2_ft` | `tflite_fp16` | 4.93 | 0.9858 | 0.9867 | 0.0000 | 0.9400 | 0.0370 | 4.01 | 4.25 |
| **`m2_mobilenetv2_ft`** | **`tflite_int8`** | **2.93** | **0.9858** | **0.9864** | **-0.0003** | 0.9267 | 0.0437 | **5.82** | 6.36 |

- **Diagnosis Kegagalan INT8 (`[Hipotesis]`):** `MobileNetV2` memakai aktivasi `ReLU6` yang terikat pada `[0, 6]` sehingga rentang kuantisasi *affine* 8-bit tetap presisi (`ΔF1 = -0.0003`). Sebaliknya, aktivasi tak terbatas / asimetris (`SiLU` di `EfficientNet-B0`, `Hard-Swish` + *Squeeze-and-Excitation* di `MobileNetV3-Large`) menimbulkan *quantization clipping/scale error* besar pasca-pelatihan — ditambah pada `m4b` INT8, delegasi XNNPACK gagal disiapkan sehingga eksekusi jatuh ke *fallback* tanpa delegasi (`p50` melambat menjadi `63.8 ms`).
- **Dimensi yang Belum Diukur (`not measured`):** Foto daun lapangan nyata (*in-situ* dengan latar kompleks/cahaya alami), latency ADB fisik di perangkat Android, dan variansi antar-*seed*.

---

## 9. Acceptance Assessment & Formal Model Selection

Karena ambang batas bisnis belum ditetapkan di awal, berikut adalah **usulan gerbang penerimaan non-kompensatoris (`[Proposed]`)**:

| Hard Gate `[Proposed]` | Ambang Batas | `m4b` (`keras_fp32`) | `m4b` (`tflite_fp16`) | `m4b` (`tflite_int8`) | `m3b` (`tflite_fp16`) | `m2` (`tflite_int8`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Validitas Protokol** | Zero leakage, 1× `test` | PASS | PASS | PASS | PASS | PASS |
| **2. `test F1-Macro`** | $\ge 0{,}980$ | **PASS** (`0.9940`) | **PASS** (`0.9943`) | **FAIL** (`0.4428`) | **PASS** (`0.9917`) | **PASS** (`0.9864`) |
| **3. `Worst-Class Recall`** | $\ge 0{,}900$ | **PASS** (`0.9600`) | **PASS** (`0.9667`) | **FAIL** (`0.0000`) | **PASS** (`0.9684`) | **PASS** (`0.9267`) |
| **4. Retensi Kuantisasi** | $|\Delta\text{F1}| \le 0{,}005$ | — | **PASS** (`+0.0004`) | **FAIL** (`-0.5512`) | **PASS** (`0.0000`) | **PASS** (`-0.0003`) |
| **5. Anggaran Deployment** | Web $\le 25\text{ MB}, p95 \le 100\text{ ms}$; Mobile $\le 10\text{ MB}, p95 \le 25\text{ ms}$ | **PASS** (`13.04 MB`, `25.04 ms`) | **PASS** (`6.24 MB`, `6.53 ms`) | **FAIL** (`76.87 ms`) | **PASS** (`8.39 MB`, `10.03 ms`) | **PASS** (`2.93 MB`, `6.36 ms`) |

### Keputusan Seleksi Model (Selaras dengan `model_metadata.json`)

1. **Web Cloud + Backend Python (Grad-CAM Aktif):** Pilih **`m4b_mobilenetv3l_ft` (`keras_fp32`)** (`Deployment Candidate`).  
   Unggul signifikan atas `m2` (`McNemar p = 0.0009`), lebih kecil dan 1,61× lebih cepat dibanding `m3b`, serta konvergensi `val_loss` terbersih.
2. **Mobile / Web Hemat RAM tanpa Grad-CAM (`FP16`):** Pilih **`m4b_mobilenetv3l_ft` (`tflite_fp16`)** (`Deployment Candidate`).  
   Memangkas ukuran ke **`6.24 MB`** dan mempercepat inferensi CPU ke **`p50 = 4.13 ms`** sambil mempertahankan `test F1 = 0.9943`.
3. **Edge Ultra-Ringan / Wajib INT8 (`INT8`):** Pilih **`m2_mobilenetv2_ft` (`tflite_int8`)** (`Deployment Candidate`).  
   Ukuran terkecil (**`2.93 MB`**, `p50 = 5.82 ms`) dengan retensi sempurna (`ΔF1 = -0.0003`). **Jangan** gunakan `m4b_int8` maupun `m3b_int8`.

---

## 10. Follow-up Experiments (Prioritas)

1. **`EXP-01` (Prioritas 1 — Pemulihan INT8 `MobileNetV3-Large`):**
   - *Temuan Pemicu:* `m4b` runtuh pada PTQ `INT8` (`F1 = 0.4428`, XNNPACK delegate fail).
   - *Hipotesis:* Melakukan *Quantization-Aware Training (QAT)* singkat (5–10 epoch) atau mempertahankan layer sensitif (`Hard-Swish`/SE/output) pada FP16 (*selective quantization*) akan mengembalikan `test F1 >= 0.985` serta memulihkan kompatibilitas XNNPACK.
   - *Intervensi & Variabel Terkendali:* Hanya ubah mode konversi/QAT pada `m4b`; pertahankan dataset, split `seed=42`, dan preprocessing identik.
   - *Kriteria Sukses:* `test F1-Macro >= 0.985`, `worst_class_recall >= 0.940`, dan alokasi XNNPACK berhasil tanpa *fallback*.
2. **`EXP-02` (Prioritas 2 — Audit Multi-Seed & Penyelarasan Monitor Checkpoint):**
   - *Temuan Pemicu:* Evaluasi baru memakai 1 *seed* (`42`), `step_history.csv` hanya ada di `m4a`/`m4b`, dan `checkpoint_monitor=val_accuracy` berbeda dari `EarlyStopping` (`val_loss`).
   - *Hipotesis:* Menyeragamkan `checkpoint_monitor=val_loss` dan mengulang pada 3 *seed* (`42, 1337, 2025`) akan mengukur simpangan baku antar-run serta memverifikasi apakah `m3b` (`ep25`) menyamai `m4b` secara konsisten.
   - *Kriteria Sukses:* Laporan mean $\pm$ std `val F1-Macro` & `ECE` di 3 *seed*.
3. **`EXP-03` (Prioritas 1 Sebelum Rilis Lapangan — Validasi Eksternal *In-Situ*):**
   - *Temuan Pemicu:* Seluruh data uji masih berasal dari latar studio seragam PlantVillage.
   - *Hipotesis:* Akurasi pada foto daun lapangan *in-situ* (PlantDoc / foto kamera HP pengguna) akan turun signifikan dibanding `0.994` akibat *background shift* dan pencahayaan alami.
   - *Intervensi:* Uji langsung `m4b` tanpa *retraining* pada set validasi eksternal terpisah (min. 20–30 foto lapangan per kelas utama) + bandingkan *augmentation domain randomization* (*random background/shadow*).

---

## 11. Limitations and Caveats

- **Tidak Ada Validasi Lapangan Nyata (*Domain Shift*):** Skor `0.9935–0.9943` hanya berlaku pada citra daun tunggal berlatar abu-abu terkendali. Model **belum teruji** pada foto kamera lapangan dengan latar tanah, bayangan kuat, tangan, atau daun tumpang tindih.
- **Single Seed:** Stabilitas pelatihan terhadap perubahan *random seed* belum diukur (`not measured`).
- **Proksi Latency CPU:** Latency `4.13–23.80 ms` diukur pada CPU server (`num_threads=4`), bukan pada SoC/NPU HP Android fisik.
- **Dugaan Label Noise & Post-T Test ECE:** Terdeteksi indikasi kuat label salah pada `GH_HL Leaf 173.2.JPG`, dan `ECE` pasca-*Temperature Scaling* hanya terverifikasi OOF di `val` (untuk penggunaan produksi, aktifkan `apply_temperature=True` dan ambang peringatan *low-confidence* `< 0.70` di `inference_api.py`, disertai pemantauan distribusi kepercayaan di produksi).
