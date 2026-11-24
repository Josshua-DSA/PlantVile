# Experiment Protocol — PlantVile (11-Dimension CV Landscape)

Dokumen ini adalah protokol eksperimen tertulis untuk proyek **PlantVile** yang menggabungkan temuan audit/EDA dataset lokal ([docs/DATA_AUDIT_AND_EDA.md](docs/DATA_AUDIT_AND_EDA.md)), literatur riset PlantVillage, dan arsitektur kode (`main.py`, `configs/kaggle_t4x2.yaml`, `src/`).

> **Aturan Utama:** Seluruh seleksi arsitektur, *fine-tuning*, dan perbandingan loss **wajib** dievaluasi pada split **`val`** (`--set eval.eval_split=val`). Split **`test`** dibekukan hingga akhir dan hanya dievaluasi **1 kali** pada model pemenang final.

---

## 1. Landasan Bukti Data & Literatur

### 1.1 Fakta Dataset Lokal (`[Fakta Terverifikasi]`)
- **Skala & Komposisi:** 15 kelas (3 spesies: Tomato `77,6%`, Pepper `12,0%`, Potato `10,4%`), `20.638` citra valid resolusi asli seragam `256×256` (`RGB` + 1 `RGBA`), menjadi **`20.624` citra** setelah deduplikasi MD5 (`14` duplikat *within-class* dibuang *in-memory*).
- **Ketimpangan Kelas:** Rasio *max/min* mencapai **21,1×** (`3.208` pada `Tomato__Tomato_YellowLeaf__Curl_Virus` vs `152` pada `Potato___healthy`, yang hanya memiliki ~23 sampel di `val`/`test`).
- **Kualitas Split & Kebocoran:** Dengan `data.deduplicate=true` dan `data.group_same_leaf=true` (`seed=42`), seluruh `101` grup foto daun fisik yang sama (`Leaf X.Y`, total `290` file) dijaga tetap utuh per split (`14.434` train / `3.095` val / `3.095` test; `OVERLAPS: 0 0 0`).
- **Risiko Latar Studio:** Analisis tepi 16 px menunjukkan variasi kecerahan antar-kelas (`BrdLum` `0,477`–`0,604`; `BrdStd < 0,10` di beberapa kelas Tomat).

### 1.2 Temuan Literatur PlantVillage
- **2-Stage Transfer Learning (*Too et al., 2019*; benchmark ImageNet CNN):** Pelatihan dua tahap (*frozen head warmup* → *partial unfreezing* pada resolusi `224×224` dengan penurunan *learning rate*) lebih stabil daripada membuka backbone dari epoch 1, dengan `MobileNetV2`, `EfficientNetB0`, dan `ResNet50` lazim bersaing pada rentang Macro F1 `0,96–0,99` (tergantung protokol split).
- **Bias Latar Belakang (*Mohanty et al., 2016*; *Noyan, 2022*; *Toda & Okura*):** *Noyan (2022)* menunjukkan model yang hanya melihat **8 piksel latar belakang** PlantVillage mencapai **49,0% accuracy** (vs 2,6% tebakan acak). Oleh karena itu, **augmentasi fotometris + geometris** serta **audit visual Grad-CAM** wajib masuk dalam protokol untuk memverifikasi model fokus pada gejala daun, bukan latar studio.

---

## 2. Rumusan Masalah & Hipotesis Eksperimen (H1–H4)

- **Masalah:** Klasifikasi supervised multi-kelas $f_\theta: X \rightarrow Y$ memetakan citra daun $X \in \mathbb{R}^{224 \times 224 \times 3}$ ke 15 kelas penyakit tanaman untuk aplikasi/UI lokal.

| ID | `[Hipotesis]` | Desain Pengujian Terkontrol | Kriteria Evaluasi & Bukti |
| :--- | :--- | :--- | :--- |
| **H1** | **2-Stage Fine-Tuning vs Frozen Base:** Membuka blok residual atas (`fine_tune_at=107` pada `MobileNetV2` setelah *head warmup* `m1`, dengan `adamw`, `lr=1e-4`) akan meningkatkan `val f1_macro` dan mengurangi kekeliruan antar-penyakit Tomat dibanding *frozen head* (`m1`) | Bandingkan `m1_mobilenetv2_head` (frozen) vs `m2_mobilenetv2_ft` (*warm-start* dari `m1/best.keras`) | `val f1_macro`, daftar `val_confused_pairs.csv`, & audit Grad-CAM |
| **H2** | **Perbandingan 3 Keluarga Arsitektur & Trade-off UI:** Pada kondisi kontrol identik (`224×224`, 2-stage fine-tuning, augmentasi & regularisasi sama), `EfficientNetB0` (`m3b`) dan `ResNet50` (`m4b`) akan bersaing tipis dengan `MobileNetV2` (`m2`); jika selisih `val f1_macro` `< 0,01`, model ringan memberikan *trade-off* ukuran & latensi UI lebih baik | Bandingkan `m2_mobilenetv2_ft`, `m3b_effnetb0_ft`, dan `m4b_resnet50_ft` | `val f1_macro`, *worst-class recall*, ukuran `best.keras` (MB) |
| **H3** | **Penanganan Imbalance 21,1× & Kalibrasi ECE:** Pada arsitektur pemenang, `focal` loss + `class_weight=none` (`m5_focal_check`) berpotensi menghasilkan kalibrasi probabilitas (**ECE**) atau keseimbangan precision-recall kelas minoritas berbeda dibanding `categorical_crossentropy + label_smoothing=0.05 + class_weight=balanced` (`BalWt=9,08`) | Bandingkan `m5_focal_check` vs run pemenang `categorical_crossentropy + balanced + label_smoothing=0.05` | `val f1_macro`, recall & precision `Potato___healthy`, & **ECE (10 bins)** |
| **H4** | **Validitas Visual & Stabilitas Multi-Seed:** Augmentasi geometris + fotometris (`aug_vflip=true`, `aug_rotation=0.08`, `aug_brightness/contrast=0.20`) akan mengarahkan fokus **Grad-CAM** ke bercak daun, dan pengulangan 3 seed (`42, 123, 2024`) pada kandidat terbaik (`m6`) akan menghasilkan variasi (`SD`) kecil sebelum 1× uji `test` | Audit `val_gradcam_grid.png` + jalankan `m6_seed123_*` dan `m6_seed2024_*` pada arsitektur pemenang | Fokus daun pada Grad-CAM & `Mean ± SD` lintas 3 seed |

---

## 3. CV Experiment Configuration Landscape (11 Dimensi)

| # | Dimensi | Konfigurasi Terkunci / Nilai Eksperimen (`src/config.py`) |
| :---: | :--- | :--- |
| **1** | **Dataset** | Kaggle `emmarex/plantdisease` → `data.root=/kaggle/input/datasets/emmarex/plantdisease/PlantVillage`, 15 kelas, `data.deduplicate=true` (buang 14 duplikat MD5 *in-memory*). |
| **2** | **Data Splitting** | `val_split=0.15`, `test_split=0.15` (`70/15/15`), `split_strategy=stratified`, `group_same_leaf=true`, `shuffle_seed=42` terkunci. |
| **3** | **Preprocessing** | `224×224` (`img_height=224`, `img_width=224`), interpolasi bilinear, `channels=3` (RGB), `rescale=true`, `normalize=imagenet` (per-backbone `preprocess_input`). |
| **4** | **Augmentation** | Hanya `train` (*batched* di `[0, 1]`): `aug_hflip=true`, `aug_vflip=true`, `aug_rotation=0.08` (~±29°), `aug_width_shift=0.10`, `aug_height_shift=0.10`, `aug_zoom=0.12`, `aug_shear=0.0`, `aug_brightness=0.20`, `aug_contrast=0.20`, `aug_fill_mode=reflect`. |
| **5** | **Model** | Pretrained `weights=imagenet`, head `GAP → Dense(256)+BN+ReLU → Dropout(0.3) → Dense(15, softmax, float32)`. Batas blok residual terverifikasi Keras 3.15.1: `mobilenetv2` (`154` layer, cut **`107`** `block_12_expand`), `efficientnetb0` (`237` layer, cut **`162`** `block6a_expand_conv`), `resnet50` (`175` layer, cut **`143`** `conv5_block1_1_conv`). Seluruh `BatchNormalization` backbone tetap *frozen*. |
| **6** | **Training** | `batch_size=64` (32/GPU T4×2). Stage 1 (Warmup Head): `optimizer=adam`, `lr=0.001`, `epochs=10`. Stage 2 (Fine-Tune): `optimizer=adamw`, `lr=0.0001`, `epochs=25`. |
| **7** | **Loss & Regularization** | Run `m1`–`m4`: `loss=categorical_crossentropy`, `label_smoothing=0.05`, `class_weight=balanced`, `dropout=0.3`, `l2_weight_decay=0.0001`. Run `m5`: `loss=focal`, `class_weight=none`, `label_smoothing=0.0`. |
| **8** | **Learning Strategy** | 2-Stage *Warm-Start* via `runtime.resume_from=<stage1>/best.keras`, `ReduceLROnPlateau` (`val_loss`, `factor=0.5`, `patience=3`, `min_lr=1e-6`), `EarlyStopping` (`val_loss`, `patience=6`, `restore_best_weights=true`), `ModelCheckpoint` (`val_accuracy`, `save_best_only=true`). |
| **9** | **Evaluation** | Wajib `eval.eval_split=val` selama `m1`–`m6`; `test` hanya 1× di akhir (`--eval`). Primer: `f1_macro`; sekunder: *worst-class recall*, `accuracy`, `top_3_accuracy`, `ECE` (10 bins), `confused_pairs`, `misclassified_grid`, `gradcam_grid`, dan ukuran `best.keras` (MB). |
| **10** | **Compute** | Kaggle T4×2 (`configs/kaggle_t4x2.yaml`): `device=gpu`, `strategy=auto` (`MirroredStrategy`), `mixed_precision=mixed_float16`, `cache=memory`, `prefetch=0` (`AUTOTUNE`), `num_parallel_calls=0`. Fallback CPU: `configs/default.yaml` (`mixed_precision=off`, `batch_size=32`, `num_parallel_calls=4`). |
| **11** | **Experiment & Reproducibility** | `runtime.seed=42` (utama), tambahan `123` & `2024` pada kandidat final (`m6`); `runs/<run_id>/` menyimpan `config_used.{yaml,json}`, `device.json`, `history.{csv,json}`, `class_names.json`, `dataset_audit.json`, dan `experiment_leaderboard.csv`. |

---

## 4. Matriks Eksperimen Final (`m1`–`m6`) & Cara Eksekusi di Kaggle

Seluruh matriks ini sudah tertanam langsung di Cell 5 [notebook/kaggle_experiment_runner.ipynb](notebook/kaggle_experiment_runner.ipynb) (yang memanggil `main.py --yes` dengan `COMMON_OVERRIDES` + `resume_from` otomatis):

| Batch | Run ID | Stage & Parent (`resume_from`) | Perubahan Kunci | Tujuan |
| :---: | :--- | :--- | :--- | :--- |
| **1** | `m1_mobilenetv2_head` | Stage 1 (Frozen, 10 ep, `lr=1e-3`) | `mobilenetv2`, `fine_tune_at=0` | Baseline ringan & warmup head `m2` |
| **1** | `m2_mobilenetv2_ft` | Stage 2 ← `m1/best.keras` (25 ep, `lr=1e-4`) | `mobilenetv2`, `fine_tune_at=107`, `adamw` | **Kandidat Utama UI Ringan** (uji H1) |
| **1** | `m3a_effnetb0_head` | Stage 1 (Frozen, 10 ep, `lr=1e-3`) | `efficientnetb0`, `fine_tune_at=0` | Warmup head `m3b` |
| **1** | `m3b_effnetb0_ft` | Stage 2 ← `m3a/best.keras` (25 ep, `lr=1e-4`) | `efficientnetb0`, `fine_tune_at=162`, `adamw` | **Komparasi EfficientNetB0** (uji H2) |
| **2** | `m4a_resnet50_head` | Stage 1 (Frozen, 10 ep, `lr=1e-3`) | `resnet50`, `fine_tune_at=0` | Warmup head `m4b` |
| **2** | `m4b_resnet50_ft` | Stage 2 ← `m4a/best.keras` (25 ep, `lr=1e-4`) | `resnet50`, `fine_tune_at=143`, `adamw` | **Komparasi ResNet50** (uji H2) |
| **3** | `m5_focal_check` | Stage 2 ← `<pemenang_head>/best.keras` | `loss=focal`, `class_weight=none`, `ls=0.0` | **Uji Imbalance & Kalibrasi ECE** (uji H3) |
| **3** | `m6_seed123_head` → `m6_seed123_ft` | 2-Stage penuh (`WINNER_ARCH`) | `runtime.seed=123` | **Stabilitas Multi-Seed** (uji H4) |
| **3** | `m6_seed2024_head` → `m6_seed2024_ft` | 2-Stage penuh (`WINNER_ARCH`) | `runtime.seed=2024` | **Stabilitas Multi-Seed** (uji H4) |
| **Final** | `FINAL_TEST` (Cell 9) | 1× `--eval` pada split **`test`** | `eval.eval_split=test` | Evaluasi akhir + kemas `ui_model_bundle.zip` |

> **Catatan CLI Lokal Sebelum Push:** Jalankan `python scripts/smoke_test.py` dan `pytest tests/test_error_analysis.py -v` untuk memverifikasi pipeline di CPU lokal sebelum melakukan `git push` ke GitHub.

---

## 5. Experiment Log Template

| Run ID | Tahap | Seed | `val f1_macro` | `val acc` | `top_3` | Worst Recall | `ECE` | Train–Val Gap | `best.keras` (MB) | Catatan / Keputusan |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `m1_mobilenetv2_head` | S1 | 42 | | | | | | | | |
| `m2_mobilenetv2_ft` | S2 | 42 | | | | | | | | |
| `m3a_effnetb0_head` | S1 | 42 | | | | | | | | |
| `m3b_effnetb0_ft` | S2 | 42 | | | | | | | | |
| `m4a_resnet50_head` | S1 | 42 | | | | | | | | |
| `m4b_resnet50_ft` | S2 | 42 | | | | | | | | |
| `m5_focal_check` | S2 | 42 | | | | | | | | |
| `m6_seed123_ft` | S2 | 123 | | | | | | | | |
| `m6_seed2024_ft` | S2 | 2024 | | | | | | | | |
| **`FINAL_TEST`** | **`test`** | **42** | | | | | | **—** | | **Kunci untuk UI & Laporan** |
