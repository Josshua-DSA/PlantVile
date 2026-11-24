# CV Experiment Configuration Landscape — PlantVile

Dokumen ini mendokumentasikan secara mandiri seluruh konfigurasi eksperimen *Computer Vision* pada proyek klasifikasi penyakit daun tanaman **PlantVile** (menggunakan dataset **PlantVillage** dengan tiga arsitektur CNN *mobile-edge*). Seluruh nilai parameter yang tercatat mencerminkan konfigurasi pelatihan terkontrol, evaluasi, dan pengemasan *deployment* yang digunakan pada eksperimen utama.

---

## 1. Dataset Configuration

| Parameter / Aspek | Nilai / Konfigurasi Aktif | Keterangan |
| :--- | :--- | :--- |
| **Sumber Dataset** | PlantVillage (`emmarex/plantdisease`) | Dataset citra daun berlatar laboratorium |
| **Varian & Format** | `color`, 3-channel RGB | Format file diterima: `.jpg`, `.jpeg`, `.png`, `.bmp` |
| **Resolusi Asli & Integritas** | `256×256` piksel, `0` citra korup | Diverifikasi melalui pemeriksaan dekode PIL dan pemindaian subfolder kelas secara otomatis |
| **Jumlah & Cakupan Kelas** | **15 kelas** (seluruh kelas & sampel digunakan) | Komposisi domain: Tomat `77,6%`, Paprika/Cabai `12,0%`, Kentang `10,4%` |
| **Total Sampel** | `20.638` valid → **`20.624` unik** | `14` duplikat persis (*within-class* berbasis hash MD5) dibuang di memori sebelum *splitting*; pemeriksaan duplikat lintas-kelas aktif |
| **Ketimpangan Kelas** | Rasio maks/min **21,1×** | Maksimum `3.208` citra (`Tomato__Tomato_YellowLeaf__Curl_Virus`), minimum `152` citra (`Potato___healthy`) |

---

## 2. Data Splitting Configuration

| Parameter / Aspek | Nilai / Konfigurasi Aktif | Keterangan |
| :--- | :--- | :--- |
| **Proporsi Split** | **70% Train / 15% Validation / 15% Test** | `val_split=0.15`, `test_split=0.15` |
| **Jumlah Sampel per Split** | `14.434` / `3.095` / `3.095` (`0` tumpang tindih) | Total `20.624` citra unik pasca-deduplikasi MD5 |
| **Strategi Pemisahan** | *Stratified per class* | Proporsi kelas dipertahankan konsisten di tiap *split* |
| **Pencegahan Leakage (Dedup)** | Deduplikasi MD5 sebelum *split* (`deduplicate=true`) | Mencegah citra identik tersebar di *train* dan *val*/*test* |
| **Pencegahan Leakage (Grouping)** | *Group-aware splitting* per daun fisik (`group_same_leaf=true`) | Pengelompokan pola penamaan `Leaf <nomor>` menjaga seluruh `101` grup multi-foto daun (`290` file) tetap utuh dalam satu *split* |
| **Random Seed Split & Shuffle** | `shuffle_seed=42`, `global_seed=42` | Pengurutan *train* tiap epoch memakai `shuffle(buffer_size=min(N_train, 10000), seed=42, reshuffle_each_iteration=True)` |

---

## 3. Preprocessing Configuration

| Parameter / Aspek | Nilai / Konfigurasi Aktif | Keterangan |
| :--- | :--- | :--- |
| **Resolusi Input Eksperimen** | **`224×224×3`** | Tinggi `224` px, lebar `224` px, `3` channel RGB |
| **Color Space & Decoding** | RGB `float32` | Dekode ke RGB 3-channel (tanpa animasi) lalu dikonversi ke rentang `[0, 1]` bertipe `float32` |
| **Resize & Interpolasi** | `bilinear` (`tf.image.resize(..., [224, 224], method="bilinear")`) | Metode dekode dan interpolasi disamakan identik antara *pipeline* training dan inferensi citra tunggal |
| **Rescale & Mode Normalisasi** | `rescale=true`, `normalize=imagenet` | Skala awal `[0, 1]` disesuaikan dengan ekspektasi prapemrosesan masing-masing backbone ImageNet |
| **Normalisasi `MobileNetV2`** | `mobilenet_v2.preprocess_input(x * 255.0)` | Rentang akhir **`[-1, 1]`** |
| **Normalisasi `EfficientNet-B0`** | `efficientnet.preprocess_input(x * 255.0)` | Rentang **`[0, 255]`** (layer internal `Rescaling`/`Normalization` aktif di dalam backbone) |
| **Normalisasi `MobileNetV3-Large`** | `mobilenet_v3.preprocess_input(x * 255.0)` | Rentang **`[0, 255]`** (layer internal `Rescaling` aktif di dalam backbone) |

---

## 4. Augmentation Configuration

Seluruh augmentasi hanya dijalankan pada *split* **`train`** saat piksel masih berada pada rentang `[0, 1]`, tepat sebelum normalisasi khusus arsitektur.

- **Mekanisme Eksekusi & Kebersihan Model:**
  - Augmentasi dan normalisasi saat pelatihan dibungkus ke dalam *training-only GPU augmentation wrapper* sehingga berjalan langsung di akselerator GPU dalam presisi `float32` (menghindari *bottleneck* CPU).
  - *Checkpoint* model terbaik dan model akhir disimpan **hanya dari model inferensi inti** (*backbone + classifier head* murni tanpa layer augmentasi atau operator acak), sehingga aman untuk Grad-CAM dan konversi TFLite.

| Transformasi | Nilai Konfigurasi | Layer / Mekanisme | Keterangan & Probabilitas |
| :--- | :--- | :--- | :--- |
| **Rotasi** | `0.08` | `RandomRotation` | Sudut acak seragam $\pm 8\%$ dari $2\pi$ ($\approx \pm 28{,}8^\circ$) per citra tiap epoch |
| **Translasi (Geser)** | Lebar `0.10`, Tinggi `0.10` | `RandomTranslation` | Pergeseran horizontal & vertikal acak hingga $\pm 10\%$ |
| **Zoom** | `0.12` | `RandomZoom` | Pembesaran/pengecilan acak hingga $\pm 12\%$ |
| **Shear** | `0.0` | — | Dinonaktifkan |
| **Flip Horizontal & Vertikal** | `hflip=true`, `vflip=true` | `RandomFlip("horizontal_and_vertical")` | Probabilitas `0.5` independen pada masing-masing sumbu |
| **Fill Mode Geometri** | `reflect` | Mode pengisian batas piksel | Mengisi tepi hasil transformasi geometris secara refleksi |
| **Kecerahan (*Brightness*)** | `0.20` | `RandomBrightness` | Variasi seragam $\pm 20\%$ pada `value_range=(0.0, 1.0)` |
| **Kontras (*Contrast*)** | `0.20` | `RandomContrast` | Variasi seragam $\pm 20\%$ pada `value_range=(0.0, 1.0)` |

---

## 5. Model Configuration

Eksperimen membandingkan secara terkontrol **3 arsitektur *mobile-edge*** dengan bobot awal `imagenet` (`include_top=false`):

| Run ID (Stage 1 / Stage 2) | Arsitektur Backbone | Batas Unfreeze Stage 2 (`fine_tune_at`) |
| :--- | :--- | :---: |
| `m1_mobilenetv2_head` / `m2_mobilenetv2_ft` | `MobileNetV2` | Indeks layer `>= 107` (~30% layer teratas dibuka) |
| `m3a_effnetb0_head` / `m3b_effnetb0_ft` | `EfficientNet-B0` | Indeks layer `>= 162` (~30% layer teratas dibuka) |
| `m4a_mobilenetv3l_head` / `m4b_mobilenetv3l_ft` | `MobileNetV3-Large` | Indeks layer `>= 124` (~30% layer teratas dibuka) |

- **Arsitektur Classifier Head (Seragam di Ketiga Model):**
  1. `Backbone(inputs, training=False)`
  2. `GlobalAveragePooling2D`
  3. `Dense(256, kernel_regularizer=L2(0.0001))`
  4. `BatchNormalization`
  5. `Activation("relu")`
  6. `Dropout(0.3)`
  7. `Dense(15, activation="softmax", dtype="float32")`
- **Kebijakan Pembekuan Backbone (*Freezing Policy*):**
  - **Stage 1 (*Head Warmup*):** Seluruh backbone dibekukan (`freeze_base=true`, `fine_tune_at=0`); hanya *classifier head* yang dilatih.
  - **Stage 2 (*Partial Fine-Tuning*):** Layer backbone di bawah batas `fine_tune_at` tetap dibekukan, sedangkan layer di atasnya dibuka **kecuali seluruh layer `BatchNormalization` di dalam backbone** yang tetap dibekukan (`trainable=False`) dan dipanggil dengan `training=False` demi stabilitas statistik normalisasi *pretrained*.

---

## 6. Training Configuration

| Parameter | Stage 1 (`m1`, `m3a`, `m4a`) | Stage 2 (`m2`, `m3b`, `m4b`) | Keterangan |
| :--- | :--- | :--- | :--- |
| **Batch Size** | `64` | `64` | `32` sampel per GPU pada konfigurasi 2 GPU (`226` *step*/epoch) |
| **Epochs Maksimum** | `10` | `25` | Batas maksimum sebelum *early stopping* |
| **Optimizer** | `Adam` | `AdamW` | — |
| **Initial Learning Rate** | `1e-3` (`0.001`) | `1e-4` (`0.0001`) | — |
| **Parameter Optimizer** | `beta_1=0.9`, `beta_2=0.999`, `epsilon=1e-7` | `beta_1=0.9`, `beta_2=0.999`, `epsilon=1e-7`, `weight_decay=0.004` | *Decoupled weight decay* `0.004` aktif di Stage 2 |
| **Warm-Start Weights** | Bobot *head* acak + backbone ImageNet | *Checkpoint* terbaik (`best.keras`) dari Stage 1 | Stage 2 selalu melanjutkan dari bobot terbaik Stage 1 |
| **Gradient Clipping** | Tidak aktif | Tidak aktif | Stabilitas gradien dijaga lewat *head warmup* Stage 1, BN backbone beku, dan *learning rate* kecil (`1e-4`) |

---

## 7. Loss & Regularization Configuration

| Parameter / Teknik | Nilai Aktif | Detail Implementasi |
| :--- | :--- | :--- |
| **Fungsi Loss** | `CategoricalCrossentropy` | Menggunakan target label *one-hot* 15 kelas |
| **Label Smoothing** | `0.05` | Mengurangi *overconfidence*; batas bawah teoretis entropi silang untuk $K=15$ adalah $H(q) = 0{,}311737$ |
| **Class Weighting** | `balanced` | Pembobotan kebalikan frekuensi kelas ($w_c = \bar{N} / N_c$, rata-rata $\approx 1{,}0$) yang dihitung **eksklusif dari split `train`** |
| **Dropout** | `0.3` | Diterapkan tepat sebelum layer output klasifikasi |
| **L2 Weight Decay** | `0.0001` | Regularisasi L2 pada layer `Dense(256)` di *classifier head* |
| **Decoupled Weight Decay** | `0.004` | Aktif melalui optimizer `AdamW` pada seluruh run Stage 2 |
| **Batch Normalization** | Aktif di *head* | Diterapkan setelah `Dense(256)` sebelum aktivasi ReLU |

---

## 8. Learning Strategy Configuration

- **Two-Stage Transfer Learning:**
  1. **Stage 1 (10 epoch):** *Head warmup* dengan backbone beku penuh (`Adam`, `lr=1e-3`) agar bobot klasifikasi baru beradaptasi sebelum mengubah bobot fitur konvolusional.
  2. **Stage 2 (25 epoch):** *Partial fine-tuning* pada ~30% lapisan atas backbone (`AdamW`, `lr=1e-4`, `weight_decay=0.004`), dimulai dari bobot terbaik Stage 1.
- **Explicit Step-Based LR Warmup:** Tidak digunakan secara terpisah; peran stabilisasi awal ditangani oleh Stage 1 *frozen-head warmup*.
- **LR Scheduler (`ReduceLROnPlateau`):**
  - Memantau `val_loss` (`mode=min`), `factor=0.5`, `patience=3`, `min_lr=1e-6`, `cooldown=0`.
- **Early Stopping (`EarlyStopping`):**
  - Memantau `val_loss` (`mode=min`), `patience=6`, `min_delta=0.0001`, dan `restore_best_weights=true`.

---

## 9. Evaluation Configuration

| Aspek Evaluasi | Konfigurasi Aktif | Keterangan |
| :--- | :--- | :--- |
| **Protokol Split & Seleksi** | Seleksi pada `val`; evaluasi final `test` **tepat 1×** | Tahap pelatihan, pemilihan model, dan penyetelan hanya memakai *split* `val`. *Split* `test` dibekukan hingga akhir dan dievaluasi 1× per kandidat Stage 2 (`m2`, `m3b`, `m4b`) dengan konfigurasi beku masing-masing run |
| **Aturan Keputusan (*Threshold*)** | Multi-class `argmax` pada probabilitas softmax | `eval_batch_size=64` |
| **Metrik Klasifikasi** | `Accuracy`, `Top-3 Accuracy`, `Precision Macro`, `Recall Macro`, `F1-Macro`, `F1-Weighted`, `Worst-Class Recall` | Dilengkapi rincian metrik per kelas untuk mendeteksi performa pada kelas minoritas |
| **Kalibrasi & Uji Statistik** | `ECE` (10 bins), *5-fold OOF Temperature Scaling*, *Bootstrap 95% CI*, *Exact McNemar Test* | *Temperature scaling* dioptimalkan hanya pada logit *split* `val` untuk mengevaluasi perbaikan kalibrasi (*reliability diagram*) |
| **Error Analysis & XAI** | *Confusion Matrix*, 20 pasangan kelas paling sering tertukar, 16 sampel salah prediksi, dan **Grad-CAM** (8 sampel, $\alpha=0{,}4$) | Memverifikasi secara visual bahwa fokus model tertuju pada lesi daun, bukan latar belakang |

---

## 10. Compute Configuration

| Aspek Komputasi | Konfigurasi Akselerator Utama (2× GPU) | Fallback CPU |
| :--- | :--- | :--- |
| **Perangkat & Strategi Distribusi** | 2× NVIDIA T4 (15 GB VRAM/GPU, 4 vCPU, 30 GB RAM) dengan `MirroredStrategy (2 GPUs)` | Eksekusi single-device CPU |
| **Mixed Precision** | `mixed_float16` (komputasi Tensor Core FP16; layer augmentasi GPU dan layer output softmax dijaga pada `float32`) | `float32` |
| **Caching & Prefetching** | *In-memory caching* pasca-Epoch 1 + `prefetch=AUTOTUNE` | Tanpa cache memori, `prefetch=AUTOTUNE` |
| **Paralelisme Input Pipeline** | `num_parallel_calls=AUTOTUNE`, transformasi pasca-*batch* digabung dalam satu `.map()`, augmentasi dieksekusi di GPU | `num_parallel_calls=4` |
| **Optimasi Memori GPU** | *GPU memory growth* aktif + *lazy CUDA module loading* | — |

---

## 11. Experiment & Reproducibility Configuration

| Aspek Reproduksibilitas | Konfigurasi Aktif | Keterangan |
| :--- | :--- | :--- |
| **Global Random Seeds** | `seed=42`, `shuffle_seed=42` | Diterapkan pada `PYTHONHASHSEED`, Python `random`, NumPy, dan TensorFlow |
| **Determinisme Op GPU** | Non-deterministik (`deterministic=false`) | Memprioritaskan kecepatan kernel cuDNN/GPU; konsistensi antar eksperimen dijaga melalui *seed* dan *split* yang identik |
| **Struktur Experiment ID** | `m1_mobilenetv2_head`, `m2_mobilenetv2_ft`, `m3a_effnetb0_head`, `m3b_effnetb0_ft`, `m4a_mobilenetv3l_head`, `m4b_mobilenetv3l_ft` | Setiap run disimpan terisolasi di direktori `runs/<run_id>/` |
| **Checkpointing** | `best.keras` (`monitor=val_accuracy`, `save_best_only=true`) & `final.keras` | *Callback checkpoint* menyimpan model inferensi inti murni; bobot `best.keras` dimuat kembali sebelum evaluasi |
| **Logging & Artefak** | CSV/JSON per epoch (`history.csv`, `history.json`), per 10 *step* (`step_history.csv`), snapshot config beku (`config_used.yaml`/`.json`), metrik, & info perangkat | `step_history.csv` mencatat `global_step`, `epoch`, `step_in_epoch`, `loss`, `accuracy`, dan `learning_rate` (pembacaan LR dibatasi hanya pada *step* pencatatan agar tidak memicu sinkronisasi GPU→CPU di tiap *batch*) untuk plot *gradient descent* dengan kurva halus EMA ($\alpha=0{,}05$) |

---

## 12. Export, Quantization & Dual-Target Deployment Configuration

Konfigurasi pasca-pelatihan mendukung dua target *deployment* dari model yang sama:

- **Target 1 — Web Cloud / Server Python (`infer_fp32.keras` + modul API inferensi):**
  - *Checkpoint* pelatihan `mixed_float16` dikonversi ulang menjadi model `float32` murni bebas *optimizer* (dengan pembaruan status layer `Normalization` dan uji paritas prediksi) agar kompatibel di CPU server maupun GPU.
  - Inferensi web dikemas dengan graf terkompilasi (`@tf.function`) serta mendukung ekstraksi heatmap **Grad-CAM**.
- **Target 2 — Smartphone / Edge (`.tflite` PTQ + panduan benchmark perangkat):**
  - Model FP32 murni dikonversi ke format TensorFlow Lite (*Post-Training Quantization*) dalam 3 varian: **`FP32`**, **`FP16`**, dan full-integer **`INT8`**.
  - **Kalibrasi INT8:** Memakai dataset representatif *stratified* sebanyak `20` citra per kelas (`300` citra total) yang diambil **eksklusif dari split `train`** guna mencegah kebocoran data evaluasi.
  - **Pengukuran Kinerja:** Mengukur ukuran file (MB), retensi F1-Macro ($\Delta\text{F1}$ terhadap FP32), ECE, serta proksi latency CPU (`num_threads=4`, `warmup=10`, `runs=50` untuk `p50`/`p95`/`mean` ms), disertai panduan perintah ADB `benchmark_model` untuk pengukuran latency asli pada perangkat Android.
- **Paket Deliverable Akhir:**
  - Bundel inferensi siap pakai (model web FP32, API inferensi, model TFLite, label kelas, metadata, dan panduan benchmark Android) serta bundel laporan metrik, tabel perbandingan, kurva *gradient descent*, *confusion matrix*, kurva kalibrasi, dan visualisasi Grad-CAM.
