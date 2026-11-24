# Experiment Protocol — PlantVile (PlantVillage CNN)

Dokumen ini adalah protokol eksperimen tertulis untuk proyek **PlantVile** yang menghubungkan prinsip *Fondasi Eksperimen Deep Learning* dengan implementasi nyata di `main.py`, `configs/default.yaml`, dan modul `src/`.

> **Aturan Utama:** Seluruh keputusan desain, pemilihan arsitektur, dan tuning hyperparameter **wajib** dievaluasi pada split **`val`** (`--set eval.eval_split=val`). Split **`test`** dibekukan hingga tahap akhir (*Final Evaluation*) dan hanya dibuka **satu kali** untuk kandidat model final.

---

## 1. Problem Formulation & Batasan Sistem

### 1.1 Formulasi Masalah ML
| Komponen | Definisi |
| :--- | :--- |
| **Masalah Nyata** | Klasifikasi otomatis penyakit daun tanaman dari citra untuk aplikasi/UI tugas akhir Neuro Computing |
| **Formulasi Formal** | $f_\theta: X \rightarrow Y$, memetakan citra daun RGB $X \in \mathbb{R}^{H \times W \times 3}$ ke label kategori penyakit $Y \in \{0, \dots, C-1\}$ |
| **Tipe Task** | *Supervised Multi-Class Image Classification* |
| **Skala Kelas** | Dinamis berdasarkan folder (`scan_dataset()` di `src/data.py`): 15 kelas atau 38 kelas PlantVillage |
| **Representasi Target** | *One-hot vector* untuk `categorical_crossentropy` / `focal` (default), atau integer untuk `sparse_categorical_crossentropy` |

### 1.2 Constraints Operasional
1. **Workflow Terpisah (Lokal vs Kaggle):** Pengembangan kode dan uji alur (`--check` dengan dataset dummy) dilakukan di lokal CPU; pelatihan penuh dijalankan di GPU Kaggle (`/kaggle/input/...` → `/kaggle/working/runs`).
2. **Ketimpangan Kelas (*Class Imbalance*):** Pada snapshot 15 kelas di `README.md`, rasio kelas terbesar/terkecil mencapai **21x** (`3.208` vs `152` gambar pada `Potato___healthy`).
3. **Risiko *Distribution Shift*:** Dataset PlantVillage mayoritas diambil pada latar seragam/terkontrol. Performa tinggi pada split `val`/`test` PlantVillage **tidak otomatis** menjamin performa yang sama pada foto daun berlatar kompleks dari kamera pengguna di aplikasi/UI.
4. **Batasan Aplikasi/UI Target:** Model terpilih harus mudah dimuat untuk inferensi lokal (asumsi target utama: Web App Python lokal seperti Streamlit/Gradio atau REST API; opsi Mobile TFLite bila diperlukan), sehingga ukuran file `.keras`, kecepatan inferensi CPU, dan konsistensi *preprocessing* menjadi syarat kelayakan.

---

## 2. Protokol Audit Dataset Kaggle (Pra-Training)

Sebelum menjalankan eksperimen training di Kaggle, jalankan audit dataset satu kali pada `/kaggle/input/<dataset-path>` (misal lewat sel Python di Kaggle Notebook atau bagian EDA di `notebook/plantvillage_cnn.ipynb`):

### Checklist Audit Wajib
1. **Verifikasi Struktur & Jumlah Kelas:**
   - Pastikan `resolve_variant_dir()` di `src/data.py` mengarah ke folder kelas yang benar (`flat` `<root>/<Class>/` atau `bertingkat` `<root>/color/<Class>/`), terutama jika dataset Kaggle memuat keduanya.
   - Catat jumlah kelas aktual ($C$) dan total gambar valid (`IMG_EXTS`: `.jpg`, `.jpeg`, `.png`, `.bmp`).
2. **Audit Distribusi Kelas & Rasio Imbalance:**
   - Hitung jumlah file per kelas, nilai minimum, maksimum, median, dan rasio $\text{max}/\text{min}$.
   - Identifikasi kelas minoritas (contoh: `Potato___healthy` = 152 gambar → pada split 70/15/15 hanya menghasilkan ~23 gambar `val` dan ~23 gambar `test`).
3. **Audit Integritas Fisik Citra:**
   - Periksa apakah ada file 0-byte, non-citra (seperti `svn-r6Yb5c`), atau citra korup yang gagal dibuka oleh `PIL.Image.open()`.
   - Verifikasi mode warna (`RGB` vs `RGBA`/`L`) dan variasi resolusi asli gambar (umumnya `256x256` pada PlantVillage).
4. **Audit Duplikasi Citra (*Anti-Leakage*):**
   - Hitung hash file (misal `hashlib.md5`) untuk mendeteksi gambar identik di dalam satu kelas atau lintas kelas.
   - Jika ditemukan duplikat persis antar-sampel, catat jumlahnya karena duplikat yang terpecah ke `train` dan `val`/`test` menyebabkan *data leakage* dan metrik terlalu optimistis.
5. **Inspeksi Visual & Bias Sampling:**
   - Tampilkan sampel acak per kelas untuk memeriksa *label noise*, artefak latar belakang seragam, atau orientasi gambar (dasar pemilihan parameter augmentasi).

### Audit Dataset Otomatis via CLI (Lokal & Kaggle)
Selain inspeksi manual di `notebook/data_audit_eda.ipynb`, gunakan skrip ringan *read-only* berikut sebelum training:
```bash
python scripts/audit_dataset.py --root $DATA_ROOT --out $RUNS_DIR/dataset_audit.json
```
*(Sebelum `git push` dari lokal ke GitHub, jalankan `python scripts/smoke_test.py` dan `pytest tests/test_error_analysis.py -v` untuk memverifikasi seluruh alur pipeline di CPU).*

---

## 3. Validation Design & Aturan `eval.eval_split=val` vs `test`

### 3.1 Protokol Pemisahan Data
Codebase membagi data per kelas (`data.split_strategy=stratified`) di `src/data.py:79-112` dengan proporsi default:
- **Training (`70%`):** Pembaruan parameter model dan perhitungan `class_weight`.
- **Validation (`15%`, `data.val_split=0.15`):** *Early stopping*, *learning rate scheduling*, seleksi arsitektur, dan seluruh perbandingan eksperimen.
- **Test (`15%`, `data.test_split=0.15`):** Estimasi performa generalisasi akhir **satu kali** setelah seluruh keputusan eksperimen dibekukan.

> **Invariansi Split:** Pastikan `data.shuffle_seed=42`, `data.val_split=0.15`, `data.test_split=0.15`, `data.split_strategy=stratified`, dan `data.max_per_class=0` tetap **identik** di seluruh eksperimen utama agar setiap model dibandingkan pada partisi data yang persis sama.

### 3.2 Aturan Ketat Isolasi `test` Set
`configs/default.yaml` secara bawaan menyetel `eval.eval_split: test`. Untuk mencegah *test-set leakage* selama eksplorasi:
1. **Selama Eksperimen (`exp00` s.d. `exp06`):** **Wajib** selalu menyertakan `--set eval.eval_split=val`. Hasil di folder run akan tertulis sebagai `val_metrics.json`, `val_confusion_matrix.png`, dll.
2. **Tahap Final (Sekali Saja):** Setelah satu konfigurasi kandidat terbaik diputuskan berdasarkan `val_metrics.json` dan analisis error, jalankan evaluasi akhir pada `test` set menggunakan snapshot `config_used.yaml` dari run tersebut:
   ```bash
   python main.py --eval /kaggle/working/runs/<final_run>/best.keras \
     --config /kaggle/working/runs/<final_run>/config_used.yaml \
     --set eval.eval_split=test
   ```
3. **Diagnosis Overfitting/Underfitting (Opsional):** Jika perlu mengecek performa di data latih secara terpisah, gunakan `--set eval.eval_split=train` dengan `--eval`.

### 3.3 Pengujian Multi-Seed (Kandidat Final)
Satu kali *run* memiliki variasi stokastik (inisialisasi classifier head, pengacakan batch, augmentasi, dan non-determinisme GPU). Untuk konfigurasi terbaik yang lolos seleksi Track A–E:
- Jalankan ulang dengan minimal **3 seed**: `runtime.seed=42`, `123`, dan `2024` (tetap dievaluasi pada `eval.eval_split=val`, dengan `data.shuffle_seed=42` dijaga tetap agar split tidak berubah).
- Laporkan performa sebagai **Mean ± Standard Deviation** untuk `f1_macro` dan `accuracy`.

---

## 4. Baseline & Matriks Eksperimen Terkontrol untuk Kaggle

Semua perintah di bawah mengasumsikan direktori kerja di root repo (`PlantVile/`) di Kaggle Notebook dengan dataset di `$DATA_ROOT` dan output di `/kaggle/working/runs`. Gunakan flag `--yes` agar `main.py` melewati konfirmasi interaktif `[y/N]` secara aman di sel Kaggle.

Set variabel shell di Kaggle terlebih dahulu:
```bash
export DATA_ROOT=/kaggle/input/datasets/emmarex/plantdisease/PlantVillage
export RUNS_DIR=/kaggle/working/runs
```

> **Notebook Siap Pakai di Kaggle:** Seluruh alur di bawah (clone repo → audit → `--check` → matriks eksperimen → leaderboard → inspeksi Grad-CAM → final `test` → zip artefak) juga tersedia dalam bentuk notebook di [notebook/kaggle_experiment_runner.ipynb](notebook/kaggle_experiment_runner.ipynb) (di-generate lewat `scripts/build_kaggle_notebook.py`).

### Optimasi Hardware Kaggle: T4×2 GPU vs Kaggle CPU
- **Mode Kaggle GPU T4×2 (Disarankan untuk Eksperimen Penuh):** Gunakan preset `configs/kaggle_t4x2.yaml` yang mengaktifkan `mixed_precision=mixed_float16`, `cache=memory`, `prefetch=0` (`AUTOTUNE`), `num_parallel_calls=0` (`AUTOTUNE`), `batch_size=64`, dan `runtime.strategy=auto` (`MirroredStrategy` jika 2 GPU terlihat).
  - *Alternatif Parallel Single-GPU:* Untuk eksperimen ringan (`mobilenetv2` / `128px`), Anda juga dapat menjalankan 2 eksperimen berbeda secara paralel per GPU dengan `CUDA_VISIBLE_DEVICES=0 ... --set runtime.strategy=none` dan `CUDA_VISIBLE_DEVICES=1 ... --set runtime.strategy=none`.
- **Mode Kaggle CPU (4 vCPU, 30 GB RAM):** Gunakan `configs/default.yaml` dengan `runtime.device=cpu`, `train.mixed_precision=off`, `data.cache=memory`, `data.batch_size=32`, dan `runtime.num_parallel_calls=4`.

### 4.0 Pilot Check (Smoke Test — Tanpa Hasil Ilmiah)
Tujuan: Memastikan GPU terdeteksi, struktur folder terbaca, dan pipeline tidak error sebelum menjalankan training penuh.
```bash
# 1. Validasi config & pipeline tanpa training
python main.py --check --config configs/default.yaml \
  --set data.root=$DATA_ROOT \
  --set runtime.device=gpu

# 2. Mini training 2 epoch (subset 200 gambar/kelas) untuk cek kecepatan GPU & I/O
python main.py --yes --config configs/default.yaml \
  --set data.root=$DATA_ROOT \
  --set data.max_per_class=200 \
  --set train.epochs=2 \
  --set eval.eval_split=val \
  --set runtime.device=gpu \
  --set runtime.runs_dir=$RUNS_DIR \
  --set runtime.run_name=pilot_smoke
```

### 4.1 Matriks Eksperimen Terkontrol (OFAT & Controlled Comparison)

Setiap eksperimen hanya mengubah **1 faktor utama** terhadap kelompok kontrolnya (*controlled variables* dijaga konstan):

| Run ID | Pertanyaan Riset & Hipotesis | Variabel Bebas (Diubah) | Kelompok Kontrol / Acuan | Variabel Terkontrol (Tetap) |
| :--- | :--- | :--- | :--- | :--- |
| **`exp00a_simple_cnn`** | **Scratch Baseline:** Berapa batas bawah performa CNN dari nol tanpa *transfer learning* dan tanpa augmentasi? | `model.arch=simple_cnn`, `data.augment=false`, `data.normalize=div255` | — | `128x128`, `adam`, `lr=1e-3`, `epochs=20`, `balanced`, `seed=42` |
| **`exp00b_mobilenet_ctrl`** | **Transfer Control Baseline:** Seberapa besar kenaikan dari bobot ImageNet *frozen* (`MobileNetV2`) tanpa augmentasi? | `model.arch=mobilenetv2`, `data.augment=false`, `data.normalize=imagenet` | `exp00a_simple_cnn` | `128x128`, `freeze_base=true`, `adam`, `lr=1e-3`, `epochs=20`, `balanced` |
| **`exp01_aug`** | **Track A (Augmentasi):** Apakah augmentasi default mengurangi *overfitting* (gap train–val) dan menaikkan `val f1_macro`? | `data.augment=true` | `exp00b_mobilenet_ctrl` | `mobilenetv2`, `128x128`, `balanced`, `adam`, `lr=1e-3`, `seed=42` |
| **`exp02a_noweight`** | **Track B1 (Imbalance):** Apakah `class_weight=balanced` benar-benar membantu recall kelas minoritas dibanding tanpa pembobotan? | `train.class_weight=none` | `exp01_aug` | `mobilenetv2`, `128x128`, `augment=true`, `loss=categorical_crossentropy` |
| **`exp02b_focal`** | **Track B2 (Imbalance):** Apakah `focal` loss lebih baik atau lebih stabil daripada `class_weight=balanced`? | `train.loss=focal`, `train.class_weight=none` | `exp01_aug` | `mobilenetv2`, `128x128`, `augment=true`, `seed=42` |
| **`exp03_res224`** | **Track C (Resolusi):** Apakah menaikkan resolusi ke `224x224` membantu menangkap gejala bercak daun halus pada `val f1_macro`? | `data.img_height=224`, `data.img_width=224` | Pemenang Track A–B | Kebijakan augmentasi & loss pemenang Track A–B, `mobilenetv2` |
| **`exp04a_resnet50`** | **Track D1 (Arsitektur):** Apakah backbone `resnet50` mengungguli `mobilenetv2` pada kondisi kontrol yang sama? | `model.arch=resnet50` | Pemenang Track C | Resolusi, augmentasi, & loss pemenang Track A–C, `freeze_base=true` |
| **`exp04b_effnetb0`** | **Track D2 (Arsitektur):** Apakah backbone `efficientnetb0` lebih akurat/efisien dibanding `mobilenetv2` dan `resnet50`? | `model.arch=efficientnetb0` | Pemenang Track C | Resolusi, augmentasi, & loss pemenang Track A–C, `freeze_base=true` |
| **`exp05_finetune`** | **Track E (Fine-Tuning):** Apakah membuka sebagian layer atas backbone (`fine_tune_at`) + `adamw` (`lr=1e-4`) meningkatkan `val f1_macro`? | `model.fine_tune_at=<cut>`, `train.optimizer=adamw`, `train.learning_rate=0.0001` | Pemenang Track D | Arsitektur, resolusi, augmentasi, & loss pemenang Track D (`freeze_base=true`) |
| **`exp06_multiseed`** | **Track F (Stabilitas):** Seberapa konsisten kandidat final pada 3 seed berbeda (`42`, `123`, `2024`)? | `runtime.seed=123`, `runtime.seed=2024` | Konfigurasi pemenang Track E (atau D) | Seluruh konfigurasi lain dikunci (`data.shuffle_seed=42` tetap) |

> **Catatan `model.fine_tune_at` pada Track E (`src/model.py:59-67`):** `fine_tune_at > 0` hanya aktif apabila `model.freeze_base=true` tetap dipertahankan (kode secara otomatis menjaga seluruh layer `BatchNormalization` di backbone tetap *frozen* agar training stabil). Cek total layer arsitektur pemenang via `python main.py --check` sebelum menentukan indeks `fine_tune_at` (misal `100` untuk `mobilenetv2` yang memiliki ~154 layer).

### 4.2 Perintah CLI Siap Salin per Eksperimen

```bash
# --- EXP00a: Simple CNN Baseline ---
python main.py --yes --config configs/default.yaml \
  --set data.root=$DATA_ROOT --set runtime.runs_dir=$RUNS_DIR --set runtime.device=gpu \
  --set eval.eval_split=val --set model.arch=simple_cnn \
  --set data.augment=false --set data.normalize=div255 \
  --set runtime.run_name=exp00a_simple_cnn

# --- EXP00b: MobileNetV2 Control Baseline (Tanpa Augmentasi) ---
python main.py --yes --config configs/default.yaml \
  --set data.root=$DATA_ROOT --set runtime.runs_dir=$RUNS_DIR --set runtime.device=gpu \
  --set eval.eval_split=val --set model.arch=mobilenetv2 \
  --set data.augment=false --set runtime.run_name=exp00b_mobilenet_ctrl

# --- EXP01: Track A - Efek Data Augmentation ---
python main.py --yes --config configs/default.yaml \
  --set data.root=$DATA_ROOT --set runtime.runs_dir=$RUNS_DIR --set runtime.device=gpu \
  --set eval.eval_split=val --set model.arch=mobilenetv2 \
  --set data.augment=true --set runtime.run_name=exp01_aug

# --- EXP02a: Track B1 - Tanpa Class Weight ---
python main.py --yes --config configs/default.yaml \
  --set data.root=$DATA_ROOT --set runtime.runs_dir=$RUNS_DIR --set runtime.device=gpu \
  --set eval.eval_split=val --set train.class_weight=none \
  --set runtime.run_name=exp02a_noweight

# --- EXP02b: Track B2 - Focal Loss ---
python main.py --yes --config configs/default.yaml \
  --set data.root=$DATA_ROOT --set runtime.runs_dir=$RUNS_DIR --set runtime.device=gpu \
  --set eval.eval_split=val --set train.class_weight=none --set train.loss=focal \
  --set runtime.run_name=exp02b_focal

# --- EXP03: Track C - Resolusi 224x224 (sesuaikan pemenang Track B) ---
python main.py --yes --config configs/default.yaml \
  --set data.root=$DATA_ROOT --set runtime.runs_dir=$RUNS_DIR --set runtime.device=gpu \
  --set eval.eval_split=val --set data.img_height=224 --set data.img_width=224 \
  --set runtime.run_name=exp03_res224

# --- EXP04a & EXP04b: Track D - Perbandingan Arsitektur (sesuaikan resolusi pemenang Track C) ---
python main.py --yes --config configs/default.yaml \
  --set data.root=$DATA_ROOT --set runtime.runs_dir=$RUNS_DIR --set runtime.device=gpu \
  --set eval.eval_split=val --set model.arch=resnet50 \
  --set data.img_height=224 --set data.img_width=224 \
  --set runtime.run_name=exp04a_resnet50

python main.py --yes --config configs/default.yaml \
  --set data.root=$DATA_ROOT --set runtime.runs_dir=$RUNS_DIR --set runtime.device=gpu \
  --set eval.eval_split=val --set model.arch=efficientnetb0 \
  --set data.img_height=224 --set data.img_width=224 \
  --set runtime.run_name=exp04b_effnetb0

# --- EXP05: Track E - Partial Fine-Tuning (contoh di MobileNetV2, sesuaikan pemenang Track D) ---
python main.py --yes --config configs/default.yaml \
  --set data.root=$DATA_ROOT --set runtime.runs_dir=$RUNS_DIR --set runtime.device=gpu \
  --set eval.eval_split=val --set model.arch=mobilenetv2 \
  --set data.img_height=224 --set data.img_width=224 \
  --set model.freeze_base=true --set model.fine_tune_at=100 \
  --set train.optimizer=adamw --set train.learning_rate=0.0001 \
  --set train.epochs=30 --set runtime.run_name=exp05_finetune

# --- EXP06: Track F - Multi-Seed pada Kandidat Final (jalankan juga seed=2024) ---
python main.py --yes --config $RUNS_DIR/<best_run>/config_used.yaml \
  --set runtime.seed=123 --set runtime.run_name=exp06_seed123
```

---

## 5. Kriteria Evaluasi & Pemilihan Model untuk Aplikasi / UI

Model dengan `accuracy` tertinggi belum tentu layak untuk aplikasi/UI maupun ilmiah. Pemilihan model final ditentukan oleh dua lapis kriteria:

### 5.1 Kriteria Ilmiah (`val_metrics.json`, Error Analysis & Grad-CAM)
1. **Metrik Primer — `f1_macro` pada `val`:** Memberikan bobot setara bagi seluruh kelas, termasuk kelas minoritas (`Potato___healthy` 152 gambar), sehingga model tidak sekadar menghafal kelas mayoritas.
2. **Metrik Sekunder & Error Analysis:**
   - **Worst-Class Recall & F1 (`val_per_class.csv`):** Pastikan tidak ada satu kelas penyakit yang kolaps (misalnya recall jauh lebih rendah dibanding kelas lain).
   - **Top-K Accuracy & Kalibrasi Confidence (`val_metrics.json`, `val_confidence_calibration.png`):** Pantau `top_k_accuracy` dan **ECE (Expected Calibration Error)** agar probabilitas yang ditampilkan di UI tidak *overconfident* pada prediksi salah.
   - **Top Confused Pairs & High-Confidence Errors (`val_confused_pairs.csv`, `val_misclassified.csv`, `val_misclassified_grid.png`):** Periksa pasangan penyakit yang paling sering tertukar dan sampel salah prediksi dengan confidence tertinggi.
   - **Audit Grad-CAM (`val_gradcam_grid.png`):** Verifikasi bahwa aktivasi berfokus pada **gejala bercak/jaringan daun**, bukan pada latar belakang studio (*shortcut learning*).
   - **Generalization Gap (`history.csv`):** Amati selisih `accuracy - val_accuracy` dan bentuk kurva `val_loss` pada `val_loss_accuracy.png`. Model dengan `val f1_macro` stabil dan gap kecil lebih diprioritaskan daripada model dengan training accuracy 99.9% namun `val_loss` menanjak.
   - **Stabilitas Multi-Seed (`exp06`):** Variasi (`SD`) antar-seed rendah dan peningkatan terhadap kontrol (`exp00b`) konsisten di seluruh seed.

### 5.2 Kriteria Kelayakan Aplikasi / UI
1. **Trade-off Akurasi vs Ukuran & Latensi Inferensi:**
   - Jika selisih `val f1_macro` antara model berat (`resnet50`/`vgg16`) dan model ringan (`mobilenetv2`/`efficientnetb0`) kecil (< 0.01–0.02), **pilih model yang lebih ringan** untuk menjaga responsivitas UI lokal dan mempermudah opsi ekspor ke TFLite bila dibutuhkan.
2. **Kontrak Artefak Serah Terima (`runs/<final_run>/`):**
   Aplikasi/UI wajib menerima paket artefak yang lengkap dan berasal dari *run* yang sama:
   - `best.keras` (bobot model terbaik berdasarkan `checkpoint_monitor`)
   - `class_names.json` (pemetaan indeks output softmax → nama kelas)
   - `config_used.json` (sumber kebenaran `img_height`, `img_width`, `channels`, `rescale`, `normalize`, dan `model.arch`)
3. **Invariansi Preprocessing di Aplikasi/UI:**
   - Kode inferensi UI **wajib** mereplikasi urutan preprocessing yang identik dengan `src/data.py:122-159` berdasarkan nilai di `config_used.json` (khususnya `preprocess_input` sesuai `model.arch` bila `normalize=imagenet`, karena rentang input `mobilenetv2` `[-1, 1]`, `resnet50`/`vgg16` Caffe BGR zero-centered, dan `efficientnetb0` `[0, 255]` berbeda satu sama lain).
4. **Uji Realistis Tambahan Sebelum Demo Aplikasi:**
   - Selain `test_metrics.json` (yang dievaluasi 1x di akhir), lakukan uji kualitatif pada beberapa gambar daun dari sumber luar/kamera HP di UI untuk mendokumentasikan keterbatasan *domain shift* di laporan UAS.

---

## 6. Experiment Log & Template Pelaporan

Salin dan isi tabel ini setelah setiap run di Kaggle selesai agar seluruh eksperimen dapat diaudit dan dibandingkan secara adil:

| Run ID | Hipotesis / Perubahan | `eval_split` | Seed | `f1_macro` | `accuracy` | Worst-Class Recall | Best Epoch | Ukuran `best.keras` | Catatan / Keputusan Next Step |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `exp00a_simple_cnn` | Scratch baseline (no aug) | `val` | 42 | | | | | | |
| `exp00b_mobilenet_ctrl` | Transfer baseline (no aug) | `val` | 42 | | | | | | |
| `exp01_aug` | + Data augmentation | `val` | 42 | | | | | | |
| `exp02a_noweight` | `class_weight=none` | `val` | 42 | | | | | | |
| `exp02b_focal` | `loss=focal` | `val` | 42 | | | | | | |
| `exp03_res224` | `224x224` resolution | `val` | 42 | | | | | | |
| `exp04a_resnet50` | Backbone `resnet50` | `val` | 42 | | | | | | |
| `exp04b_effnetb0` | Backbone `efficientnetb0` | `val` | 42 | | | | | | |
| `exp05_finetune` | `fine_tune_at` + `adamw` | `val` | 42 | | | | | | |
| `exp06_seed123` | Multi-seed check | `val` | 123 | | | | | | |
| `exp06_seed2024` | Multi-seed check | `val` | 2024 | | | | | | |
| **`FINAL_TEST`** | **1x evaluasi model final** | **`test`** | **42** | | | | **—** | | **Kunci untuk UI & Laporan** |
