# Laporan Audit Data & EDA — PlantVillage (15 Kelas)

Dokumen ini merangkum hasil audit integritas data dan *Exploratory Data Analysis* (EDA) pada dataset `data/PlantVillage` yang digunakan dalam proyek **PlantVile**, beserta hipotesis eksperimen yang diturunkan dari bukti empiris data. Seluruh angka di bawah dapat direproduksi menggunakan notebook [notebook/data_audit_eda.ipynb](../notebook/data_audit_eda.ipynb).

---

## 1. Ringkasan Eksekutif & Status Struktur Dataset

- `[Fakta Terverifikasi]` **Struktur Saat Ini (`data/PlantVillage`):** Direktori lokal kini bersih berisi tepat **15 subfolder kelas**, **20.638 citra valid** (`326,38 MB`), dan **1 file non-citra 0-byte** (`svn-r6Yb5c`).
- `[Fakta Terverifikasi]` **Temuan & Pembersihan Folder Duplikat Bersarang:** Sebelum pembersihan, di dalam `data/PlantVillage/` terdapat subfolder bersarang `PlantVillage/` berisi 15 folder kelas yang identik (`20.639 / 20.639` file cocok nama dan ukuran byte-nya, serta `30 / 30` spot-check MD5 identik). Folder duplikat bersarang tersebut berpotensi memicu error pada `scan_dataset()` di `src/data.py:44-71` (menganggap `PlantVillage` sebagai folder kelas kosong). Folder bersarang tersebut telah dihapus oleh pengguna di lokal sehingga menghemat ~326 MB.
- **Catatan Penting untuk Kaggle:** Pada dataset `emmarex/plantdisease` di `/kaggle/input/...` (bersifat *read-only*), struktur *double-nesting* bawaan tidak bisa dihapus langsung. Oleh karena itu, saat menjalankan training di Kaggle, pastikan `data.root` diarahkan langsung ke folder terdalam yang berisi folder-folder kelas (atau tambahkan proteksi di `scan_dataset()` untuk mengabaikan subfolder tanpa file citra).

---

## 2. Audit Integritas & Kualitas Fisik Citra (`[Fakta Terverifikasi]`)

| Aspek Audit | Hasil Terukur | Dampak / Implikasi Pipeline |
| :--- | :--- | :--- |
| **Jumlah Kelas & Citra** | **15 kelas**, **20.638 citra valid** (`326,38 MB`) | Subset 3 spesies (Tomat, Cabai, Kentang), bukan 38 kelas PlantVillage penuh |
| **Distribusi Ekstensi** | `.JPG` (`20.482`), `.jpg` (`154`), `.png` (`1`), `.jpeg` (`1`), tanpa ekstensi (`1`) | 1 file non-citra 0-byte (`Tomato__Tomato_YellowLeaf__Curl_Virus/svn-r6Yb5c`) otomatis diabaikan oleh filter `IMG_EXTS` di `src/data.py:9` |
| **Ukuran File (Bytes)** | Min `3.440 B` · P5 `9.002 B` · Median `17.000 B` · P95 `23.979 B` · Max `121.652 B` | Outlier terbesar sesuai keberadaan 1 file `.png` |
| **Verifikasi PIL (`1.528` file)** *(100/kelas stratifikasi + seluruh size outliers `<5 KB` & `>40 KB`)* | **0 file korup**, resolusi seragam **`256 × 256` px**, mode **`RGB` (`1.527`)** dan **`RGBA` (`1`)** | `tf.io.decode_image(..., channels=3)` di `src/data.py:124` aman mengonversi 1 citra `RGBA` ke 3 channel RGB |

---

## 3. Audit Duplikasi & Risiko Kebocoran Data (*Leakage*)

`[Fakta Terverifikasi]`
- **Exact MD5 Duplicate Check** (pada seluruh `14.158` kandidat file yang memiliki pasangan ukuran byte sama dari `20.638` citra):
  - Ditemukan **14 grup duplikat persis** (**14 file ekstra**).
  - **14 grup *within-class*** dan **0 grup *cross-class*** → tidak ada konflik label antar-kelas.
  - Contoh pasangan duplikat:
    - `Tomato_healthy`: `GH_HL Leaf 342.1.JPG`, `GH_HL Leaf 466.1.JPG`, `GH_HL Leaf 220.JPG` (masing-masing muncul 2 kali dengan UUID berbeda).
    - `Tomato_Late_blight`: `GHLB2 Leaf 9011.JPG`, `GHLB_PS Leaf 24 Day 16.jpg` (masing-masing muncul 2 kali dengan UUID berbeda).
- **Same-Leaf Near-Duplicates** *(pola penamaan `Leaf X.1`, `Leaf X.2`, dst.)*:
  - `Tomato_Late_blight`: **54 grup** (**195 file**).
  - `Tomato_healthy`: **47 grup** (**95 file**).
- **Pre-Augmentasi Sintetis:** Ditemukan **0 file** dengan pola nama augmentasi bawaan (`_flip`, `_rot`, `180deg`, `90deg`, `270deg`, `final_masked`). Dataset merupakan citra mentah PlantVillage.

**Implikasi Risiko:** Meskipun tidak ditemukan *cross-class label conflict*, *random stratified split* (`src/data.py:79-112`) dapat memecah 14 pasang duplikat MD5 dan grup *same-leaf* ke dalam `train`, `val`, dan `test`, sehingga berpotensi menimbulkan kebocoran minor.

---

## 4. Distribusi Kelas, Komposisi Spesies, & Simulasi Split 70/15/15 (`[Fakta Terverifikasi]`)

**Komposisi per Spesies:** Tomato `16.011` (**77,6%**, 10 kelas) · Pepper `2.475` (**12,0%**, 2 kelas) · Potato `2.152` (**10,4%**, 3 kelas).

Simulasi split mengikuti implementasi aktual `split_dataset()` (`stratified`, `val_split=0.15`, `test_split=0.15`, `shuffle_seed=42`) dan `compute_class_weights(..., mode="balanced")` (`mean_train / n_train`) di `src/data.py`:

| Kelas | Total | Proporsi | Train (70%) | Val (15%) | Test (15%) | `BalWt` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Tomato__Tomato_YellowLeaf__Curl_Virus` | 3.208 | 15,54% | 2.246 | 481 | 481 | 0,43 |
| `Tomato_Bacterial_spot` | 2.127 | 10,31% | 1.489 | 319 | 319 | 0,65 |
| `Tomato_Late_blight` | 1.909 | 9,25% | 1.337 | 286 | 286 | 0,72 |
| `Tomato_Septoria_leaf_spot` | 1.771 | 8,58% | 1.239 | 266 | 266 | 0,78 |
| `Tomato_Spider_mites_Two_spotted_spider_mite` | 1.676 | 8,12% | 1.174 | 251 | 251 | 0,82 |
| `Tomato_healthy` | 1.591 | 7,71% | 1.113 | 239 | 239 | 0,87 |
| `Pepper__bell___healthy` | 1.478 | 7,16% | 1.034 | 222 | 222 | 0,93 |
| `Tomato__Target_Spot` | 1.404 | 6,80% | 982 | 211 | 211 | 0,98 |
| `Potato___Early_blight` | 1.000 | 4,85% | 700 | 150 | 150 | 1,38 |
| `Potato___Late_blight` | 1.000 | 4,85% | 700 | 150 | 150 | 1,38 |
| `Tomato_Early_blight` | 1.000 | 4,85% | 700 | 150 | 150 | 1,38 |
| `Pepper__bell___Bacterial_spot` | 997 | 4,83% | 697 | 150 | 150 | 1,38 |
| `Tomato_Leaf_Mold` | 952 | 4,61% | 666 | 143 | 143 | 1,45 |
| `Tomato__Tomato_mosaic_virus` | 373 | 1,81% | 261 | 56 | 56 | 3,69 |
| **`Potato___healthy`** | **152** | **0,74%** | **106** | **23** | **23** | **9,08** |

- **Rasio Imbalance:** **21,1×** (`3.208` berbanding `152`).
- **Sensitivitas Evaluasi Kelas Minoritas:** Pada `Potato___healthy`, jumlah sampel di `val` maupun `test` hanya **23 gambar** — satu kesalahan prediksi langsung menurunkan recall kelas tersebut sebesar **4,35 poin persentase**.
- **Efek Pembobotan:** `class_weight=balanced` memberi bobot **9,08×** pada `Potato___healthy` (dan **3,69×** pada `Tomato__Tomato_mosaic_virus`) dibanding **0,43×** pada `Tomato__Tomato_YellowLeaf__Curl_Virus`.

---

## 5. EDA Piksel, Latar Belakang (*Border vs Center*), & Confounding Sumber

### 5.1 Tag Sumber Pengambilan pada Nama File (`[Fakta Terverifikasi]`)
- Mayoritas kelas berasal dari 1–2 tag sumber/kolektor yang spesifik (contoh: `Crnl_L.Mold` pada `Tomato_Leaf_Mold`, `PSU_CG` pada `Tomato__Tomato_mosaic_virus`, `Com.G_SpM_FL` dan `Com.G_TgS_FL`, `UF.GRC_YLCV_Lab`, `RS_*`, `JR_*`).
- Ditemukan 1 anomali nama file sumber web pada `Pepper__bell___healthy`: `bell-pepper-plant-61726.jpg`.

### 5.2 Statistik Piksel & Kontras Latar Tepi (1.500 Sampel Stratifikasi: 100/Kelas, `seed=42`)

Metrik dihitung pada citra RGB terskala `[0, 1]`: luminansi `0,299 R + 0,587 G + 0,114 B`, tepi (`Brd*`) diambil dari pita pinggir 16 piksel, dan tengah (`Ctr*`) dari area 50% tengah citra (`[Fakta Terverifikasi]`):

| Kelas | `Bright` | `Contr` | `R / G / B mean` | `BrdStd` | `CtrStd` | `BrdLum` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Pepper__bell___Bacterial_spot` | 0,468 | 0,197 | `0,47 / 0,48 / 0,39` | 0,154 | 0,166 | 0,554 |
| `Pepper__bell___healthy` | 0,503 | 0,180 | `0,47 / 0,53 / 0,44` | 0,145 | 0,127 | 0,604 |
| `Potato___Early_blight` | 0,535 | 0,189 | `0,53 / 0,55 / 0,46` | 0,174 | 0,153 | 0,593 |
| `Potato___Late_blight` | 0,462 | 0,155 | `0,47 / 0,47 / 0,41` | 0,113 | 0,132 | 0,548 |
| `Potato___healthy` | 0,506 | 0,161 | `0,49 / 0,52 / 0,45` | 0,112 | 0,134 | 0,598 |
| `Tomato_Bacterial_spot` | 0,408 | 0,132 | `0,40 / 0,42 / 0,36` | 0,097 | 0,082 | 0,497 |
| `Tomato_Early_blight` | 0,435 | 0,184 | `0,43 / 0,44 / 0,40` | 0,148 | 0,146 | 0,531 |
| `Tomato_Late_blight` | 0,450 | 0,147 | `0,45 / 0,46 / 0,41` | 0,114 | 0,136 | 0,478 |
| `Tomato_Leaf_Mold` | 0,444 | 0,140 | `0,43 / 0,45 / 0,45` | 0,099 | 0,129 | 0,502 |
| `Tomato_Septoria_leaf_spot` | 0,518 | 0,155 | `0,52 / 0,53 / 0,47` | 0,103 | 0,140 | 0,584 |
| `Tomato_Spider_mites_Two_spotted_spider_mite` | 0,468 | 0,178 | `0,48 / 0,47 / 0,42` | 0,126 | 0,166 | 0,550 |
| `Tomato__Target_Spot` | 0,471 | 0,176 | `0,47 / 0,47 / 0,44` | 0,137 | 0,157 | 0,547 |
| `Tomato__Tomato_YellowLeaf__Curl_Virus` | 0,444 | 0,132 | `0,43 / 0,47 / 0,34` | 0,093 | 0,115 | 0,491 |
| `Tomato__Tomato_mosaic_virus` | 0,466 | 0,139 | `0,47 / 0,46 / 0,47` | 0,096 | 0,150 | 0,512 |
| `Tomato_healthy` | 0,452 | 0,150 | `0,44 / 0,46 / 0,43` | 0,129 | 0,144 | 0,477 |

**Interpretasi Epistemik:** Perbedaan nilai `BrdLum` (`0,477` hingga `0,604`) dan `BrdStd` (`0,093` hingga `0,174`) antar-kelas **mengindikasikan asosiasi statistik, bukan hubungan sebab-akibat**. `[Hipotesis]` Model CNN berpotensi memanfaatkan kecerahan atau tekstur latar tepi sebagai fitur pintas (*shortcut learning*), yang wajib diuji melalui eksperimen augmentasi dan inspeksi **Grad-CAM** sebelum model dipilih untuk aplikasi/UI.

---

## 6. Keterkaitan Temuan EDA dengan Rancangan Hipotesis Mobile-Edge (H1–H4)

Seluruh hipotesis operasional proyek mengacu pada protokol eksperimen terkini di [EXPERIMENT_PROTOCOL.md](../EXPERIMENT_PROTOCOL.md) dengan seleksi di `val` (`--set eval.eval_split=val`) dan evaluasi 1× di `test`:

| ID | Temuan Pemicu dari EDA | `[Hipotesis]` | Desain Pengujian Terkontrol | Bukti & Kriteria |
| :--- | :--- | :--- | :--- | :--- |
| **H1** | 10 kelas merupakan penyakit Tomat dengan bercak visual halus & latar homogen (`BrdStd < 0,10`); resolusi ditingkatkan ke `224×224` | *2-stage fine-tuning* (membuka blok residual atas setelah *head warmup* 10 ep, `AdamW`, `lr=1e-4`) menaikkan `val f1_macro` dan menekan kekeliruan antar-kelas Tomat dibanding *frozen head* | `m1↔m2` (`MobileNetV2`), `m3a↔m3b` (`EfficientNet-B0`), `m4a↔m4b` (`MobileNetV3-Large`) | `val f1_macro`, kurva GD Stage 1→2, `val_confused_pairs.csv` |
| **H2** | Kebutuhan efisiensi *deployment* web cloud & perangkat *mobile-edge* tanpa mengorbankan akurasi lintas 15 kelas | Pada konfigurasi kontrol identik (`224×224`, 25 ep Stage 2, `ls=0.05`, `class_weight=balanced`), perbandingan `MobileNetV2`, `EfficientNet-B0`, dan `MobileNetV3-Large` mengungkap *trade-off* antara `f1_macro`, ukuran file (MB), dan latency CPU | Bandingkan `m2_mobilenetv2_ft`, `m3b_effnetb0_ft`, dan `m4b_mobilenetv3l_ft` | `f1_macro`, *Bootstrap 95% CI*, *Exact McNemar p-value*, ukuran MB, p50 ms |
| **H3** | Target optimasi mobile offline (`FP16`/`INT8`) sekaligus keandalan server hemat RAM | Kuantisasi TFLite PTQ `FP16` mempertahankan $\Delta\text{F1} \le 0{,}005$ dengan ukuran ~50% lebih kecil, sedangkan `INT8` (kalibrasi *stratified* 300 citra `train`) menekan ukuran ~70–75% dengan potensi sensitivitas berbeda per arsitektur | Evaluasi `scripts/export_tflite.py` (`keras_fp32`, `tflite_fp32`, `tflite_fp16`, `tflite_int8`) | `ΔF1 vs FP32`, *Exact McNemar*, ukuran `.tflite` (MB), CPU latency (ms) |
| **H4** | Variasi `BrdLum` tepi (`0,477`–`0,604`) berisiko *shortcut learning* + potensi *miscalibration* probabilitas | Augmentasi fotometris + geometris memusatkan fokus **Grad-CAM** pada gejala daun, dan *5-fold OOF Temperature Scaling* pada logit `val` menekan **ECE** tanpa mengubah akurasi | Audit `gradcam_grid` + *Temperature Scaling* pada logit `val` | Fokus lesi di Grad-CAM & `ECE` *before → after OOF* |

---

## 7. Rekomendasi Teknis Sebelum Training di Kaggle

1. **Perlindungan & Anti-Leakage di `src/data.py` (Sudah Diimplementasikan):** `scan_dataset()` kini mengabaikan subfolder tanpa citra langsung (aman dari *double-nesting* Kaggle), `data.deduplicate=true` membuang 14 duplikat MD5 persis secara *in-memory*, dan `data.group_same_leaf=true` menjaga grup foto daun fisik yang sama (`Leaf X.Y`) tetap dalam satu split.
2. **Evaluasi Realistis untuk Aplikasi/UI:** Karena PlantVillage berlatar seragam, keberhasilan pada `val`/`test` PlantVillage hanya mengukur generalisasi dalam distribusi studio (`[Asumsi]`). Uji tambahan dengan beberapa foto daun latar alami tetap dibutuhkan sebelum demo aplikasi akhir.
3. **Reproduksibilitas EDA:** Gunakan `python scripts/build_eda_notebook.py` untuk memperbarui [notebook/data_audit_eda.ipynb](../notebook/data_audit_eda.ipynb) bila diperlukan (jangan mengedit JSON notebook secara manual).
