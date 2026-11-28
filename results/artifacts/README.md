# PlantVile — Artefak Model & Panduan Pipeline Inferensi (`artifacts/`)

Folder ini murni berisi **bobot model terlatih, metadata kalibrasi, daftar label, dan spesifikasi *inference pipeline*** (tanpa skrip kode yang perlu di-`import`). Pengembang aplikasi bebas menyalin file bobot terpilih ke repositori aplikasinya sendiri dan mengimplementasikan *pipeline* sesuai spesifikasi.

> [!IMPORTANT]
> - **Target Saat Ini:** Pengembangan **Web / Localhost**.
> - **Mobile / On-Device:** **Opsional / Tidak Wajib** — file `.tflite` tetap tersedia sebagai bagian dari evaluasi eksperimen dan kesiapan ekspansi di luar tenggat saat ini.

---

## 1. Struktur Isi Folder

| File / Folder | Fungsi |
| :--- | :--- |
| **[INFERENCE_PIPELINE.md](INFERENCE_PIPELINE.md)** | Spesifikasi teknis 7 langkah *end-to-end inference pipeline*, dependensi runtime yang disarankan, rumus kalibrasi, & pemetaan 15 label kelas UI. |
| **[MODEL_CARD.md](MODEL_CARD.md)** | Rincian metrik terverifikasi (`test`), performa per spesies, & *disclaimer* batasan domain untuk UI. |
| **[MOBILE_DEPLOYMENT.md](MOBILE_DEPLOYMENT.md)** | *(Opsional / Tidak Wajib)* Kontrak tensor mobile & panduan ADB `benchmark_model` untuk ekspansi di masa depan. |
| `model_metadata.json` | Metadata arsitektur, path relatif file bobot, nilai suhu kalibrasi `temperature_T`, ambang *low-confidence*, & ringkasan benchmark `test`. |
| `class_names.json` / `labels.txt` | Daftar 15 nama kelas berurutan indeks `0..14`. |
| `models/` | File model `.keras` FP32 dan `.tflite` (`fp32`, `fp16`, `int8`) untuk `m4b_mobilenetv3l_ft`, `m3b_effnetb0_ft`, dan `m2_mobilenetv2_ft`. |

---

## 2. Tabel Pemilihan Model Cepat (`test`, $n = 3.095$)

| Skenario | `model_key` | File Bobot Utama (`models/...`) | Ukuran | `F1-Macro` | CPU `p50` | Grad-CAM |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Web / Localhost Utama** *(rekomendasi)* | `m4b_mobilenetv3l_ft` | `m4b_mobilenetv3l_ft/m4b_mobilenetv3l_ft_infer_fp32.keras` | `13,04 MB` | `0,9940` | `23,80 ms` | **Ya** |
| **Web Hemat RAM** *(atau opsional Mobile FP16)* | `m4b_mobilenetv3l_ft` | `m4b_mobilenetv3l_ft/m4b_mobilenetv3l_ft_fp16.tflite` | `6,24 MB` | **`0,9943`** | **`4,13 ms`** | Tidak |
| **Opsional Edge Ultra-Ringan (`INT8`)** | `m2_mobilenetv2_ft` | `m2_mobilenetv2_ft/m2_mobilenetv2_ft_int8.tflite` | **`2,93 MB`** | `0,9864` | `5,82 ms` | Tidak |

> [!CAUTION]
> **PERINGATAN KERAS — JANGAN DIPAKAI:**
> - **`models/m4b_mobilenetv3l_ft/m4b_mobilenetv3l_ft_int8.tflite`**: Runtuh saat kuantisasi INT8 (`F1-Macro = 0,4428`, `worst_class_recall = 0,0`, `p50 = 63,80 ms`).
> - **`models/m3b_effnetb0_ft/m3b_effnetb0_ft_int8.tflite`**: Mengalami penurunan signifikan (`F1-Macro = 0,9467`, $\Delta\text{F1} = -0{,}0451$).

---

## 3. Langkah Mulai Integrasi di Backend Anda

1. Salin **hanya file model yang dibutuhkan** (mis. `m4b_mobilenetv3l_ft_infer_fp32.keras`), `labels.txt` (atau `class_names.json`), dan `model_metadata.json` ke repositori backend aplikasi Anda.
2. Terapkan 7 langkah *preprocessing*, *forward pass*, *Temperature Scaling*, pemetaan label, dan (opsional) **Grad-CAM** sesuai spesifikasi di **[INFERENCE_PIPELINE.md](INFERENCE_PIPELINE.md)** langsung di kode backend Anda sendiri.
3. Pasang informasi batasan cakupan & peringatan prediksi dari **[MODEL_CARD.md](MODEL_CARD.md)** di antarmuka web Anda.
