# PlantVile — Paket Deliverable Model, Dokumentasi & Laporan Hasil

Folder ini adalah paket deliverable final dari proyek klasifikasi penyakit daun tanaman **PlantVile** (15 kelas: Tomat, Paprika/Cabai, Kentang), berisi **artefak bobot model terlatih, spesifikasi *inference pipeline***, serta **laporan evaluasi eksperimen**.

> [!IMPORTANT]
> **Batasan Scope Pengembangan Saat Ini:**
> - **Target Utama:** Aplikasi **Web / Localhost** (backend + UI dikembangkan langsung di repositori aplikasi Anda sendiri, tanpa meng-`import` skrip apa pun dari folder `results/`).
> - **Pengembangan Mobile (Opsional / Tidak Wajib):** Aplikasi *on-device* Android/Flutter **tidak wajib** dikerjakan dalam cakupan waktu saat ini. File `.tflite` dan [artifacts/MOBILE_DEPLOYMENT.md](artifacts/MOBILE_DEPLOYMENT.md) disertakan hanya sebagai bukti evaluasi kuantisasi dan referensi opsional bila suatu saat ingin diekspansi ke mobile.

---

## 1. Struktur Isi Paket Deliverable

| Subfolder | Ukuran | Kegunaan |
| :--- | :---: | :--- |
| **[artifacts/](artifacts/README.md)** | ~110,8 MB | **Bobot Model, Metadata & Spesifikasi Pipeline** — File `.keras` FP32 & `.tflite` (FP32/FP16/INT8), `model_metadata.json`, `labels.txt`, `class_names.json`, serta panduan *inference pipeline*. |
| **[reports/](reports/REPORTS_INDEX.md)** | ~104,7 MB | **Dokumentasi Hasil & Bahan Laporan** — Leaderboard `val` & `test`, benchmark kuantisasi, uji statistik Exact McNemar, kurva *gradient descent*, *confusion matrix*, kurva kalibrasi, & visualisasi Grad-CAM. |

> [!NOTE]
> Jika di lingkungan lokal Anda terdapat subfolder lain seperti `runs/` (~570 MB) atau `__results___files/` (~5 MB), subfolder tersebut hanyalah **arsip pelatihan mentah internal** dan **tidak perlu** disertakan saat menyerahkan hasil ke rekan tim.

---

## 2. Ringkasan Pilihan Model untuk Web / Localhost (& Opsional Mobile)

| Skenario Penggunaan | Model & Format yang Dipakai | Ukuran | `test F1-Macro` | Latency CPU (`p50`) |
| :--- | :--- | :---: | :---: | :---: |
| **Web / Localhost + Grad-CAM** *(rekomendasi utama)* | `m4b_mobilenetv3l_ft` (`keras_fp32`) | `13,04 MB` | **`0,9940`** | `23,80 ms` |
| **Web Hemat RAM** *(tanpa Grad-CAM, opsional Mobile FP16)* | `m4b_mobilenetv3l_ft` (`tflite_fp16`) | `6,24 MB` | **`0,9943`** | `4,13 ms` |
| **Opsional Edge Ultra-Ringan (`INT8`)** | `m2_mobilenetv2_ft` (`tflite_int8`) | **`2,93 MB`** | `0,9864` | `5,82 ms` |

> [!CAUTION]
> **JANGAN GUNAKAN** `m4b_mobilenetv3l_ft_int8.tflite` (`F1 = 0,4428`) maupun `m3b_effnetb0_ft_int8.tflite` (`F1 = 0,9467`) karena mengalami degradasi parah saat kuantisasi INT8. Untuk `INT8`, gunakan **hanya `m2_mobilenetv2_ft_int8.tflite`**.

---

## 3. Peta Dokumentasi Lengkap

1. **[artifacts/README.md](artifacts/README.md)** — Ringkasan isi artefak model & tabel pemilihan cepat.
2. **[artifacts/INFERENCE_PIPELINE.md](artifacts/INFERENCE_PIPELINE.md)** — Spesifikasi teknis 7 langkah *end-to-end inference pipeline* (input → resize → normalisasi → forward pass → *Temperature Scaling* → Top-K/warning → Grad-CAM) & pemetaan 15 label kelas untuk UI.
3. **[artifacts/MODEL_CARD.md](artifacts/MODEL_CARD.md)** — Metrik evaluasi `test`, performa per spesies, efek kalibrasi, & teks *disclaimer* UI terkait cakupan domain.
4. **[artifacts/MOBILE_DEPLOYMENT.md](artifacts/MOBILE_DEPLOYMENT.md)** — *(Opsional / Tidak Wajib)* Kontrak tensor mobile, preprocessing per arsitektur, & perintah ADB `benchmark_model`.
5. **[reports/REPORTS_INDEX.md](reports/REPORTS_INDEX.md)** — Peta file statistik, tabel, dan gambar di `reports/` untuk keperluan dokumentasi/laporan.
