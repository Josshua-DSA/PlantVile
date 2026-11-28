# Model Card — PlantVile Leaf Disease Classifier

Dokumen ini merangkum profil kinerja terverifikasi, perilaku kalibrasi, serta batas cakupan penggunaan model untuk dipasang sebagai informasi/disclaimer di UI aplikasi dan bahan laporan.

---

## 1. Informasi Umum Model

- **Tugas:** Klasifikasi citra daun tanaman tunggal (*single-label multi-class*, 15 kelas: Tomat, Paprika/Cabai, Kentang).
- **Dataset Pelatihan & Uji:** PlantVillage (`color`), `20.624` citra unik (pasca-deduplikasi MD5 & *group-aware split* per daun fisik; rasio `14.434` train / `3.095` val / `3.095` test).
- **Resolusi Input:** `224×224×3` (`RGB`, resize *bilinear*).
- **Kandidat Model Utama (`Deployment Candidate` pada domain PlantVillage):**
  - **Web Cloud + Grad-CAM (`keras_fp32`) & Mobile FP16 (`tflite_fp16`):** `m4b_mobilenetv3l_ft` (*MobileNetV3-Large* fine-tuned).
  - **Edge Ultra-Ringan (`tflite_int8`):** `m2_mobilenetv2_ft` (*MobileNetV2* fine-tuned).

---

## 2. Ringkasan Performa Terverifikasi (Split `test`, $n = 3.095$)

| Model & Format | Ukuran | `Accuracy` [95% CI] | `F1-Macro` [95% CI] | `Worst Recall` | `ECE` Mentah | Latency CPU `p50` / `p95` (`4 threads`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`m4b` (`keras_fp32`)** *(Web)* | `13,04 MB` | `0,9935` `[0,9906, 0,9961]` | **`0,9940`** `[0,9912, 0,9965]` | `0,9600` | `0,0407` | `23,80 / 25,04 ms` |
| **`m4b` (`tflite_fp16`)** *(Mobile/Lite)* | `6,24 MB` | **`0,9939`** `[0,9913, 0,9964]` | **`0,9943`** `[0,9916, 0,9968]` | **`0,9667`** | `0,0406` | `4,13 / 6,53 ms` |
| **`m2` (`tflite_int8`)** *(Edge INT8)* | **`2,93 MB`** | `0,9858` `[0,9813, 0,9900]` | `0,9864` `[0,9820, 0,9904]` | `0,9267` | `0,0437` | `5,82 / 6,36 ms` |
| `m3b` (`tflite_fp16`) *(Pembanding)* | `8,39 MB` | `0,9922` `[0,9890, 0,9952]` | `0,9917` `[0,9868, 0,9954]` | `0,9684` | `0,0452` | `9,55 / 10,03 ms` |

- **Uji Signifikansi (`Exact McNemar`):** `m4b` lebih akurat secara signifikan dibanding `m2` (`p = 0,0009` pada `keras_fp32`), sedangkan retensi kuantisasi `m2` dari `FP32 → INT8` identik secara statistik (`p = 1,0`, $\Delta\text{F1} = -0{,}0003$).
- **Kalibrasi Probabilitas (*Temperature Scaling*):** Pelatihan memakai *label smoothing* `0,05` sehingga probabilitas mentah sedikit *underconfident*. Penerapan `apply_temperature=True` memakai temperatur hasil *5-fold OOF* pada `val`:
  - `m4b_mobilenetv3l_ft`: $T = 0{,}64288$ (menurunkan `val ECE` dari `0,0385` → `0,0020`).
  - `m2_mobilenetv2_ft`: $T = 0{,}60732$ (menurunkan `val ECE` dari `0,0389` → `0,0029`).

---

## 3. Performa per Spesies & Kelas Tersulit (`m4b`, Split `test`)

- **Paprika/Cabai (`Pepper`, $n = 372$, 2 kelas):** `100%` akurasi & recall (`0` error).
- **Kentang (`Potato`, $n = 323$, 3 kelas):** `100%` akurasi & recall (`0` error; kelas minoritas `Potato___healthy` hanya berjumlah $n = 23$ pada `test`).
- **Tomat (`Tomato`, $n = 2.400$, 10 kelas):** `99,17%` akurasi (`20` error; **100% kesalahan `m4b` terjadi pada Tomat**).
- **Pola Kekeliruan Tersering:** Kemiripan lesi bercak antar-penyakit Tomat — khususnya `Tomato_Early_blight` (`Recall = 0,9600`, `4` salah prediksi ke `Tomato_Septoria_leaf_spot`) dan `Tomato_Late_blight` (`Recall = 0,9789`).

---

## 4. Intended Use, Out-of-Scope & Disclaimer untuk UI

- **Cakupan Penggunaan (*Intended Use*):** Alat bantu skrining/edukasi awal klasifikasi penyakit daun **Tomat, Paprika, dan Kentang** berbasis foto **satu helai daun** yang jelas dan terpusat.
- **Batasan Penting (*Out-of-Scope* & Risiko *Domain Shift*):**
  1. **Latar Belakang Studio:** Seluruh citra PlantVillage diambil pada latar abu-abu/studio terkendali. Performa pada foto kebun nyata (*in-situ* dengan latar tanah, bayangan berat, tangan, atau banyak daun sekaligus) **belum diukur** dan berpotensi lebih rendah dari `99%`.
  2. **Terbatas 15 Kelas:** Tanaman di luar daftar (mis. cabai rawit varietas jauh berbeda, padi, jagung) atau penyakit di luar 15 kelas akan tetap dipetakan paksa ke salah satu dari 15 kelas yang ada (*closed-set softmax*).
  3. **Bukan Pengganti Ahli Agronomi:** Gunakan bersama tampilan **Grad-CAM** (untuk memeriksa apakah fokus model berada pada bercak daun, bukan latar belakang) serta peringatan *low-confidence* (`< 70%`).

> **Contoh Teks Disclaimer Siap Pasang di Footer UI:**  
> *"Hasil deteksi merupakan prediksi awal berbasis AI untuk 15 kategori daun Tomat, Paprika, dan Kentang. Untuk hasil terbaik, unggah foto 1 helai daun yang fokus dengan latar polos. Selalu konfirmasikan penanganan di lapangan dengan petugas/ahli pertanian."*
