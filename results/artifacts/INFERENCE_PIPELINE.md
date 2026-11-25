# Spesifikasi Inference Pipeline — PlantVile

Dokumen ini menjelaskan spesifikasi lengkap **7 langkah *End-to-End Inference Pipeline*** dari gambar daun mentah hingga hasil prediksi terkalibrasi (serta visualisasi opsional **Grad-CAM**). Panduan ini bersifat **netral terhadap *framework*** agar tim pengembang aplikasi dapat mengimplementasikan backend dan UI langsung di repositorinya sendiri menggunakan bahasa atau *framework* web pilihannya.

---

## 1. Diagram Alur *End-to-End Inference Pipeline*

```text
[1. Input Gambar Daun]  (JPG/PNG/BMP, fokus 1 helai daun)
         │
         ▼
[2. Decode RGB & Resize]  (3 channel RGB, bilinear resize ke 224×224, float32)
         │
         ▼
[3. Normalisasi Spesifik Arsitektur]
   ├─ m4b (MobileNetV3-Large): biarkan rentang [0.0, 255.0] float32 (Rescaling internal)
   └─ m2  (MobileNetV2):       (pixel / 127.5) - 1.0  ->  rentang [-1.0, 1.0] float32
         │
         ▼
[4. Forward Pass Model]  (input [1, 224, 224, 3] float32 -> softmax output [1, 15] float32)
         │
         ▼
[5. Kalibrasi Temperature Scaling]  (p_cal = softmax(log(clip(p, 1e-12, 1.0)) / T))
   ├─ m4b_mobilenetv3l_ft: T = 0.64288
   └─ m2_mobilenetv2_ft:   T = 0.60732
         │
         ▼
[6. Keputusan, Top-K & Flag]  (argmax + top_k=3 + map ke labels.txt + warning jika conf < 0.70)
         │
         ▼
[7. (Opsional, khusus keras_fp32) Grad-CAM Overlay]  (heatmap jet alpha=0.4 pada gambar 224×224)
```

---

## 2. Rincian Teknis Setiap Langkah Pipeline

1. **Input & Validasi Gambar:**
   - Terima gambar (`.jpg`, `.jpeg`, `.png`, `.bmp`) dari *memory buffer* (upload HTTP) atau path lokal, dan validasi bahwa file dapat dibuka sebagai gambar yang sah sebelum diproses model.
2. **Decode & Resize:**
   - Konversi gambar ke **3 channel RGB** (bukan BGR atau RGBA).
   - *Resize* ke **`224×224`** menggunakan interpolasi **`bilinear`** bertipe **`float32`**.
   - Simpan salinan gambar ter-resize `224×224` versi `uint8 [0..255]` bila ingin menampilkan atau memadukan overlay Grad-CAM.
3. **Normalisasi Input (Wajib Cocok dengan Model Terpilih):**
   - **`m4b_mobilenetv3l_ft` (`MobileNetV3-Large`, rekomendasi Web & FP16):** Biarkan nilai piksel pada **`[0.0, 255.0]`** (`float32`), karena layer `Rescaling` sudah tertanam di dalam graf model (`keras.applications.mobilenet_v3.preprocess_input` tidak mengubah skala `[0, 255]`).
   - **`m2_mobilenetv2_ft` (`MobileNetV2`):** Ubah skala piksel dari `[0, 255]` menjadi **`[-1.0, 1.0]`** dengan rumus `(pixel_255 / 127.5) - 1.0`.
   - Tambahkan dimensi *batch* di depan sehingga shape tensor menjadi **`[1, 224, 224, 3]`**.
4. **Forward Pass Model:**
   - Muat file model **sekali saja saat aplikasi/server menyala** (*startup / singleton*), bukan berulang per *request*.
   - Pilihan file model di `models/`:
     - **`models/m4b_mobilenetv3l_ft/m4b_mobilenetv3l_ft_infer_fp32.keras`** (`13,04 MB`, `test F1 = 0,9940`): rekomendasi utama untuk Web/Localhost karena mendukung Grad-CAM dan inferensi cepat (`tf.function`).
     - **`models/m4b_mobilenetv3l_ft/m4b_mobilenetv3l_ft_fp16.tflite`** (`6,24 MB`, `test F1 = 0,9943`): opsi ringan & hemat RAM (tanpa Grad-CAM).
     - **`models/m2_mobilenetv2_ft/m2_mobilenetv2_ft_int8.tflite`** (`2,93 MB`, `test F1 = 0,9864`): opsi ultra-ringan INT8 (tanpa Grad-CAM).
   - **Peringatan:** Jangan gunakan `m4b_mobilenetv3l_ft_int8.tflite` maupun `m3b_effnetb0_ft_int8.tflite` karena akurasinya rusak pasca-kuantisasi INT8.
   - *(Catatan teknis TFLite: jika memuat file `.tflite` pada lingkungan CPU tertentu gagal dengan error delegasi `TfLiteXNNPackDelegate`, inisialisasi ulang interpreter tanpa delegasi default / `BUILTIN_WITHOUT_DEFAULT_DELEGATES`).*
   - Keluaran model adalah vektor probabilitas softmax **`[1, 15]`** (`float32`).
5. **Post-Processing Kalibrasi (*Temperature Scaling*):**
   - Karena model dilatih dengan *label smoothing* `0,05`, probabilitas mentah sedikit konservatif (*underconfident*). Kalibrasi ulang memakai nilai `temperature_T` dari `model_metadata.json` (`0.64288` untuk `m4b`, `0.60732` untuk `m2`):
     ```text
     log_p         = log(clip(probs, 1e-12, 1.0))
     scaled_logits = log_p / T
     probs_cal     = softmax(scaled_logits - max(scaled_logits))
     ```
6. **Keputusan, Top-K & Flag *Low-Confidence*:**
   - Kelas prediksi = indeks `argmax(probs_cal)` (`0..14`), dipetakan ke `labels.txt` / tabel di Bagian 4.
   - Ekstrak 3 prediksi teratas (`top_k = 3`) beserta nilai probabilitas terkalibrasinya.
   - Jika `confidence < 0.70` (nilai `low_confidence_threshold` di `model_metadata.json`), tandai sebagai **`low_confidence_warning = true`** dan tampilkan saran di UI agar pengguna mengunggah ulang foto satu helai daun yang fokus dengan latar polos.
7. **Visualisasi Grad-CAM (Opsional — Khusus `keras_fp32`):**
   - Ambil aktivasi peta fitur 4D terakhir dari sub-model backbone (pada `m4b`, layer pertama bertipe Functional/Model di dalam `m4b_mobilenetv3l_ft_infer_fp32.keras`), lalu hitung gradien *pre-softmax logit* kelas prediksi terhadap peta fitur tersebut.
   - Hitung bobot tiap channel dengan *Global Average Pooling* pada gradien, kalikan dengan peta fitur dan jumlahkan sepanjang sumbu channel, lalu terapkan `ReLU` dan normalisasi ke `[0, 1]`.
   - *Resize bilinear* heatmap ke `224×224`, terapkan *colormap* `jet`, dan padukan dengan gambar RGB `224×224` menggunakan rumus `(1 - 0.4) * orig_rgb + 0.4 * heatmap_rgb` (klip `0..255`, `uint8`).

---

## 3. Catatan Dependensi Backend (Jika Menggunakan Python)

Untuk mengimplementasikan 7 langkah di atas di repositori backend Anda sendiri:
- **Minimal inferensi `.keras` atau `.tflite`:** `tensorflow`, `numpy`, `pillow`.
- **Tambahan jika merender *colormap* `jet` Grad-CAM:** `matplotlib` (atau gunakan OpenCV `applyColorMap(..., COLORMAP_JET)` tanpa `matplotlib`).

---

## 4. Tabel Pemetaan 15 Kelas (`labels.txt` → Label Tampilan UI)

Urutan indeks `0..14` berikut wajib sama persis dengan `labels.txt` dan `class_names.json`:

| Indeks | `pred_class` (Kunci Mentah) | Spesies | Kondisi | Nama Tampilan (Indonesia) | Nama Tampilan (English) |
| :---: | :--- | :--- | :--- | :--- | :--- |
| `0` | `Pepper__bell___Bacterial_spot` | Paprika | Sakit | Paprika — Bercak Bakteri | Bell Pepper — Bacterial Spot |
| `1` | `Pepper__bell___healthy` | Paprika | Sehat | Paprika — Sehat | Bell Pepper — Healthy |
| `2` | `Potato___Early_blight` | Kentang | Sakit | Kentang — Bercak Kering (*Early Blight*) | Potato — Early Blight |
| `3` | `Potato___Late_blight` | Kentang | Sakit | Kentang — Busuk Daun (*Late Blight*) | Potato — Late Blight |
| `4` | `Potato___healthy` | Kentang | Sehat | Kentang — Sehat | Potato — Healthy |
| `5` | `Tomato_Bacterial_spot` | Tomat | Sakit | Tomat — Bercak Bakteri | Tomato — Bacterial Spot |
| `6` | `Tomato_Early_blight` | Tomat | Sakit | Tomat — Bercak Kering (*Early Blight*) | Tomato — Early Blight |
| `7` | `Tomato_Late_blight` | Tomat | Sakit | Tomat — Busuk Daun (*Late Blight*) | Tomato — Late Blight |
| `8` | `Tomato_Leaf_Mold` | Tomat | Sakit | Tomat — Jamur Daun (*Leaf Mold*) | Tomato — Leaf Mold |
| `9` | `Tomato_Septoria_leaf_spot` | Tomat | Sakit | Tomat — Bercak Septoria | Tomato — Septoria Leaf Spot |
| `10` | `Tomato_Spider_mites_Two_spotted_spider_mite` | Tomat | Hama/Sakit | Tomat — Tungau Laba-laba (*Spider Mites*) | Tomato — Two-Spotted Spider Mite |
| `11` | `Tomato__Target_Spot` | Tomat | Sakit | Tomat — Bercak Target (*Target Spot*) | Tomato — Target Spot |
| `12` | `Tomato__Tomato_YellowLeaf__Curl_Virus` | Tomat | Sakit | Tomat — Virus Keriting Daun Kuning | Tomato — Yellow Leaf Curl Virus |
| `13` | `Tomato__Tomato_mosaic_virus` | Tomat | Sakit | Tomat — Virus Mosaik Tomat | Tomato — Tomato Mosaic Virus |
| `14` | `Tomato_healthy` | Tomat | Sehat | Tomat — Sehat | Tomato — Healthy |
