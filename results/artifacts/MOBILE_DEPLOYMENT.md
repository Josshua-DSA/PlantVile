# Panduan Deployment Mobile / On-Device & Benchmark Android

> [!IMPORTANT]
> **`[OPSIONAL / TIDAK WAJIB]`**  
> Target pengembangan aplikasi saat ini adalah **Web / Localhost** (lihat [README.md](README.md) dan [INFERENCE_PIPELINE.md](INFERENCE_PIPELINE.md)). Dokumen ini **tidak wajib dikerjakan** untuk rilis Web saat ini, melainkan disediakan sebagai referensi opsional bila suatu saat aplikasi diekspansi ke **Android / iOS / Flutter (*offline on-device*)** atau untuk verifikasi latency fisik via ADB.

---

## 1. Model Mobile yang Direkomendasikan

| Prioritas / Kasus | File `.tflite` | Ukuran | `test F1-Macro` | CPU `p50` (4 threads) |
| :--- | :--- | :---: | :---: | :---: |
| **Utama (Akurasi Maks & Cepat di CPU)** | `models/m4b_mobilenetv3l_ft/m4b_mobilenetv3l_ft_fp16.tflite` | `6,24 MB` | **`0,9943`** | **`4,13 ms`** |
| **Alternatif Ultra-Ringan / Wajib INT8** | `models/m2_mobilenetv2_ft/m2_mobilenetv2_ft_int8.tflite` | **`2,93 MB`** | `0,9864` | `5,82 ms` |

> [!CAUTION]
> - **JANGAN PERNAH** men-deploy `models/m4b_mobilenetv3l_ft/m4b_mobilenetv3l_ft_int8.tflite` (`F1 = 0,4428`, dan gagal pada delegasi XNNPACK CPU) maupun `models/m3b_effnetb0_ft/m3b_effnetb0_ft_int8.tflite` (`F1 = 0,9467`).
> - Untuk ekspansi mobile tanpa QAT (*Quantization-Aware Training*) tambahan, pilih **`m4b` `FP16`** jika ukuran `6,24 MB` masih sesuai anggaran aplikasi, atau **`m2` `INT8`** jika butuh ukuran terkecil (`2,93 MB`).

---

## 2. Kontrak Tensor & Preprocessing per Arsitektur

- **Input Tensor:** Shape `[1, 224, 224, 3]`, tipe **`float32`** (bukan `uint8`!), urutan channel **`RGB`**, metode resize **`bilinear`**.
- **Output Tensor:** Shape `[1, 15]`, tipe **`float32`** (vektor probabilitas softmax terurut sesuai `labels.txt` indeks `0..14`).

| Model | Rumus Normalisasi Input dari Piksel RGB `[0..255]` | Alasan Teknis |
| :--- | :--- | :--- |
| **`m4b_mobilenetv3l_ft`** (`MobileNetV3-Large`) | Biarkan pada **`[0.0, 255.0]`** (`float32`) | Layer `Rescaling` sudah tertanam di dalam graf model `.tflite` |
| **`m3b_effnetb0_ft`** (`EfficientNet-B0`) | Biarkan pada **`[0.0, 255.0]`** (`float32`) | Layer `Rescaling` & `Normalization` sudah tertanam di dalam graf |
| **`m2_mobilenetv2_ft`** (`MobileNetV2`) | **`(pixel / 127.5) - 1.0`** → rentang **`[-1.0, 1.0]`** (`float32`) | Tidak memiliki layer normalisasi internal |

---

## 3. Post-Processing di Sisi Klien (Kalibrasi Probabilitas)

Agar nilai persentase *confidence* yang tampil di aplikasi mobile selaras dengan standar kalibrasi evaluasi:

1. Ambil skor `p[i]` untuk `i = 0..14` dan nilai `temperature_T` dari `model_metadata.json`:
   - `m4b_mobilenetv3l_ft`: `T = 0.64288`
   - `m2_mobilenetv2_ft`: `T = 0.60732`
2. Hitung ulang probabilitas terkalibrasi:
   $$z_i = \frac{\ln(\max(p_i, 10^{-12}))}{T}, \qquad p^{\text{cal}}_i = \frac{\exp(z_i - \max_j z_j)}{\sum_j \exp(z_j - \max_j z_j)}$$
3. Tandai sebagai peringatan *low-confidence* jika $\max_i p^{\text{cal}}_i < 0{,}70$.

---

## 4. Pengukuran Latency Asli di Smartphone Android (ADB `benchmark_model`)

Untuk mengukur latency nyata pada chipset HP Android (menggunakan biner resmi `android_aarch64_benchmark_model` dari TensorFlow Lite Benchmark Tools):

1. Aktifkan **USB Debugging** di HP Android, sambungkan kabel USB, dan cek koneksi:
   ```bash
   adb devices
   ```
2. Push biner `android_aarch64_benchmark_model` ke perangkat:
   ```bash
   adb push android_aarch64_benchmark_model /data/local/tmp/
   adb shell chmod +x /data/local/tmp/android_aarch64_benchmark_model
   ```
3. Jalankan benchmark pada dua model mobile rekomendasi (4 CPU threads, 10 warmup, 50 run):
   ```bash
   # 1) Rekomendasi Akurasi Maks: MobileNetV3-Large FP16 (6.24 MB)
   adb push models/m4b_mobilenetv3l_ft/m4b_mobilenetv3l_ft_fp16.tflite /data/local/tmp/
   adb shell /data/local/tmp/android_aarch64_benchmark_model \
     --graph=/data/local/tmp/m4b_mobilenetv3l_ft_fp16.tflite \
     --num_threads=4 --warmup_runs=10 --num_runs=50

   # 2) Rekomendasi Ultra-Ringan: MobileNetV2 INT8 (2.93 MB)
   adb push models/m2_mobilenetv2_ft/m2_mobilenetv2_ft_int8.tflite /data/local/tmp/
   adb shell /data/local/tmp/android_aarch64_benchmark_model \
     --graph=/data/local/tmp/m2_mobilenetv2_ft_int8.tflite \
     --num_threads=4 --warmup_runs=10 --num_runs=50
   ```
4. Catat nilai `Inference timings in us: avg=...` (dibagi `1.000` untuk mendapatkan **ms**). Untuk menguji akselerasi GPU/NPU Android, tambahkan flag `--use_gpu=true` atau `--use_nnapi=true`.
