# Indeks Laporan Eksperimen (`reports/`)

Folder ini berisi 168 file hasil audit dataset, metrik training/evaluasi, kurva *gradient descent*, analisis kesalahan, dan benchmark kuantisasi TFLite. Gunakan indeks ini untuk mengambil angka dan gambar yang relevan bagi laporan atau presentasi tanpa perlu membuka semua folder satu per satu.

---

## 1. Tabel Ringkasan Utama (Siap Masuk Laporan / Slide)

| File | Isi Utama | Kegunaan |
| :--- | :--- | :--- |
| [experiment_leaderboard.csv](experiment_leaderboard.csv) | Ranking seluruh 6 run (`m1`–`m4b`) pada *split* `val` | Menunjukkan peningkatan Stage 1 (*frozen head*) → Stage 2 (*fine-tuning*) |
| [test_leaderboard.csv](test_leaderboard.csv) | Ranking 3 kandidat Stage 2 (`m4b`, `m3b`, `m2`) pada *split* `test` ($n=3.095$) | Tabel hasil akurasi & F1-Macro final |
| [mobile_export/mobile_benchmark_summary.csv](mobile_export/mobile_benchmark_summary.csv) | 12 baris benchmark (`3 model × 4 format`: `keras_fp32`, `tflite_fp32`, `fp16`, `int8`) | Tabel ukuran MB, Bootstrap 95% CI, $\Delta\text{F1}$, ECE, & latency CPU (`p50`/`p95` ms) |
| [mobile_export/mcnemar_pairwise.csv](mobile_export/mcnemar_pairwise.csv) | Uji signifikansi berpasangan *Exact McNemar* ($p$-value) | Bukti statistik perbandingan antar-arsitektur & efek kuantisasi |
| [dataset_audit.json](dataset_audit.json) | Audit integritas PlantVillage (`20.638` valid, rasio ketimpangan `21,1×`, `0` korup) | Bagian Metodologi / Dataset |

---

## 2. Grafik Kurva Gradient Descent (`gradient_descent_plots/`)

| File | Deskripsi |
| :--- | :--- |
| [gd_comparison_stage2.png](gradient_descent_plots/gd_comparison_stage2.png) | Perbandingan kurva `train_loss`, `val_loss`, dan `val_accuracy` Stage 2 untuk ke-3 arsitektur + garis lantai entropi $H(q)=0{,}3117$ |
| [gd_combined_mobilenetv3_large.png](gradient_descent_plots/gd_combined_mobilenetv3_large.png) | Kurva kontinu Stage 1 (`m4a`) → Stage 2 (`m4b`) untuk model utama `MobileNetV3-Large` |
| [gd_combined_efficientnetb0.png](gradient_descent_plots/gd_combined_efficientnetb0.png) | Kurva kontinu Stage 1 (`m3a`) → Stage 2 (`m3b`) untuk `EfficientNet-B0` |
| [gd_combined_mobilenetv2.png](gradient_descent_plots/gd_combined_mobilenetv2.png) | Kurva kontinu Stage 1 (`m1`) → Stage 2 (`m2`) untuk `MobileNetV2` |
| [gd_step_m4b_mobilenetv3l_ft.png](gradient_descent_plots/gd_step_m4b_mobilenetv3l_ft.png) | Kurva *mini-batch gradient descent* **per step** (5.650 step, raw loss + garis halus EMA $\alpha=0{,}05$ + `val_loss`) untuk `m4b` |
| [gd_summary.json](gradient_descent_plots/gd_summary.json) | Statistik konvergensi (`best_val_loss`, `last5_val_loss_slope`, `last5_val_loss_diff_std`, `final_ema_loss`) |

---

## 3. Artefak Evaluasi & Error Analysis Model Terpilih (`m4b_mobilenetv3l_ft/`)

Direkomendasikan mengambil visualisasi dari model kandidat utama **`m4b_mobilenetv3l_ft`** (tersedia juga folder sejenis untuk `m3b_effnetb0_ft` dan `m2_mobilenetv2_ft`):

| File (`test` split) | Kegunaan |
| :--- | :--- |
| [m4b_mobilenetv3l_ft/test_confusion_matrix.png](m4b_mobilenetv3l_ft/test_confusion_matrix.png) | Matriks konfusi 15×15 pada split `test` |
| [m4b_mobilenetv3l_ft/test_per_class.csv](m4b_mobilenetv3l_ft/test_per_class.csv) | Precision, Recall, F1, & Support per kelas (15 kelas) |
| [m4b_mobilenetv3l_ft/test_confused_pairs.csv](m4b_mobilenetv3l_ft/test_confused_pairs.csv) | Pasangan kelas yang paling sering tertukar (`Early_blight -> Septoria_leaf_spot`, dll.) |
| [m4b_mobilenetv3l_ft/test_misclassified.csv](m4b_mobilenetv3l_ft/test_misclassified.csv) | Daftar 20 citra salah prediksi beserta skor *confidence* |
| [m4b_mobilenetv3l_ft/test_confidence_calibration.png](m4b_mobilenetv3l_ft/test_confidence_calibration.png) | *Reliability diagram* & histogram distribusi *confidence* |
| [m4b_mobilenetv3l_ft/test_gradcam_grid.png](m4b_mobilenetv3l_ft/test_gradcam_grid.png) | Visualisasi fokus lesi daun (Grad-CAM) pada sampel salah prediksi *test* |
| [m4b_mobilenetv3l_ft/test_sample_predictions.png](m4b_mobilenetv3l_ft/test_sample_predictions.png) | Grid contoh prediksi acak pada split `test` |
