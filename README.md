# UAS Neuro Computing — Plant Disease Classification (PlantVillage + CNN)

Klasifikasi penyakit daun dari citra RGB menggunakan CNN.
Seluruh parameter dataset, model, training, dan evaluasi bisa diatur dari terminal.

**Dataset terpasang:** PlantVillage **15 kelas / 20.638 gambar**, resolusi seragam
256x256 RGB, disimpan flat di `UAS/data/<Nama_Kelas>/`.

- **Dataset:** [PlantVillage (emmarex/plantdisease)](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Task:** image classification, supervised learning
- **Framework:** TensorFlow 2.21 / Keras 3.15
- **Python:** `/home/josjiez/my_env/ai_env/bin/python` (sudah berisi TF, sklearn, seaborn)

---

## 1. Struktur dataset (sudah terpasang)

```
UAS/data/
├── Pepper__bell___Bacterial_spot/                 997
├── Pepper__bell___healthy/                       1478
├── Potato___Early_blight/                        1000
├── Potato___healthy/                              152
├── Potato___Late_blight/                         1000
├── Tomato_Bacterial_spot/                        2127
├── Tomato_Early_blight/                          1000
├── Tomato_Late_blight/                           1909
├── Tomato_Leaf_Mold/                              952
├── Tomato_Septoria_leaf_spot/                    1771
├── Tomato_Spider_mites_Two_spotted_spider_mite/  1676
├── Tomato__Target_Spot/                          1404
├── Tomato__Tomato_YellowLeaf__Curl_Virus/        3208
├── Tomato__Tomato_mosaic_virus/                   373
└── Tomato_healthy/                               1591
```

Total **20.638** gambar. Satu file 0-byte `svn-r6Yb5c` (sisa metadata SVN)
otomatis diabaikan karena ekstensinya bukan citra.

Kode menerima dua bentuk struktur:
- **flat** seperti di atas -> `data.root: data`
- **bertingkat** `color/ grayscale/ segmented/` -> `data.root: data/PlantVillage`

Deteksi otomatis; `variant` diabaikan kalau strukturnya flat.

**Catatan imbalance.** Rasio kelas terbesar/terkecil = **21x**
(`Tomato__Tomato_YellowLeaf__Curl_Virus` 3208 vs `Potato___healthy` 152).
`train.class_weight=balanced` sudah aktif secara default untuk mengompensasi ini.

---

## 2. Jalankan

```bash
cd "SDT A Semester 5 2026/Neuro Computing/UAS"
PY=/home/josjiez/my_env/ai_env/bin/python

# a) Menu interaktif — semua ~86 parameter ditanyakan, Enter = default
$PY main.py

# b) Langsung dari YAML
$PY main.py --config configs/default.yaml

# c) YAML + override sebagian
$PY main.py --config configs/default.yaml \
    --set model.arch=vgg16 --set train.epochs=10 --set data.batch_size=64

# d) Validasi saja: config + scan dataset + build model. TIDAK training.
$PY main.py --check --config configs/default.yaml

# e) Evaluasi model tersimpan (tanpa training)
$PY main.py --eval runs/<run_name>/best.keras

# f) Daftar semua parameter
$PY main.py --list-params
```

Saat mode training, `main.py` menampilkan seluruh konfigurasi lalu bertanya
`Mulai training? [y/N]`. Jawab `N` untuk batal.

## 3. Output

Setiap run menulis ke `runs/<run_name>/` (default `run_<timestamp>`):

| File | Isi |
|---|---|
| `best.keras` | bobot terbaik menurut `train.checkpoint_monitor` |
| `final.keras` | bobot akhir training |
| `history.csv`, `history.json` | loss/accuracy per epoch |
| `config_used.yaml`, `config_used.json` | snapshot konfigurasi |
| `class_names.json` | urutan label |
| `device.json` | device yang benar-benar dipakai |
| `test_metrics.json` | accuracy, precision/recall/f1 macro & weighted |
| `test_classification_report.txt` | report per kelas |
| `test_per_class.json`, `test_per_class.csv` | metrik per kelas |
| `test_confusion_matrix.png` | confusion matrix |
| `test_loss_accuracy.png` | kurva training |
| `test_sample_predictions.png` | grid gambar: label asli vs prediksi |

## 4. Parameter

`$PY main.py --list-params` mencetak semuanya. Ringkasan per seksi:

**`data.*`** — `root`, `variant` (color/grayscale/segmented), `img_height`, `img_width`,
`channels`, `class_subset`, `max_per_class`, `val_split`, `test_split`, `split_strategy`,
`shuffle_seed`, `batch_size`, `cache`, `prefetch`, `augment` + 10 sub-parameter augmentasi
(`aug_rotation`, `aug_width_shift`, `aug_height_shift`, `aug_zoom`, `aug_shear`, `aug_hflip`,
`aug_vflip`, `aug_brightness`, `aug_contrast`, `aug_fill_mode`), `rescale`, `normalize`.

**`model.*`** — `arch` (simple_cnn/vgg16/mobilenetv2/resnet50/efficientnetb0), `include_top`,
`weights`, `freeze_base`, `fine_tune_at`, `dropout`, `dense_units`, `batch_norm`,
`l2_weight_decay`, `conv_blocks`, `filters_base`, `kernel_size`, `pool_size`.

**`train.*`** — `optimizer` (adam/adamw/sgd/rmsprop/nadam), `learning_rate`, `momentum`,
`beta_1`, `beta_2`, `epsilon`, `loss` (cce/sparse_cce/focal), `label_smoothing`, `epochs`,
`class_weight` (none/balanced/manual) + `class_weight_manual`, early stopping
(`early_stopping`, `es_monitor`, `es_patience`, `es_min_delta`, `es_mode`, `es_restore_best`),
reduce-lr (`reduce_lr`, `lr_monitor`, `lr_factor`, `lr_patience`, `lr_min_lr`, `lr_cooldown`),
checkpoint (`checkpoint_monitor`, `save_best_only`), `mixed_precision`,
`steps_per_epoch`, `validation_steps`.

**`runtime.*`** — `device` (cpu/gpu/auto), `run_name`, `runs_dir`, `seed`, `verbose`,
`log_csv`, `tensorboard`, `resume_from`, `num_parallel_calls`, `deterministic`.

**`eval.*`** — `eval_batch_size`, `top_k`, `confusion_matrix`, `classification_report`,
`per_class_metrics`, `sample_predictions`, `save_plots`, `eval_split` (val/test/train).

### Contoh resep

```bash
# Cepat di CPU (simple CNN, citra kecil, subset 1000/kelas)
$PY main.py --config configs/default.yaml \
  --set data.root=data/PlantVillage --set model.arch=simple_cnn \
  --set data.img_height=96 --set data.img_width=96 \
  --set data.max_per_class=1000 --set data.cache=memory --set train.epochs=15

# Akurasi maksimal (MobileNetV2 + fine-tune 30 layer terakhir, butuh GPU)
$PY main.py --config configs/default.yaml \
  --set runtime.device=gpu --set data.img_height=224 --set data.img_width=224 \
  --set model.fine_tune_at=100 --set model.learning_rate=0.0001 \
  --set train.optimizer=adamw --set train.epochs=40 --set data.batch_size=64

# Uji coba dataset palsu dulu (tanpa Kaggle)
$PY scripts/make_dummy_dataset.py --out data/dummy --classes 3 --per-class 12
$PY main.py --check --config configs/default.yaml --set data.root=data/dummy
```

## 5. Struktur project

```
UAS/
├── main.py                     entrypoint: wizard / YAML / --check / --eval
├── configs/default.yaml        semua default, berkomentar
├── src/
│   ├── config.py               SPEC (sumber tunggal semua parameter) + validasi + wizard
│   ├── data.py                 scan folder, split stratified, tf.data, augmentasi
│   ├── model.py                simple_cnn + 4 backbone transfer learning, compile
│   ├── train.py                callbacks + fit + artefak run
│   ├── evaluate.py             metrik, confusion matrix, kurva, grid prediksi
│   └── utils.py                seed, device, run dir, IO
├── scripts/make_dummy_dataset.py
├── scripts/build_notebook.py   generator notebook/plantvillage_cnn.ipynb
├── notebook/plantvillage_cnn.ipynb   antarmuka eksperimen (semua parameter)
├── data/                       (dataset, tidak masuk git)
└── runs/                       (output training, tidak masuk git)
```

## 5b. Notebook eksperimen

`notebook/plantvillage_cnn.ipynb` adalah antarmuka interaktif untuk project ini.
Logika training tetap di `src/` — notebook hanya mengatur parameter dan menampilkan hasil,
jadi hasilnya identik dengan `python main.py`.

```bash
# daftar kernel sekali saja (sudah terdaftar sebagai ncomp-ai)
$PY -m ipykernel install --user --name ncomp-ai --display-name "Python (NeuroComputing ai_env)"

jupyter lab notebook/plantvillage_cnn.ipynb     # pilih kernel "Python (NeuroComputing ai_env)"
```

Isi notebook: setup & device → konfigurasi (86 parameter di satu sel) → EDA →
verifikasi split → inspeksi pipeline/augmentasi → arsitektur → training →
evaluasi → kurva → inferensi → ekspor → referensi parameter.

Sel training dipagari `RUN_TRAINING = False`. `Run All` **tidak** akan memulai
training sampai flag itu diubah manual.

Notebook di-generate, jangan diedit langsung — ubah `scripts/build_notebook.py`
lalu jalankan `$PY scripts/build_notebook.py`.

Menambah parameter = satu baris di `SPEC` (`src/config.py`); default, YAML, `--set`,
wizard, validasi, dan `--list-params` ikut menyesuaikan.

## 6. Catatan GPU

RTX 3050 6 GB terdeteksi di sistem, tetapi build TensorFlow 2.21 saat ini tidak menemukan
library CUDA. `runtime.device`:

- `auto` — pakai GPU kalau TF melihatnya, kalau tidak CPU (default)
- `cpu` — GPU disembunyikan paksa
- `gpu` — gagal cepat dengan pesan jelas kalau TF tidak melihat GPU

Untuk mengaktifkan GPU: pasang `nvidia-cuda-runtime-cu12`, `nvidia-cudnn-cu12` dan
kawan-kawan yang sesuai ke environment yang sama, lalu cek dengan
`python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`.

## 7. Catatan dataset

- 38 kelas tidak seimbang (mis. Apple_healthy 1645 vs Potato_healthy 152) → default
  `train.class_weight=balanced`.
- Jumlah gambar dibaca dari isi folder saat runtime, tidak di-hardcode; kalau versi
  Kaggle yang kamu unduh berbeda jumlahnya, kode tetap jalan.
- Split dilakukan **per kelas** (`stratified`) sehingga semua kelas muncul di val/test.
