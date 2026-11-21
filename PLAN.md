# UAS Neuro Computing — Plant Disease CNN Classifier

**Goal:** Project dir di `UAS/` yang bisa training dari terminal dengan **seluruh** parameter
dataset / model / training / evaluasi di-set dari input terminal (interaktif, atau lewat
YAML / flag). Tanpa menjalankan training oleh agen.

**Studi kasus:** klasifikasi 38 kelas penyakit daun (PlantVillage, 14 spesies tanaman),
image classification supervised, CNN.
**Dataset:** `kaggle datasets download -d emmarex/plantdisease`
→ struktur `PlantVillage/{color,grayscale,segmented}/<Tanaman>___<Kondisi>/img.jpg`

---

## 0. Fakta lingkungan (sudah dicek, bukan asumsi)

| Hal | Hasil |
|---|---|
| Python ML env | `/home/josjiez/my_env/ai_env/bin/python` (3.12.3) |
| Framework | TensorFlow **2.21.0** + Keras **3.15.1** ✅ terpasang |
| PyTorch | **tidak ada** → pakai TF/Keras (jangan install torch, sudah ada yang jalan) |
| GPU | RTX 3050 Laptop **6 GB** ada, tapi TF **tidak melihat CUDA driver** → saat ini **CPU-only** |
| Kaggle CLI | tidak terpasang |
| Disk | 230 GB bebas |
| Konvensi praktikum | notebook M11/M12: `tf.keras`, `sns`, seed 42, output `*.png` + `*_summary.json` |

**Implikasi:** 54k citra di CPU = lambat. Butuh keputusan GPU (lihat §6 Q2).

---

## 1. Struktur file (usulan)

```
UAS/
├── PLAN.md                  ← file ini
├── README.md                ← cara pakai, tabel semua parameter, contoh perintah
├── requirements.txt
├── .gitignore               ← data/, runs/, *.keras, *.h5
├── main.py                  ← ENTRYPOINT. menu interaktif + flag mode
├── configs/
│   └── default.yaml         ← semua default, dikomentari
└── src/
    ├── config.py            ← YAML + flag + prompt → satu objek Config tervalidasi
    ├── data.py              ← index folder, split, tf.data pipeline, augmentasi
    ├── model.py             ← CNN scratch + transfer learning, compile
    ├── train.py             ← loop training, callbacks, class weight
    ├── evaluate.py          ← confusion matrix, report, grafik, prediksi sampel
    └── utils.py             ← seed, logging, path run dir, save json
```

6 modul. Tidak lebih: tidak ada `predict.py` terpisah (evaluate sudah bisa load model
dan prediksi), tidak ada `api/`, tidak ada ORM, tidak ada CLI framework (argparse cukup).

---

## 2. Mode pemakaian `main.py`

| Perintah | Fungsi |
|---|---|
| `python main.py` | menu interaktif penuh (semua parameter, default ditampilkan, Enter = pakai default) |
| `python main.py --config configs/default.yaml` | langsung dari YAML, tanpa prompt |
| `python main.py --config X --set model.arch=mobilenetv2 --set train.epochs=5` | override per-key |
| `python main.py --check` | **validasi saja**: baca config → scan dataset → build model → print `summary()` → keluar. Tidak training. |
| `python main.py --eval runs/<run>/best.keras` | evaluasi model tersimpan di test split |
| `python main.py --list-params` | dump semua parameter + tipe + default |

Menu interaktif dikelompokkan 5 seksi, tiap prompt: `nama (default: X) [Enter=default]:`
dengan validasi tipe + range langsung di tempat (input salah → ulang, tidak crash).

---

## 3. Parameter yang bisa di-set (lengkap)

### 3.1 `data.*`
`root` (path dataset), `variant` (color|grayscale|segmented), `img_height`, `img_width`,
`channels`, `class_subset` (list kelas / `all`), `max_per_class` (0=all),
`val_split`, `test_split`, `split_strategy` (stratified|per_class_tail),
`shuffle_seed`, `batch_size`, `cache` (none|memory|disk), `prefetch`,
`augment` (on|off) + sub-key `rotation_range`, `width_shift`, `height_shift`, `zoom_range`,
`shear_range`, `horizontal_flip`, `vertical_flip`, `brightness_range`, `contrast_range`,
`fill_mode`, `rescale` (on|off), `normalize` (none|div255|imagenet).

### 3.2 `model.*`
`arch` (simple_cnn|vgg16|mobilenetv2|resnet50|efficientnetb0), `include_top`,
`weights` (imagenet|none), `freeze_base`, `fine_tune_at` (layer index, 0=off),
`dropout`, `dense_units`, `batch_norm` (on|off), `l2_weight_decay`,
`conv_blocks` (khusus simple_cnn: jumlah blok), `filters_base`, `kernel_size`, `pool_size`.

### 3.3 `train.*`
`optimizer` (adam|adamw|sgd|rmsprop|nadam), `learning_rate`, `momentum`, `beta_1`, `beta_2`,
`epsilon`, `loss` (categorical_crossentropy|sparse_cce|focal), `label_smoothing`,
`epochs`, `class_weight` (none|balanced|manual), `early_stopping` (on|off) +
`monitor`, `patience`, `min_delta`, `mode`, `restore_best_weights`,
`reduce_lr` (on|off) + `factor`, `patience`, `min_lr`, `cooldown`,
`checkpoint_monitor`, `save_best_only`, `mixed_precision` (off|mixed_float16),
`steps_per_epoch` (0=auto), `validation_steps`.

### 3.4 `runtime.*`
`device` (cpu|gpu|auto), `run_name`, `runs_dir`, `seed`, `verbose` (0|1|2),
`log_csv` (on|off), `tensorboard` (on|off), `resume_from` (path .keras / none),
`num_parallel_calls`, `deterministic` (on|off).

### 3.5 `eval.*`
`batch_size`, `top_k`, `confusion_matrix` (on|off), `classification_report` (on|off),
`per_class_metrics` (on|off), `sample_predictions` (jumlah gambar grid),
`save_plots` (on|off), `split` (val|test).

Total **~70 parameter**, semuanya punya default di `configs/default.yaml` dan semuanya
bisa diubah dari terminal.

---

## 4. Isi tiap file (ringkas)

**`src/config.py`** — `@dataclass` per seksi (`DataCfg`, `ModelCfg`, `TrainCfg`, `RuntimeCfg`,
`EvalCfg`) + `Config`. Fungsi: `from_yaml()`, `apply_overrides(key=value)`,
`interactive_wizard()`, `validate()` (cek path ada, split < 1, arch dikenal, lr > 0,
img size kelipatan 32 untuk transfer learning, dll).

**`src/data.py`** — `scan_dataset(root, variant)` → list `(path, label)`; cek jumlah per kelas
dan cetak tabel distribusi; `split_indices()` stratified; `build_dataset()` → `tf.data.Dataset`
dengan `map`(decode→resize→augment→normalize)→`batch`→`cache`→`prefetch`; `compute_class_weights()`.
Tangani nama folder bertanda `___` dan karakter aneh (`(`, `,`, `_`) — label = string apa adanya.

**`src/model.py`** — `build_model(cfg, num_classes)`: `simple_cnn` (stack Conv-BN-ReLU-Pool
sesuai `conv_blocks`/`filters_base`) atau transfer learning (base frozen + GAP + Dropout + Dense
softmax). `compile_model()` memilih optimizer/loss/metrics dari cfg. `focal_loss()` kalau dipilih.

**`src/train.py`** — `run_training(cfg)`: seed → scan → split → dataset → model → callbacks
(`ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`, `CSVLogger`, `TensorBoard` opsional)
→ `model.fit()` → simpan `best.keras` + `history.json` + `config_used.yaml` di `runs/<run_name>/`.

**`src/evaluate.py`** — load model, prediksi split, `classification_report`, confusion matrix
(heatmap 38 kelas), kurva loss/accuracy, grid `sample_predictions` gambar + label asli vs prediksi,
simpan `metrics.json` + `*_cm.png` + `*_loss_acc.png` + `*_preds.png` (pola sama dengan M11/M12).

**`src/utils.py`** — `set_seed()`, `get_run_dir()`, `save_json()`, `print_config_table()`.

---

## 5. Verifikasi yang akan saya jalankan (BUKAN training)

1. `python main.py --list-params` → semua ~70 parameter tampil.
2. `python main.py --check --config configs/default.yaml` → config valid, model `summary()`
   tercetak, jumlah kelas & distribusi dataset terhitung.
3. `python main.py --check --set model.arch=mobilenetv2 --set data.img_height=128` → override bekerja.
4. Menu interaktif: jalankan, tekan Enter berulang sampai prompt selesai, lalu **batal di
   konfirmasi akhir** ("Mulai training? [y/N]" → `N`). Membuktikan wizard jalan tanpa training.
5. `python -m py_compile` semua file.

Training hanya dijalankan kalau kamu perintahkan.

---

## 6. Keputusan (sudah di-ACC user)

| # | Pertanyaan | Keputusan |
|---|---|---|
| Q1 | Dataset | **Download manual dari Kaggle**, taruh di `UAS/data/PlantVillage/`. Tidak ada script download. |
| Q2 | GPU | `runtime.device` = `cpu` / `gpu` / `auto`. Default `auto` (pakai GPU kalau TF melihatnya, kalau tidak CPU). `gpu` gagal cepat dengan pesan jelas. |
| Q3 | Kelas | 38 kelas penuh, dengan `data.class_subset` untuk subset opsional. |
| Q4 | Arsitektur | Default **`mobilenetv2`** pretrained. `simple_cnn`, `vgg16`, `resnet50`, `efficientnetb0` tersedia. |

Hasil akhir: **86 parameter** (bukan ~70 seperti estimasi awal), karena augmentasi,
optimizer, dan callback dipecah ke key masing-masing agar semuanya bisa di-set.

---

## 7. Status implementasi

Semua file di §1 sudah dibuat dan diverifikasi **tanpa training**:

| Verifikasi | Hasil |
|---|---|
| `--list-params` | 86 parameter tercetak |
| `--check` mobilenetv2 | Config OK, 3 kelas, 36 gambar, pixel `-1.000..1.000` ✅ |
| `--check` simple_cnn / vgg16 / resnet50 / efficientnetb0 | keempatnya build; normalisasi per-arch benar (vgg/resnet `-116..151`, efficientnet `0..243`) |
| Wizard interaktif | 86 prompt jalan, konfirmasi akhir dijawab `N` → "Dibatalkan. Tidak ada training dijalankan." |
| `--eval` model tersimpan | load `best.keras` → metrik + confusion matrix + report PNG/JSON ✅ |
| Validasi config salah | split > 1, lr = 0, img bukan kelipatan 32, `device=gpu` tanpa GPU, key typo → pesan jelas, exit 1 |
| `py_compile` semua file | OK |

**Bug yang ditemukan & diperbaiki saat verifikasi:**

1. Augmentasi berjalan **setelah** normalisasi. `RandomBrightness`/`RandomContrast` punya
   `value_range=(0,1)` default dan meng-clip ImageNet `[-1,1]` kembali ke `[0,1]`
   (terbukti: pixel range `0.000..0.983` alih-alih `-1..1`). Urutan dibalik:
   decode → augment → normalize → batch.
2. YAML 1.1 mem-parse `off` tanpa kutip sebagai `False` → `mixed_precision: off` gagal
   validasi. Diperbaiki di YAML (dikutip) **dan** di `coerce()` (bool → "on"/"off").
3. `validate()` membaca `data["weights"]` padahal key-nya `model.weights` → `KeyError`.
4. `RandomShear` di Keras 3 butuh `x_factor`/`y_factor` terpisah, bukan satu argumen.
5. Override dengan nama parameter typo membocorkan traceback → sekarang ditangkap
   jadi `Config tidak valid: 'parameter tidak dikenal: model.archh'`.

---

## 8. Catatan risiko

- `emmarex/plantdisease` adalah snapshot lama (~342 MB). Jumlah file per kelas di kartu
  dataset tidak konsisten dengan klaim 54k, jadi kode **tidak** meng-hardcode jumlah/kelas:
  semua dibaca dari nama folder saat runtime. Kalau ternyata jumlahnya beda, tetap jalan.
- 38 kelas tidak seimbang (Apple_healthy 1645 vs Potato_healthy 152) → `class_weight=balanced`
  jadi default yang disarankan.
- TF di CPU memakai semua core → mesin bisa nge-lag saat training. Default `num_parallel_calls`
  saya set 4, bukan `AUTOTUNE`.
