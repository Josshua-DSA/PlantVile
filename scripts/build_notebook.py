"""Generate UAS/notebook/plantvillage_cnn.ipynb.

Run:  /home/josjiez/my_env/ai_env/bin/python scripts/build_notebook.py
The notebook never trains unless RUN_TRAINING is flipped to True by hand.
"""
import json
import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "notebook", "plantvillage_cnn.ipynb")
KERNEL = "ncomp-ai"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(True)}


cells = []

# ---------------------------------------------------------------- 0. header
cells.append(md("""# UAS Neuro Computing — Klasifikasi Penyakit Daun (PlantVillage + CNN)

**Nama:** Joshua Remedial Syeba
**NRP:** 3324600011
**Kelas:** 3 D4 Sains Data Terapan A (SDT A Semester 5)
**Mata Kuliah:** Neuro Computing
**Laboratorium:** Knowledge Engineering Laboratory (iKnow Lab) — PENS

---

## Tentang notebook ini

Notebook ini adalah **antarmuka eksperimen** dari project di `UAS/`.
Logika training, model, dan evaluasi tetap ada di `src/` — notebook hanya mengatur
parameter dan menampilkan hasil. Artinya: hasil di sini identik dengan
`python main.py --config configs/default.yaml`.

| Bagian | Isi |
|---|---|
| 1 | Setup environment & device |
| 2 | Konfigurasi — **semua parameter bisa diubah di sini** |
| 3 | Eksplorasi dataset (EDA) |
| 4 | Verifikasi split |
| 5 | Inspeksi pipeline & augmentasi |
| 6 | Arsitektur model |
| 7 | Training (dijalankan manual, tidak otomatis) |
| 8 | Evaluasi & metrik |
| 9 | Kurva pembelajaran |
| 10 | Inferensi satu gambar |
| 11 | Ekspor hasil |
| 12 | Referensi seluruh parameter |

> **Penting:** sel training (bagian 7) dipagari flag `RUN_TRAINING = False`.
> `Run All` tidak akan memulai training sampai kamu ubah flag itu sendiri.
"""))

# ---------------------------------------------------------------- 1. setup
cells.append(md("""## 1. Setup

Menentukan working directory project, mengimpor modul `src/`, dan mengecek device.
"""))

cells.append(code("""import os, sys, json
from pathlib import Path

# --- cari root project (folder yang berisi main.py + src/) ---
ROOT = Path.cwd()
while ROOT != ROOT.parent and not (ROOT / "main.py").exists():
    ROOT = ROOT.parent
assert (ROOT / "main.py").exists(), "main.py tidak ditemukan - buka notebook dari dalam folder UAS/"
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print("Project root:", ROOT)
"""))

cells.append(code("""import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config as C
from src.data import prepare_datasets, scan_dataset, split_dataset, build_dataset
from src.model import build_model, compile_model
from src.utils import configure_device, set_seed, print_kv_table, make_run_dir

print("TensorFlow :", tf.__version__)
print("Keras      :", keras.__version__)
print("NumPy      :", np.__version__)
print("GPU terlihat TF:", tf.config.list_physical_devices("GPU") or "tidak ada (CPU-only)")

# plot default yang enak dibaca
plt.rcParams["figure.dpi"] = 110
plt.rcParams["figure.figsize"] = (12, 5)
"""))

# ---------------------------------------------------------------- 2. config
cells.append(md("""## 2. Konfigurasi

Semua parameter ada di satu dict. Cara mengubahnya:

- langsung edit nilai di dict `OVERRIDES` di bawah, **atau**
- pakai `--set`-style string, contoh `"data.batch_size=64"`

Nilai yang tersedia per parameter bisa dilihat di **bagian 12** atau dengan
`python main.py --list-params`.
"""))

cells.append(code("""# ============================================================
#  UBAH PARAMETER DI SINI
#  Format: "section.nama=value"  (sama seperti CLI --set)
# ============================================================
# Seluruh 86 parameter ditampilkan, dikelompokkan per section.
# Nilainya = default dari configs/default.yaml. Ubah yang perlu saja.
OVERRIDES = [
    # ---- data (27) --------------------------------------------------------
    "data.root=data",                          # path
    "data.variant=color",                      # color | grayscale | segmented
    "data.img_height=128",                     # int
    "data.img_width=128",                      # int
    "data.channels=3",                         # int (1=grayscale, 3=RGB)
    "data.class_subset=",                      # list, kosong = semua kelas
    "data.max_per_class=0",                    # int, 0 = semua gambar
    "data.val_split=0.15",                     # float
    "data.test_split=0.15",                    # float
    "data.split_strategy=stratified",          # stratified | per_class_tail
    "data.shuffle_seed=42",                    # int
    "data.deduplicate=true",                   # bool
    "data.group_same_leaf=true",               # bool
    "data.batch_size=32",                      # int
    "data.cache=none",                         # none | memory | disk
    "data.prefetch=0",                         # int (0 = AUTOTUNE)
    "data.augment=true",                       # bool
    "data.aug_rotation=0.04",                  # float (fraksi 2pi)
    "data.aug_width_shift=0.10",               # float
    "data.aug_height_shift=0.10",              # float
    "data.aug_zoom=0.10",                      # float
    "data.aug_shear=0.0",                      # float
    "data.aug_hflip=true",                     # bool
    "data.aug_vflip=false",                    # bool
    "data.aug_brightness=0.15",                # float
    "data.aug_contrast=0.15",                  # float
    "data.aug_fill_mode=reflect",              # constant | reflect | wrap | nearest
    "data.rescale=true",                       # bool
    "data.normalize=imagenet",                 # none | div255 | imagenet

    # ---- model (13) -------------------------------------------------------
    "model.arch=mobilenetv2",                  # simple_cnn | vgg16 | mobilenetv2 | resnet50 | efficientnetb0
    "model.include_top=false",                 # bool
    "model.weights=imagenet",                  # imagenet | none
    "model.freeze_base=true",                  # bool
    "model.fine_tune_at=0",                    # int, 0 = semua layer base frozen
    "model.dropout=0.3",                       # float
    "model.dense_units=256",                   # int
    "model.batch_norm=true",                   # bool
    "model.l2_weight_decay=0.0",               # float
    "model.conv_blocks=4",                     # int (simple_cnn saja)
    "model.filters_base=32",                   # int (simple_cnn saja)
    "model.kernel_size=3",                     # int (simple_cnn saja)
    "model.pool_size=2",                       # int (simple_cnn saja)

    # ---- train (28) -------------------------------------------------------
    "train.optimizer=adam",                    # adam | adamw | sgd | rmsprop | nadam
    "train.learning_rate=0.001",               # float
    "train.momentum=0.9",                      # float (sgd)
    "train.beta_1=0.9",                        # float
    "train.beta_2=0.999",                      # float
    "train.epsilon=1e-07",                     # float
    "train.loss=categorical_crossentropy",     # categorical_crossentropy | sparse_categorical_crossentropy | focal
    "train.label_smoothing=0.0",               # float
    "train.epochs=20",                         # int
    "train.class_weight=balanced",             # none | balanced | manual
    "train.class_weight_manual=",              # list, dipakai kalau class_weight=manual
    "train.early_stopping=true",               # bool
    "train.es_monitor=val_loss",               # val_loss | val_accuracy
    "train.es_patience=5",                     # int
    "train.es_min_delta=0.0001",               # float
    "train.es_mode=auto",                      # auto | min | max
    "train.es_restore_best=true",              # bool
    "train.reduce_lr=true",                    # bool
    "train.lr_monitor=val_loss",               # val_loss | val_accuracy
    "train.lr_factor=0.5",                     # float
    "train.lr_patience=3",                     # int
    "train.lr_min_lr=1e-06",                   # float
    "train.lr_cooldown=0",                     # int
    "train.checkpoint_monitor=val_accuracy",   # val_loss | val_accuracy
    "train.save_best_only=true",               # bool
    "train.mixed_precision=off",               # off | mixed_float16
    "train.steps_per_epoch=0",                 # int, 0 = otomatis
    "train.validation_steps=0",                # int, 0 = otomatis

    # ---- runtime (11) -----------------------------------------------------
    "runtime.device=auto",                     # cpu | gpu | auto
    "runtime.run_name=",                       # str, kosong = timestamp otomatis
    "runtime.runs_dir=runs",                   # str
    "runtime.seed=42",                         # int
    "runtime.verbose=1",                       # int (0/1/2)
    "runtime.log_csv=true",                    # bool
    "runtime.tensorboard=false",               # bool
    "runtime.resume_from=",                    # str, path .keras untuk lanjut training
    "runtime.num_parallel_calls=4",            # int (0 = AUTOTUNE)
    "runtime.deterministic=false",             # bool
    "runtime.strategy=auto",                   # auto | none | mirrored

    # ---- eval (14) --------------------------------------------------------
    "eval.eval_batch_size=32",                 # int
    "eval.top_k=3",                            # int
    "eval.confusion_matrix=true",              # bool
    "eval.classification_report=true",         # bool
    "eval.per_class_metrics=true",             # bool
    "eval.sample_predictions=16",              # int
    "eval.save_plots=true",                    # bool
    "eval.eval_split=test",                    # val | test | train
    "eval.error_analysis=true",                # bool
    "eval.misclassified_samples=16",           # int
    "eval.top_confused_pairs=20",              # int
    "eval.gradcam=true",                       # bool
    "eval.gradcam_samples=8",                  # int
    "eval.gradcam_alpha=0.4",                  # float
]

cfg = C.from_yaml(str(ROOT / "configs" / "default.yaml"))
cfg = C.apply_overrides(cfg, OVERRIDES)

problems = C.validate(cfg)
if problems:
    print("PERINGATAN konfigurasi:")
    for p in problems:
        print("  !", p)
else:
    print("Konfigurasi valid.")
"""))

cells.append(code("""# tabel konfigurasi aktif
rows = []
for section, params in C.SPEC.items():
    for name, *_ in params:
        rows.append(("%s.%s" % (section, name), cfg[section][name]))
print_kv_table(rows, title="Konfigurasi aktif")
"""))

# ---------------------------------------------------------------- 3. EDA
cells.append(md("""## 3. Eksplorasi dataset

Memindai folder, menghitung jumlah gambar per kelas, dan melihat contoh citra.
Tidak mengubah data apa pun (read-only).
"""))

cells.append(code("""paths, labels, class_names, counts = scan_dataset(
    cfg["data"]["root"], cfg["data"]["variant"],
    cfg["data"]["class_subset"], cfg["data"]["max_per_class"],
)

print("Jumlah kelas   :", len(class_names))
print("Total gambar   :", len(paths))
print("Resolusi target:", cfg["data"]["img_height"], "x", cfg["data"]["img_width"])
print()
df_counts = pd.DataFrame({
    "kelas": class_names,
    "jumlah": [counts[i] for i in range(len(class_names))],
}).sort_values("jumlah", ascending=False).reset_index(drop=True)
df_counts["persen"] = (df_counts["jumlah"] / df_counts["jumlah"].sum() * 100).round(2)
df_counts
"""))

cells.append(code("""fig, ax = plt.subplots(figsize=(11, 7))
ax.barh(df_counts["kelas"][::-1], df_counts["jumlah"][::-1], color="#2f7d5d")
ax.set_xlabel("jumlah gambar")
ax.set_title("Distribusi kelas - total %d gambar" % len(paths))
for i, v in enumerate(df_counts["jumlah"][::-1]):
    ax.text(v + max(df_counts["jumlah"]) * 0.01, i, str(v), va="center", fontsize=9)
plt.tight_layout()
plt.show()

ratio = df_counts["jumlah"].max() / df_counts["jumlah"].min()
print("Rasio imbalance (max/min): %.1fx" % ratio)
print("Kelas terkecil:", df_counts.iloc[-1]["kelas"], df_counts.iloc[-1]["jumlah"])
print("Kelas terbesar:", df_counts.iloc[0]["kelas"], df_counts.iloc[0]["jumlah"])
"""))

cells.append(code("""# contoh citra per kelas
import random
from PIL import Image

rng = random.Random(0)
n_show = min(12, len(class_names))
picked = rng.sample(range(len(class_names)), n_show)

fig, axes = plt.subplots(3, 4, figsize=(13, 9))
for ax, idx in zip(axes.ravel(), picked):
    cls = class_names[idx]
    member = [p for p, l in zip(paths, labels) if l == idx]
    fp = rng.choice(member)
    with Image.open(fp) as im:
        ax.imshow(im.convert("RGB"))
    ax.set_title("%s\\n%s" % (cls, os.path.basename(fp)), fontsize=8)
    ax.axis("off")
plt.suptitle("Contoh citra per kelas", y=1.01)
plt.tight_layout()
plt.show()
"""))

cells.append(code("""# properti fisik citra (sampel 300 file)
sample = rng.sample(paths, min(300, len(paths)))
modes, sizes = {}, {}
for fp in sample:
    with Image.open(fp) as im:
        modes[im.mode] = modes.get(im.mode, 0) + 1
        sizes[im.size] = sizes.get(im.size, 0) + 1

print("Mode warna :", modes)
print("Ukuran     :", dict(sorted(sizes.items(), key=lambda kv: -kv[1])[:5]))
print()
print("Nama folder pakai '___' :", sum(1 for c in class_names if "___" in c))
print("Nama folder pakai '_'   :", sum(1 for c in class_names if "___" not in c))
"""))

# ---------------------------------------------------------------- 4. split
cells.append(md("""## 4. Verifikasi split

Split dilakukan **per kelas** (stratified) supaya proporsi tiap kelas sama di
train/val/test. Sel ini memastikan tidak ada kebocoran dan proporsinya benar.
"""))

cells.append(code("""tr_idx, va_idx, te_idx = split_dataset(
    paths, labels,
    cfg["data"]["val_split"], cfg["data"]["test_split"],
    cfg["data"]["split_strategy"], cfg["data"]["shuffle_seed"],
)

print("train : %5d (%.1f%%)" % (len(tr_idx), 100*len(tr_idx)/len(paths)))
print("val   : %5d (%.1f%%)" % (len(va_idx), 100*len(va_idx)/len(paths)))
print("test  : %5d (%.1f%%)" % (len(te_idx), 100*len(te_idx)/len(paths)))

# cek kebocoran: tidak boleh ada path yang muncul di dua split
s_tr, s_va, s_te = set(tr_idx), set(va_idx), set(te_idx)
assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), "ADA KEBOCORAN SPLIT"
print("\\nTidak ada kebocoran antar split.")
"""))

cells.append(code("""# distribusi per kelas di tiap split
rows = []
for i, cls in enumerate(class_names):
    n_tr = sum(1 for j in tr_idx if labels[j] == i)
    n_va = sum(1 for j in va_idx if labels[j] == i)
    n_te = sum(1 for j in te_idx if labels[j] == i)
    rows.append({"kelas": cls, "train": n_tr, "val": n_va, "test": n_te,
                 "total": n_tr + n_va + n_te})
pd.DataFrame(rows)
"""))

# ---------------------------------------------------------------- 5. pipeline
cells.append(md("""## 5. Pipeline & augmentasi

Menampilkan satu batch nyata dari `tf.data`: bentuk tensor, rentang piksel, dan
efek augmentasi. Ini cara cepat memastikan normalisasi tidak salah
(untuk MobileNetV2 rentangnya harus `-1..1`).
"""))

cells.append(code("""bundle = prepare_datasets(cfg, verbose=False)

print("Kelas         :", bundle["num_classes"])
print("Train / Val / Test:", len(bundle["train"]["paths"]),
      "/", len(bundle["val"]["paths"]), "/", len(bundle["test"]["paths"]))
print("Class weight  :", bundle["class_weights"] is not None)

x_batch, y_batch = next(iter(bundle["train"]["ds"]))
print("\\nBentuk batch :", x_batch.shape, y_batch.shape)
print("Rentang piksel: %.3f .. %.3f" % (float(x_batch.numpy().min()),
                                         float(x_batch.numpy().max())))
"""))

cells.append(code("""# grid batch + labelnya
fig, axes = plt.subplots(4, 4, figsize=(11, 11))
for ax, img, lab in zip(axes.ravel(), x_batch.numpy(), y_batch.numpy()):
    shown = (img - img.min()) / max(img.max() - img.min(), 1e-9)   # hanya untuk display
    ax.imshow(shown)
    idx = int(np.argmax(lab)) if np.ndim(lab) else int(lab)
    ax.set_title(class_names[idx], fontsize=7)
    ax.axis("off")
plt.suptitle("Satu batch training (setelah augmentasi + normalisasi)")
plt.tight_layout()
plt.show()
"""))

cells.append(code("""# efek augmentasi: satu gambar, 6 variasi
aug_ds = build_dataset([paths[tr_idx[0]]] * 6, [labels[tr_idx[0]]] * 6,
                       cfg["data"], cfg["runtime"], cfg["model"],
                       training=True, batch_size=6)
ax_imgs, _ = next(iter(aug_ds))

fig, axes = plt.subplots(2, 3, figsize=(11, 7))
for ax, img in zip(axes.ravel(), ax_imgs.numpy()):
    shown = (img - img.min()) / max(img.max() - img.min(), 1e-9)
    ax.imshow(shown)
    ax.axis("off")
plt.suptitle("Augmentasi acak pada satu citra yang sama")
plt.tight_layout()
plt.show()
"""))

# ---------------------------------------------------------------- 6. model
cells.append(md("""## 6. Arsitektur model

Membangun model sesuai `model.arch`. Backbone pretrained dibekukan sesuai
`model.freeze_base`, lalu ditambah head klasifikasi.
"""))

cells.append(code("""set_seed(cfg["runtime"]["seed"], cfg["runtime"]["deterministic"])
device_info = configure_device(cfg["runtime"]["device"])
print("Device aktif:", device_info["active"])

model = build_model(cfg, bundle["num_classes"])
model = compile_model(model, cfg["train"])
model.summary()
"""))

cells.append(code("""# ringkas parameter
n_train = int(sum(np.prod(w.shape) for w in model.trainable_weights))
n_frozen = int(sum(np.prod(w.shape) for w in model.non_trainable_weights))
print("trainable : %s" % f"{n_train:,}")
print("frozen    : %s" % f"{n_frozen:,}")
print("total     : %s" % f"{n_train + n_frozen:,}")
"""))

# ---------------------------------------------------------------- 7. training
cells.append(md("""## 7. Training

> **Sel ini TIDAK jalan otomatis.** `RUN_TRAINING = False` di bawah.
> Ubah ke `True` hanya kalau kamu memang mau mulai training.
>
> Estimasi waktu (CPU, 128px, MobileNetV2, 20.638 gambar): ~4-8 menit/epoch.
> Kalau mau cepat untuk uji alur, set `data.max_per_class=500` dan `train.epochs=2`
> di bagian 2.
"""))

cells.append(code("""RUN_TRAINING = False   # <-- ubah ke True untuk mulai training

# run_training() di src/train.py sudah menangani device, mixed precision,
# resume, callbacks, dan penyimpanan artefak ke runs/<run_name>/.
from src.train import run_training

if not RUN_TRAINING:
    print("Training dilewati (RUN_TRAINING = False).")
    print("Ubah flag di atas ke True kalau sudah siap.")
    print("Estimasi: %d epoch, %d gambar train, batch %d"
          % (cfg["train"]["epochs"], len(bundle["train"]["paths"]),
             cfg["data"]["batch_size"]))
else:
    model, history, run_dir, bundle = run_training(cfg)
    print("")
    print("Epoch dijalankan :", len(history["loss"]))
    print("Artefak          :", run_dir)
"""))

# ---------------------------------------------------------------- 8. eval
cells.append(md("""## 8. Evaluasi

Mengukur model pada split test: akurasi, precision/recall/F1, confusion matrix,
dan classification report per kelas.

Bisa dipakai dua cara:
- setelah training di bagian 7, atau
- memuat model yang sudah tersimpan (`MODEL_PATH`).
"""))

cells.append(code("""from src.evaluate import evaluate as run_evaluation

MODEL_PATH = ""   # contoh: "runs/20261121-190000_mobilenetv2/best.keras"

# Model hanya dievaluasi kalau memang sudah terlatih.
# Model hasil build di bagian 6 belum terlatih - mengevaluasinya cuma membuang waktu.
TRAINED = bool(MODEL_PATH) or bool(globals().get("RUN_TRAINING"))

eval_model = None
if MODEL_PATH:
    eval_model = keras.models.load_model(MODEL_PATH, compile=False)
    print("Model dimuat dari", MODEL_PATH)
elif TRAINED and "model" in dir():
    eval_model = model
    print("Memakai model hasil training dari bagian 7.")
else:
    print("Evaluasi belum relevan: model belum dilatih.")
    print("Jalankan bagian 7 (RUN_TRAINING = True) atau isi MODEL_PATH.")
"""))

cells.append(code("""if eval_model is not None:
    metrics = run_evaluation(
        eval_model, cfg,
        bundle=bundle,
        run_dir=str(ROOT / "runs" / "notebook_eval"),
        tag="nb_" + cfg["eval"]["eval_split"],
    )
else:
    print("Evaluasi dilewati: model belum terlatih.")
"""))

# ---------------------------------------------------------------- 9. curves
cells.append(md("""## 9. Kurva pembelajaran

Membaca `history.json` dari run terakhir (kalau ada) untuk melihat akurasi dan loss
per epoch — dipakai untuk menilai underfit / overfit.
"""))

cells.append(code("""history_path = None
if "run_dir" in dir() and os.path.isfile(os.path.join(run_dir, "history.json")):
    history_path = os.path.join(run_dir, "history.json")

if history_path is None:
    print("Belum ada history.json. Jalankan training dulu di bagian 7.")
else:
    with open(history_path) as fh:
        hist = json.load(fh)
    print("epoch tercatat:", len(next(iter(hist.values()))))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(hist["accuracy"], label="train")
    axes[0].plot(hist["val_accuracy"], label="val")
    axes[0].set_title("Akurasi")
    axes[0].set_xlabel("epoch"); axes[0].legend(); axes[0].grid(alpha=.3)

    axes[1].plot(hist["loss"], label="train")
    axes[1].plot(hist["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("epoch"); axes[1].legend(); axes[1].grid(alpha=.3)

    plt.tight_layout(); plt.show()

    gap = hist["accuracy"][-1] - hist["val_accuracy"][-1]
    print("Selisih akurasi train-val terakhir: %+.3f" % gap)
    print("Interpretasi: gap > 0.15 -> indikasi overfit; "
          "keduanya rendah -> underfit.")
"""))

# ---------------------------------------------------------------- 10. inference
cells.append(md("""## 10. Inferensi satu gambar + Grad-CAM

Uji model pada satu file citra sekaligus memvisualisasikan peta aktivasi Grad-CAM
untuk melihat area daun yang mendorong prediksi model.
"""))

cells.append(code("""def predict_image(path, top_k=3):
    \"\"\"Prediksi + Grad-CAM untuk satu citra memakai src.gradcam.\"\"\"
    from src.gradcam import explain_image

    return explain_image(
        eval_model, path, cfg, class_names,
        top_k=top_k, alpha=cfg["eval"]["gradcam_alpha"],
    )


SAMPLE = ""   # kosong = ambil otomatis satu citra dari split test

if eval_model is None:
    print("Belum ada model terlatih - inferensi dilewati.")
else:
    if not SAMPLE:
        SAMPLE = bundle["test"]["paths"][0]
        print("Demo otomatis memakai citra test:", SAMPLE)

    result = predict_image(SAMPLE, top_k=cfg["eval"]["top_k"])
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(result["original_rgb"])
    axes[0].set_title("Citra Asli")
    axes[0].axis("off")
    axes[1].imshow(result["overlay_rgb"])
    axes[1].set_title("Grad-CAM: %s (%.2f)" % (result["pred_class"], result["pred_prob"]))
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()

    print("")
    print("Prediksi top-%d:" % len(result["top_k"]))
    for item in result["top_k"]:
        print("  %-46s %.4f" % (item["class"], item["prob"]))
    print("")
    if TRAINED:
        print("Model sudah dilatih - prediksi di atas berasal dari bobot hasil training.")
    else:
        print("Model belum dilatih: bobot masih acak, prediksi belum bermakna.")
"""))

# ---------------------------------------------------------------- 11. export
cells.append(md("""## 11. Ekspor hasil

Menyatukan metrik + konfigurasi ke satu file JSON untuk lampiran laporan.
"""))

cells.append(code("""summary = {
    "dataset": {
        "root": cfg["data"]["root"],
        "kelas": len(class_names),
        "total_gambar": len(paths),
        "train": len(bundle["train"]["paths"]),
        "val": len(bundle["val"]["paths"]),
        "test": len(bundle["test"]["paths"]),
    },
    "model": {"arch": cfg["model"]["arch"], "weights": cfg["model"]["weights"]},
    "training": {k: cfg["train"][k] for k in
                 ("optimizer", "learning_rate", "loss", "epochs", "class_weight")},
    "metrics": metrics if "metrics" in dir() else None,
    "catatan": "model belum dilatih di notebook ini" if not TRAINED else "hasil training",
}

out = ROOT / "runs" / "notebook_summary.json"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as fh:
    json.dump(summary, fh, indent=2, default=str)
print("Ringkasan disimpan:", out)
print(json.dumps(summary, indent=2, default=str)[:1200])
"""))

# ---------------------------------------------------------------- 12. reference
cells.append(md("""## 12. Referensi parameter

Daftar lengkap seluruh parameter beserta tipe dan default-nya.
"""))

cells.append(code("""print(C.list_params())
"""))

cells.append(code("""# versi ringkas sebagai tabel
rows = []
for section, params in C.SPEC.items():
    for name, kind, default, choices, help_text in params:
        rows.append({"parameter": "%s.%s" % (section, name), "tipe": kind,
                     "default": default, "pilihan": choices or "", "keterangan": help_text})
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 200)
pd.DataFrame(rows)
"""))

for _i, _c in enumerate(cells):
    _c["id"] = "cell-%02d" % _i

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python (NeuroComputing ai_env)",
                       "language": "python", "name": KERNEL},
        "language_info": {"name": "python", "version": "3.12.3",
                          "mimetype": "text/x-python",
                          "file_extension": ".py",
                          "pygments_lexer": "ipython3",
                          "nbconvert_exporter": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)
print("Notebook ditulis: %s" % OUT)
print("cells: %d (%d markdown, %d code)"
      % (len(cells),
         sum(1 for c in cells if c["cell_type"] == "markdown"),
         sum(1 for c in cells if c["cell_type"] == "code")))