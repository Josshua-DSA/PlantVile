"""Generate notebook/data_audit_eda.ipynb for PlantVillage audit and EDA.

Run:
    python scripts/build_eda_notebook.py

The generated notebook uses only standard library + NumPy, Pandas, Pillow,
Matplotlib, and Seaborn (no TensorFlow/Keras dependency) so it runs smoothly
both in lightweight local environments and on Kaggle.
"""
import os
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "notebook" / "data_audit_eda.ipynb"


def md(text: str):
    """Create a markdown cell."""
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    """Create a code cell with stripped outputs."""
    return nbf.v4.new_code_cell(text.strip() + "\n")


def build_cells():
    """Construct the ordered list of notebook cells."""
    cells = []

    # ---------------------------------------------------------------- 0. Header
    cells.append(md("""
# Audit Data & EDA — PlantVillage (PlantVile)

Notebook ini berisi **audit integritas dataset** dan **Exploratory Data Analysis (EDA)** secara *read-only* sebelum eksperimen training di Kaggle.
Seluruh angka dan visualisasi di sini menjadi dasar penyusunan hipotesis eksperimen pada `docs/DATA_AUDIT_AND_EDA.md` dan `EXPERIMENT_PROTOCOL.md`.

| Bagian | Fokus Analisis |
| :--- | :--- |
| 1 | Setup environment & resolusi path dataset (`data/PlantVillage`) |
| 2 | Audit struktur folder, ekstensi, file non-citra, & ukuran byte |
| 3 | Audit duplikat persis (*Exact MD5*) & risiko kebocoran antar-split |
| 4 | Verifikasi fisik citra via Pillow (korup, resolusi, mode warna) |
| 5 | Audit tag sumber nama file, cek pre-augmentasi, & *same-leaf near-duplicates* |
| 6 | Distribusi kelas, komposisi spesies, simulasi split 70/15/15, & `class_weight` |
| 7 | Inspeksi visual sampel acak tiap kelas |
| 8 | EDA statistik piksel, kanal RGB, & kontras latar tepi (*Border vs Center*) |
| 9 | Ringkasan temuan & hipotesis eksperimen (H1–H5) |
"""))

    # ---------------------------------------------------------------- 1. Setup
    cells.append(md("""
## 1. Setup & Resolusi Path Dataset

Mencari root repositori secara otomatis dan memverifikasi folder `DATA_ROOT`.
Jika dijalankan di Kaggle, ubah `DATA_ROOT` ke path `/kaggle/input/...`.
"""))

    cells.append(code("""import hashlib
import os
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

SEED = 42
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")

ROOT = Path.cwd()
while ROOT != ROOT.parent and not (ROOT / "main.py").exists():
    ROOT = ROOT.parent
assert (ROOT / "main.py").exists(), "main.py tidak ditemukan; jalankan dari dalam folder proyek."

# Untuk Kaggle, ganti ke: Path("/kaggle/input/plantdisease/PlantVillage")
DATA_ROOT = ROOT / "data" / "PlantVillage"
assert DATA_ROOT.is_dir(), f"DATA_ROOT tidak ditemukan: {DATA_ROOT}"

plt.rcParams["figure.dpi"] = 110
sns.set_style("whitegrid")

print("Project root :", ROOT)
print("DATA_ROOT    :", DATA_ROOT)
"""))

    cells.append(code("""# Deteksi struktur folder & peringatan double-nesting (mis. PlantVillage/PlantVillage)
top_dirs = sorted(d.name for d in DATA_ROOT.iterdir() if d.is_dir())
empty_or_nested = []
class_dirs = []

for dname in top_dirs:
    sub = DATA_ROOT / dname
    has_img = any(
        f.is_file() and f.name.endswith(IMG_EXTS)
        for f in sub.iterdir()
    )
    if has_img:
        class_dirs.append(dname)
    else:
        empty_or_nested.append(dname)

print(f"Subfolder berisi citra (kelas): {len(class_dirs)}")
if empty_or_nested:
    print(f"PERINGATAN - Subfolder tanpa citra langsung: {empty_or_nested}")
    print("Pastikan folder bersarang tidak ikut terbaca oleh scan_dataset() di src/data.py.")
else:
    print("Struktur folder bersih: seluruh subfolder langsung berisi file citra.")
"""))

    # ---------------------------------------------------------------- 2. Integrity
    cells.append(md("""
## 2. Audit Struktur File, Ekstensi, & Ukuran Byte

Memindai seluruh file di dalam folder kelas untuk mencatat jumlah citra valid, file non-citra (misal `svn-r6Yb5c` 0-byte), variasi ekstensi, dan sebaran ukuran byte.
"""))

    cells.append(code("""ext_counts = Counter()
non_image_files = []
all_records = []  # (class_name, filename, full_path, byte_size)

for cls in class_dirs:
    cdir = DATA_ROOT / cls
    with os.scandir(cdir) as it:
        entries = sorted((e for e in it if e.is_file()), key=lambda e: e.name)
    for entry in entries:
        fsize = entry.stat().st_size
        _, ext = os.path.splitext(entry.name)
        ext_counts[ext] += 1
        if not entry.name.endswith(IMG_EXTS):
            non_image_files.append((cls, entry.name, fsize))
            continue
        all_records.append((cls, entry.name, entry.path, fsize))

byte_sizes = np.array([r[3] for r in all_records], dtype=np.int64)
print(f"Total kelas       : {len(class_dirs)}")
print(f"Total citra valid : {len(all_records):,}")
print(f"Total ukuran      : {byte_sizes.sum() / (1024 * 1024):.2f} MB")
print(
    "Ukuran file (B)   : min=%d | p5=%d | median=%d | p95=%d | max=%d"
    % (
        int(byte_sizes.min()),
        int(np.percentile(byte_sizes, 5)),
        int(np.median(byte_sizes)),
        int(np.percentile(byte_sizes, 95)),
        int(byte_sizes.max()),
    )
)
print("Distribusi ekstensi:", dict(ext_counts))
print("File non-citra     :", non_image_files)
"""))

    # ---------------------------------------------------------------- 3. MD5 Duplicates
    cells.append(md("""
## 3. Audit Duplikasi Persis (*Exact MD5*)

Mengelompokkan kandidat berdasarkan ukuran byte yang sama terlebih dahulu, lalu menghitung hash MD5 secara paralel untuk mendeteksi file identik di dalam kelas (*within-class*) maupun lintas kelas (*cross-class*).
"""))

    cells.append(code("""size_groups = defaultdict(list)
for cls, fname, fpath, fsize in all_records:
    size_groups[fsize].append((cls, fname, fpath))

dup_candidates = [item for grp in size_groups.values() if len(grp) > 1 for item in grp]


def _compute_md5(item):
    cls, fname, fpath = item
    with open(fpath, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest(), cls, fname


hashes = defaultdict(list)
with ThreadPoolExecutor(max_workers=16) as pool:
    for digest, cls, fname in pool.map(_compute_md5, dup_candidates):
        hashes[digest].append((cls, fname))

dup_groups = {h: items for h, items in hashes.items() if len(items) > 1}
within_class_dups = [items for items in dup_groups.values() if len({c for c, _ in items}) == 1]
cross_class_dups = [items for items in dup_groups.values() if len({c for c, _ in items}) > 1]
extra_dup_files = sum(len(items) - 1 for items in dup_groups.values())

print(f"Kandidat ukuran byte sama : {len(dup_candidates):,} / {len(all_records):,}")
print(f"Grup duplikat MD5 persis  : {len(dup_groups)} (file ekstra: {extra_dup_files})")
print(f"  - Within-class          : {len(within_class_dups)}")
print(f"  - Cross-class           : {len(cross_class_dups)}")

if dup_groups:
    rows = []
    for h, items in list(dup_groups.items())[:10]:
        rows.append({
            "md5_prefix": h[:10],
            "classes": ", ".join(sorted({c for c, _ in items})),
            "count": len(items),
            "files": " | ".join(f for _, f in items),
        })
    display(pd.DataFrame(rows))
"""))

    # ---------------------------------------------------------------- 4. PIL Verify
    cells.append(md("""
## 4. Verifikasi Fisik Citra (Pillow)

Memeriksa integritas file (`im.verify()`), mode warna (`RGB`/`RGBA`/`L`), dan resolusi pada sampel stratifikasi (100 gambar/kelas) ditambah seluruh *size outliers* (`< 5 KB` atau `> 40 KB`).
"""))

    cells.append(code("""rng = random.Random(SEED)
by_cls_records = defaultdict(list)
for rec in all_records:
    by_cls_records[rec[0]].append(rec)

verify_map = {r[2]: r for r in all_records if r[3] < 5000 or r[3] > 40000}
for cls in class_dirs:
    items = by_cls_records[cls]
    for rec in rng.sample(items, min(100, len(items))):
        verify_map[rec[2]] = rec


def _verify_image(rec):
    cls, fname, fpath, fsize = rec
    try:
        with Image.open(fpath) as im:
            im.verify()
        with Image.open(fpath) as im:
            return True, im.mode, im.size, cls, fname, fsize
    except Exception as exc:
        return False, str(exc), None, cls, fname, fsize


modes, resolutions, corrupt_files = Counter(), Counter(), []
with ThreadPoolExecutor(max_workers=16) as pool:
    for ok, mode_or_err, size, cls, fname, fsize in pool.map(_verify_image, verify_map.values()):
        if ok:
            modes[mode_or_err] += 1
            resolutions[size] += 1
        else:
            corrupt_files.append((cls, fname, fsize, mode_or_err))

print(f"Total file diverifikasi : {len(verify_map):,} (sampel + size outliers)")
print(f"File korup              : {len(corrupt_files)}")
print(f"Mode warna              : {dict(modes)}")
print(f"Resolusi                : {dict(resolutions)}")
"""))

    # ---------------------------------------------------------------- 5. Filename patterns
    cells.append(md("""
## 5. Audit Tag Sumber, Cek Pre-Augmentasi, & *Same-Leaf Near-Duplicates*

Memeriksa pola nama file untuk mendeteksi:
1. Apakah ada citra pre-augmentasi sintetis (`_flip`, `_rot`, `180deg`, dll.).
2. Tag sumber/kolektor di belakang `___` (*confounding factor* potensial).
3. Kelompok foto berurutan dari daun fisik yang sama (pola `Leaf <nomor>.<subfoto>`).
"""))

    cells.append(code("""AUG_PATTERNS = ("_flip", "_rot", "180deg", "90deg", "270deg", "final_masked")
pattern_rows = []

for cls in class_dirs:
    files = [r[1] for r in by_cls_records[cls]]
    tags = Counter()
    aug_hits = 0
    leaf_groups = defaultdict(list)

    for fname in files:
        lower = fname.lower()
        if any(p in lower for p in AUG_PATTERNS):
            aug_hits += 1
        rhs = fname.split("___", 1)[1] if "___" in fname else fname
        m_tag = re.match(r"^([A-Za-z0-9._-]+)", rhs)
        if m_tag:
            tags[m_tag.group(1)] += 1
        m_leaf = re.search(r"(Leaf\\s+\\d+)\\.\\d+", rhs, flags=re.I)
        if m_leaf:
            leaf_groups[m_leaf.group(1)].append(fname)

    multi_leaf_groups = [v for v in leaf_groups.values() if len(v) > 1]
    pattern_rows.append({
        "class": cls,
        "count": len(files),
        "pre_aug_files": aug_hits,
        "same_leaf_groups": len(multi_leaf_groups),
        "same_leaf_files": sum(len(v) for v in multi_leaf_groups),
        "top_source_tags": ", ".join(f"{k} ({v})" for k, v in tags.most_common(2)),
    })

df_patterns = pd.DataFrame(pattern_rows)
print("Total file pre-augmentasi terdeteksi:", int(df_patterns["pre_aug_files"].sum()))
display(df_patterns)
"""))

    # ---------------------------------------------------------------- 6. Class distribution & split simulation
    cells.append(md("""
## 6. Distribusi Kelas, Komposisi Spesies, & Simulasi Split 70/15/15

Mereplikasi logika `split_dataset(..., strategy="stratified", val_split=0.15, test_split=0.15, seed=42)` dan `compute_class_weights(..., mode="balanced")` dari `src/data.py` untuk memverifikasi jumlah sampel per split dan bobot penalti tiap kelas.
"""))

    cells.append(code("""paths = [r[2] for r in all_records]
labels = [class_dirs.index(r[0]) for r in all_records]

rng_split = random.Random(SEED)
by_class_idx = defaultdict(list)
for i, lbl in enumerate(labels):
    by_class_idx[lbl].append(i)

train_idx, val_idx, test_idx = [], [], []
for lbl, idxs in sorted(by_class_idx.items()):
    idxs = list(idxs)
    rng_split.shuffle(idxs)
    n = len(idxs)
    n_test = int(round(n * 0.15))
    n_val = int(round(n * 0.15))
    test_idx.extend(idxs[:n_test])
    val_idx.extend(idxs[n_test : n_test + n_val])
    train_idx.extend(idxs[n_test + n_val :])

assert not (set(train_idx) & set(val_idx))
assert not (set(train_idx) & set(test_idx))
assert not (set(val_idx) & set(test_idx))
assert len(train_idx) + len(val_idx) + len(test_idx) == len(paths)

train_counts = Counter(labels[i] for i in train_idx)
val_counts = Counter(labels[i] for i in val_idx)
test_counts = Counter(labels[i] for i in test_idx)
mean_train = sum(train_counts.values()) / len(class_dirs)

dist_rows = []
species_totals = Counter()
for i, cls in enumerate(class_dirs):
    total_cls = len(by_class_idx[i])
    sp = cls.split("_")[0]
    species_totals[sp] += total_cls
    dist_rows.append({
        "class": cls,
        "species": sp,
        "total": total_cls,
        "pct": round(100.0 * total_cls / len(paths), 2),
        "train": train_counts[i],
        "val": val_counts[i],
        "test": test_counts[i],
        "balanced_weight": round(mean_train / train_counts[i], 2),
    })

df_dist = pd.DataFrame(dist_rows).sort_values("total", ascending=False).reset_index(drop=True)
imbalance_ratio = df_dist["total"].max() / df_dist["total"].min()
print(f"Split total -> train={len(train_idx):,} | val={len(val_idx):,} | test={len(test_idx):,}")
print(f"Rasio imbalance (max/min): {imbalance_ratio:.1f}x")
print("Komposisi spesies:", dict(species_totals))
display(df_dist)
"""))

    cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [2.2, 1.0]})

df_plot = df_dist.iloc[::-1]
colors = ["#c44e52" if n < 400 else "#2f7d5d" for n in df_plot["total"]]
axes[0].barh(df_plot["class"], df_plot["total"], color=colors)
axes[0].set_title(f"Distribusi 15 Kelas PlantVillage (Total {len(paths):,}, Imbalance {imbalance_ratio:.1f}x)")
axes[0].set_xlabel("Jumlah Gambar")
for idx_bar, val in enumerate(df_plot["total"]):
    axes[0].text(val + 35, idx_bar, f"{val:,}", va="center", fontsize=8)

sp_series = pd.Series(species_totals).sort_values(ascending=False)
axes[1].bar(sp_series.index, sp_series.values, color=["#2f7d5d", "#4c72b0", "#dd8452"])
axes[1].set_title("Komposisi per Spesies")
axes[1].set_ylabel("Jumlah Gambar")
for i, v in enumerate(sp_series.values):
    axes[1].text(i, v + 250, f"{v:,}\\n({100*v/len(paths):.1f}%)", ha="center", fontsize=9)
axes[1].set_ylim(0, sp_series.max() * 1.18)

plt.tight_layout()
plt.show()
plt.close(fig)
"""))

    # ---------------------------------------------------------------- 7. Visual samples
    cells.append(md("""
## 7. Inspeksi Visual Sampel per Kelas

Menampilkan satu gambar acak (seeded) dari masing-masing 15 kelas untuk memeriksa konsistensi pencahayaan, ukuran daun, dan latar belakang.
"""))

    cells.append(code("""rng_vis = random.Random(SEED)
fig, axes = plt.subplots(3, 5, figsize=(15, 9.5))

for ax, (i, cls) in zip(axes.ravel(), enumerate(class_dirs)):
    fp = rng_vis.choice([r[2] for r in by_cls_records[cls]])
    with Image.open(fp) as im:
        rgb = im.convert("RGB")
        ax.imshow(rgb)
        ax.set_title(f"{cls}\\n({rgb.size[0]}x{rgb.size[1]}, n={len(by_cls_records[cls])})", fontsize=7.5)
    ax.axis("off")

plt.suptitle("Sampel Acak per Kelas (seed=42)", y=1.01, fontsize=12)
plt.tight_layout()
plt.show()
plt.close(fig)
"""))

    # ---------------------------------------------------------------- 8. Pixel & border-vs-center EDA
    cells.append(md("""
## 8. EDA Piksel, Kanal RGB, & Latar Belakang (*Border vs Center*)

Mengukur kecerahan global, kontras, rata-rata R/G/B, serta membandingkan pita tepi 16 px (`BrdLum`, `BrdStd`) terhadap area 50% tengah (`CtrStd`) pada 100 sampel stratifikasi per kelas (`seed=42`).
Perbedaan antar-kelas di sini menunjukkan **asosiasi statistik**, bukan bukti sebab-akibat langsung, tetapi menjadi sinyal risiko *shortcut learning* latar belakang yang perlu diuji dengan **Grad-CAM**.
"""))

    cells.append(code("""rng_px = random.Random(SEED)
px_rows = []

for cls in class_dirs:
    picked = rng_px.sample(by_cls_records[cls], min(100, len(by_cls_records[cls])))
    b_vals, c_vals, rgb_vals, brd_std_vals, ctr_std_vals, brd_lum_vals = [], [], [], [], [], []
    for _, _, fp, _ in picked:
        with Image.open(fp) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        h, w = lum.shape
        border = np.concatenate([
            lum[:16, :].ravel(),
            lum[-16:, :].ravel(),
            lum[:, :16].ravel(),
            lum[:, -16:].ravel(),
        ])
        center = lum[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]

        b_vals.append(float(lum.mean()))
        c_vals.append(float(lum.std()))
        rgb_vals.append(arr.mean(axis=(0, 1)))
        brd_std_vals.append(float(border.std()))
        ctr_std_vals.append(float(center.std()))
        brd_lum_vals.append(float(border.mean()))

    rgb_mean = np.mean(rgb_vals, axis=0)
    px_rows.append({
        "class": cls,
        "species": cls.split("_")[0],
        "bright": round(float(np.mean(b_vals)), 3),
        "contrast": round(float(np.mean(c_vals)), 3),
        "r_mean": round(float(rgb_mean[0]), 3),
        "g_mean": round(float(rgb_mean[1]), 3),
        "b_mean": round(float(rgb_mean[2]), 3),
        "brd_std": round(float(np.mean(brd_std_vals)), 3),
        "ctr_std": round(float(np.mean(ctr_std_vals)), 3),
        "brd_lum": round(float(np.mean(brd_lum_vals)), 3),
    })

df_px = pd.DataFrame(px_rows)
display(df_px)
"""))

    cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

palette = {"Tomato": "#c44e52", "Pepper": "#2f7d5d", "Potato": "#4c72b0"}
for sp, grp in df_px.groupby("species"):
    axes[0].scatter(grp["brd_lum"], grp["brd_std"], s=90, label=sp, color=palette.get(sp, "gray"), alpha=0.85)
for _, row in df_px.iterrows():
    short = row["class"].replace("Tomato__", "").replace("Tomato_", "T_").replace("Pepper__bell___", "Pep_").replace("Potato___", "Pot_")
    axes[0].annotate(short, (row["brd_lum"], row["brd_std"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
axes[0].set_title("Profil Latar Tepi (16px Border): Luminansi vs Std")
axes[0].set_xlabel("Border Luminance (BrdLum)")
axes[0].set_ylabel("Border Std (BrdStd)")
axes[0].legend()

x_pos = np.arange(len(df_px))
w_bar = 0.26
axes[1].bar(x_pos - w_bar, df_px["r_mean"], width=w_bar, color="#d62728", alpha=0.8, label="R")
axes[1].bar(x_pos, df_px["g_mean"], width=w_bar, color="#2ca02c", alpha=0.8, label="G")
axes[1].bar(x_pos + w_bar, df_px["b_mean"], width=w_bar, color="#1f77b4", alpha=0.8, label="B")
axes[1].set_title("Rata-rata Kanal R/G/B per Kelas")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(
    [c.replace("Tomato__", "").replace("Tomato_", "T_").replace("Pepper__bell___", "Pep_").replace("Potato___", "Pot_") for c in df_px["class"]],
    rotation=60, ha="right", fontsize=7.5,
)
axes[1].set_ylabel("Intensitas Mean [0, 1]")
axes[1].legend()

plt.tight_layout()
plt.show()
plt.close(fig)
"""))

    # ---------------------------------------------------------------- 9. Hypotheses
    cells.append(md("""
## 9. Ringkasan Temuan & Hipotesis Eksperimen (H1–H5)

| ID | Bukti Audit / EDA | Hipotesis Eksperimen | Variabel Bebas (OFAT) | Bukti Pengujian di Protokol |
| :--- | :--- | :--- | :--- | :--- |
| **H1** | 10 kelas adalah variasi penyakit Tomat; 2 kelas bersampel kecil (`106` & `261` train) | Pretrained `mobilenetv2` (`imagenet`) mengungguli `simple_cnn` dari nol pada `val f1_macro` | `model.arch`: `simple_cnn` vs `mobilenetv2` | `val f1_macro` & recall kelas minoritas |
| **H2** | Variasi `BrdLum` (`0,477`–`0,604`) & latar homogen (`BrdStd < 0,10` di beberapa kelas Tomat) | Augmentasi geometris + `brightness`/`contrast` mengurangi overfitting & mengarahkan fokus fitur ke daun | `data.augment`: `false` vs `true` | `val f1_macro`, gap `train - val`, & inspeksi **Grad-CAM** |
| **H3** | Imbalance **21,1x** (`BalWt=9,08` vs `0,43`) dengan 23 sampel `val` pada `Potato___healthy` | `class_weight=balanced` atau `loss=focal` menaikkan *worst-class recall* dibanding `none`, dengan potensi *trade-off* precision/kalibrasi | `class_weight`: `none` vs `balanced` vs `loss=focal` | `val f1_macro`, `val_per_class.csv`, & **ECE** |
| **H4** | Citra asli seragam `256×256` px; default resize `128×128` membuang 75% piksel | Resolusi `224×224` menurunkan kekeliruan antar-penyakit Tomat bercak halus | `img_height/width`: `128` vs `224` | `val f1_macro` & **Top Confused Pairs** |
| **H5** | Perbedaan gejala antar-penyakit dalam 1 spesies bersifat *fine-grained* | `fine_tune_at` + `adamw` (`lr=1e-4`) atau perbandingan `efficientnetb0`/`resnet50` menaikkan `val f1_macro` dengan *trade-off* ukuran/latensi UI | `model.arch` & `fine_tune_at` | `val f1_macro` 3 seed (`Mean ± SD`), ukuran `.keras`, & latensi CPU |

Detail lengkap tersedia di `docs/DATA_AUDIT_AND_EDA.md` dan matriks perintah Kaggle di `EXPERIMENT_PROTOCOL.md`.
"""))

    return cells


def main():
    nb = nbf.v4.new_notebook(cells=build_cells())
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "mimetype": "text/x-python",
            "file_extension": ".py",
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)

    n_md = sum(1 for c in nb.cells if c.cell_type == "markdown")
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"Notebook written: {OUT_PATH} ({len(nb.cells)} cells: {n_md} markdown, {n_code} code)")


if __name__ == "__main__":
    main()
