"""Dataset: index the PlantVillage folder tree, split it, and build tf.data pipelines."""
import hashlib
import os
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")
AUTOTUNE = tf.data.AUTOTUNE


# --------------------------------------------------------------------------- index
def resolve_variant_dir(root: str, variant: str) -> str:
    """PlantVillage ships either as <root>/{color,grayscale,segmented} or already
    flattened as <root>/<Class>. Accept both, plus one extra nesting level."""
    candidates = [
        os.path.join(root, variant),
        os.path.join(root, "PlantVillage", variant),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    # fallback: root already holds class folders (each with image files)
    if os.path.isdir(root):
        for entry in sorted(os.listdir(root)):
            sub = os.path.join(root, entry)
            if not os.path.isdir(sub):
                continue
            if any(f.endswith(IMG_EXTS) for f in os.listdir(sub)):
                return root

    raise FileNotFoundError(
        "folder '%s' tidak ditemukan di bawah %s (dicoba: %s) dan %s juga tidak "
        "berisi folder kelas" % (variant, os.path.abspath(root),
                                 ", ".join(candidates), os.path.abspath(root))
    )


def _md5_file(item):
    idx, fpath = item
    with open(fpath, "rb") as fh:
        return idx, hashlib.md5(fh.read()).hexdigest()


def _deduplicate_paths(paths: list[str], labels: list[int]) -> tuple[list[str], list[int], int]:
    """Drop subsequent exact MD5 duplicates deterministically without touching disk."""
    by_size = defaultdict(list)
    for idx, fpath in enumerate(paths):
        by_size[os.path.getsize(fpath)].append((idx, fpath))

    candidates = [item for grp in by_size.values() if len(grp) > 1 for item in grp]
    if not candidates:
        return paths, labels, 0

    digests = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        for idx, digest in pool.map(_md5_file, candidates):
            digests[idx] = digest

    seen_hashes = set()
    kept_paths, kept_labels = [], []
    dropped = 0
    for idx, (fpath, lbl) in enumerate(zip(paths, labels)):
        digest = digests.get(idx)
        if digest is not None:
            if digest in seen_hashes:
                dropped += 1
                continue
            seen_hashes.add(digest)
        kept_paths.append(fpath)
        kept_labels.append(lbl)
    return kept_paths, kept_labels, dropped


def scan_dataset(root: str, variant: str, class_subset=None, max_per_class: int = 0,
                 deduplicate: bool = False):
    """Return (paths, labels, class_names) plus a per-class count dict."""
    variant_dir = resolve_variant_dir(root, variant)
    raw_dirs = sorted(
        d for d in os.listdir(variant_dir)
        if os.path.isdir(os.path.join(variant_dir, d))
    )
    all_classes = []
    skipped_dirs = []
    for d in raw_dirs:
        cdir = os.path.join(variant_dir, d)
        has_images = any(
            f.endswith(IMG_EXTS) and os.path.isfile(os.path.join(cdir, f))
            for f in os.listdir(cdir)
        )
        if has_images:
            all_classes.append(d)
        else:
            skipped_dirs.append(d)

    if skipped_dirs:
        print("[WARN] Mengabaikan subfolder tanpa citra langsung: %s"
              % ", ".join(skipped_dirs))
    if not all_classes:
        raise RuntimeError("tidak ada subfolder kelas di %s" % variant_dir)

    if class_subset:
        missing = [c for c in class_subset if c not in all_classes]
        if missing:
            raise ValueError("kelas tidak ada di dataset: %s\nkelas tersedia: %s"
                             % (missing, all_classes))
        class_names = sorted(class_subset)
    else:
        class_names = all_classes

    paths, labels = [], []
    for label, cls in enumerate(class_names):
        cls_dir = os.path.join(variant_dir, cls)
        files = sorted(
            os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
            if f.endswith(IMG_EXTS) and os.path.isfile(os.path.join(cls_dir, f))
        )
        if not files:
            raise RuntimeError("kelas '%s' tidak berisi gambar" % cls)
        paths.extend(files)
        labels.extend([label] * len(files))

    if deduplicate:
        paths, labels, dropped = _deduplicate_paths(paths, labels)
        if dropped:
            print("[INFO] Deduplikasi MD5: membuang %d file duplikat persis" % dropped)

    if max_per_class:
        rng = random.Random(0)  # fixed order before the split seed is applied
        by_label = defaultdict(list)
        for fpath, lbl in zip(paths, labels):
            by_label[lbl].append(fpath)
        capped_paths, capped_labels = [], []
        for label in range(len(class_names)):
            files = by_label[label]
            if len(files) > max_per_class:
                files = sorted(rng.sample(files, max_per_class))
            if not files:
                raise RuntimeError("kelas '%s' tidak berisi gambar" % class_names[label])
            capped_paths.extend(files)
            capped_labels.extend([label] * len(files))
        paths, labels = capped_paths, capped_labels

    return paths, labels, class_names, dict(Counter(labels))


# --------------------------------------------------------------------------- split
def _leaf_group_key(path: str) -> str:
    """Group multi-photo captures of the same physical leaf (e.g. Leaf 259.1/.2)."""
    fname = os.path.basename(path)
    rhs = fname.split("___", 1)[1] if "___" in fname else fname
    match = re.search(r"^(.*?\bLeaf\s+\d+)(?:\.\d+)?\b", rhs, flags=re.I)
    if match:
        return " ".join(match.group(1).lower().split())
    return path


def split_dataset(paths, labels, val_split: float, test_split: float,
                  strategy: str = "stratified", seed: int = 42,
                  group_same_leaf: bool = False):
    """Per-class (optionally group-aware) split. Returns index lists for train/val/test."""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for idx, label in enumerate(labels):
        by_class[label].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for label, indices in sorted(by_class.items()):
        indices = list(indices)
        n = len(indices)
        n_test = int(round(n * test_split))
        n_val = int(round(n * val_split))

        if group_same_leaf:
            grouped = defaultdict(list)
            for idx in indices:
                grouped[_leaf_group_key(paths[idx])].append(idx)
            groups = [grouped[k] for k in sorted(grouped.keys())]
            if strategy == "per_class_tail":
                ordered_groups = list(reversed(groups))
            else:
                rng.shuffle(groups)
                ordered_groups = groups

            test_c, val_c, train_c = [], [], []
            rem = n
            for grp in ordered_groups:
                if len(test_c) < n_test and rem - len(grp) >= 1:
                    test_c.extend(grp)
                elif len(val_c) < n_val and rem - len(grp) >= 1:
                    val_c.extend(grp)
                else:
                    train_c.extend(grp)
                rem -= len(grp)
            test_idx.extend(test_c)
            val_idx.extend(val_c)
            train_idx.extend(train_c)
        elif strategy == "per_class_tail":
            test_slice = indices[n - n_test:] if n_test else []
            val_slice = indices[n - n_test - n_val: n - n_test] if n_val else []
            train_slice = indices[: n - n_test - n_val]
            test_idx.extend(test_slice)
            val_idx.extend(val_slice)
            train_idx.extend(train_slice)
        else:  # stratified
            rng.shuffle(indices)
            test_idx.extend(indices[:n_test])
            val_idx.extend(indices[n_test: n_test + n_val])
            train_idx.extend(indices[n_test + n_val:])

    if not train_idx:
        raise RuntimeError("split menghasilkan data training kosong; "
                           "kecilkan val_split/test_split")
    return train_idx, val_idx, test_idx


# --------------------------------------------------------------------------- tf.data
def _make_decoder(cfg_data):
    height = cfg_data["img_height"]
    width = cfg_data["img_width"]
    channels = cfg_data["channels"]
    rescale = cfg_data["rescale"]

    def decode(path, label):
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=channels, expand_animations=False)
        image = tf.image.convert_image_dtype(image, tf.float32)  # -> [0,1]
        image.set_shape([None, None, channels])
        image = tf.image.resize(image, [height, width], method="bilinear")
        if not rescale:
            image = image * 255.0  # keep raw pixel range for backbones that want it
        return image, label

    return decode


def _make_normalizer(cfg_data, arch: str):
    """Pixel scaling on top of the decoder. Values are [0,1] unless rescale=false."""
    mode = cfg_data["normalize"]
    channels = cfg_data["channels"]
    scaled = cfg_data["rescale"]

    if mode == "none":
        return lambda image: image
    if mode == "div255":
        return lambda image: image  # decoder already produced [0,1] when rescale=true
    if mode == "imagenet" and arch != "simple_cnn" and channels == 3:
        from keras.applications import (
            efficientnet, mobilenet_v2, resnet, vgg16,
        )
        preprocess = {
            "vgg16": vgg16.preprocess_input,
            "resnet50": resnet.preprocess_input,
            "mobilenetv2": mobilenet_v2.preprocess_input,
            "efficientnetb0": efficientnet.preprocess_input,
        }[arch]
        # preprocess_input expects 0..255 floats; rescale to that range when needed
        if scaled:
            return lambda image: preprocess(image * 255.0)
        return lambda image: preprocess(image)
    return lambda image: image


def preprocess_image(image_or_path, cfg_data, arch: str):
    """Load/resize one image with the exact same normalization as tf.data.

    Returns (display_rgb_uint8, model_input_batch_float32).
    """
    from PIL import Image

    height = cfg_data["img_height"]
    width = cfg_data["img_width"]
    channels = cfg_data["channels"]

    if isinstance(image_or_path, (str, os.PathLike)):
        with Image.open(image_or_path) as im:
            rgb = im.convert("RGB")
            resized_rgb = rgb.resize((width, height), Image.Resampling.BILINEAR)
    else:
        resized_rgb = image_or_path.convert("RGB").resize(
            (width, height), Image.Resampling.BILINEAR
        )

    display_rgb = np.asarray(resized_rgb, dtype=np.uint8)
    if channels == 1:
        gray = resized_rgb.convert("L")
        arr = np.asarray(gray, dtype=np.float32)[..., None]
    else:
        arr = display_rgb.astype(np.float32)

    if cfg_data["rescale"]:
        arr = arr / 255.0

    batch = np.expand_dims(arr, axis=0)
    normalize = _make_normalizer(cfg_data, arch)
    batch = np.asarray(normalize(batch), dtype=np.float32)
    return display_rgb, batch


def _make_augmenter(cfg_data):
    from keras import layers, Sequential

    augmenters = []
    if cfg_data["aug_rotation"] > 0:
        augmenters.append(layers.RandomRotation(cfg_data["aug_rotation"]))
    if cfg_data["aug_width_shift"] > 0 or cfg_data["aug_height_shift"] > 0:
        augmenters.append(layers.RandomTranslation(
            cfg_data["aug_height_shift"], cfg_data["aug_width_shift"],
            fill_mode=cfg_data["aug_fill_mode"]))
    if cfg_data["aug_zoom"] > 0:
        augmenters.append(layers.RandomZoom(cfg_data["aug_zoom"],
                                            fill_mode=cfg_data["aug_fill_mode"]))
    if cfg_data["aug_shear"] > 0:
        shear = cfg_data["aug_shear"]
        augmenters.append(layers.RandomShear(shear, shear,
                                             fill_mode=cfg_data["aug_fill_mode"]))
    if cfg_data["aug_hflip"] and cfg_data["aug_vflip"]:
        augmenters.append(layers.RandomFlip("horizontal_and_vertical"))
    elif cfg_data["aug_hflip"]:
        augmenters.append(layers.RandomFlip("horizontal"))
    elif cfg_data["aug_vflip"]:
        augmenters.append(layers.RandomFlip("vertical"))
    if cfg_data["aug_brightness"] > 0:
        augmenters.append(layers.RandomBrightness(cfg_data["aug_brightness"],
                                                  value_range=(0.0, 1.0)))
    if cfg_data["aug_contrast"] > 0:
        augmenters.append(layers.RandomContrast(cfg_data["aug_contrast"],
                                                value_range=(0.0, 1.0)))
    if not augmenters:
        return None
    return Sequential(augmenters, name="augment")


def build_dataset(paths, labels, cfg_data, cfg_runtime, cfg_model,
                  training: bool = False, batch_size=None, one_hot: bool = False,
                  split_name: str = "split"):
    """tf.data pipeline: decode -> cache -> shuffle(train) -> batch -> augment(train) -> normalize -> prefetch."""
    batch_size = batch_size or cfg_data["batch_size"]
    decode = _make_decoder(cfg_data)
    normalize = _make_normalizer(cfg_data, cfg_model["arch"])
    augment = _make_augmenter(cfg_data) if (training and cfg_data["augment"]) else None
    num_classes = cfg_runtime.get("_num_classes", 0)
    par_calls = (
        AUTOTUNE
        if cfg_runtime["num_parallel_calls"] <= 0
        else cfg_runtime["num_parallel_calls"]
    )
    prefetch_buf = AUTOTUNE if cfg_data["prefetch"] <= 0 else cfg_data["prefetch"]

    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    ds = ds.map(decode, num_parallel_calls=par_calls)

    if cfg_data["cache"] == "memory":
        ds = ds.cache()
    elif cfg_data["cache"] == "disk":
        cache_path = os.path.join(
            cfg_runtime["runs_dir"], ".cache",
            "cache_%s_%d_%d_%s" % (
                split_name, cfg_data["img_height"], cfg_data["img_width"],
                cfg_data["variant"],
            ),
        )
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        ds = ds.cache(cache_path)

    if training:
        ds = ds.shuffle(
            buffer_size=min(len(paths), 10000),
            seed=cfg_runtime["seed"],
            reshuffle_each_iteration=True,
        )

    ds = ds.batch(batch_size, drop_remainder=False)
    # Augment while pixels are still in [0,1] (batched on CPU/GPU) before ImageNet scaling.
    if augment is not None:
        ds = ds.map(
            lambda x, y: (augment(x, training=True), y),
            num_parallel_calls=par_calls,
        )

    ds = ds.map(lambda x, y: (normalize(x), y), num_parallel_calls=par_calls)
    if one_hot and num_classes:
        ds = ds.map(
            lambda x, y: (x, tf.one_hot(y, num_classes)),
            num_parallel_calls=par_calls,
        )
    ds = ds.prefetch(prefetch_buf)
    return ds


def compute_class_weights(labels, num_classes, mode="balanced", manual=None):
    """balanced -> inverse frequency, scaled so the mean weight is ~1."""
    if mode == "none":
        return None
    if mode == "manual":
        if not manual:
            return None
        weights = {i: float(w) for i, w in enumerate(manual) if i < num_classes}
        return weights or None

    counts = Counter(labels)
    total = sum(counts.values())
    present = [counts.get(i, 0) for i in range(num_classes)]
    nonzero = [c for c in present if c > 0]
    if not nonzero:
        return None
    mean_count = total / len(nonzero)
    weights = {}
    for index in range(num_classes):
        count = counts.get(index, 0)
        if count:
            weights[index] = float(mean_count / count)
    return weights


# --------------------------------------------------------------------------- bundle
def prepare_datasets(cfg, verbose: bool = True):
    """Scan + split + build three pipelines. Returns a dict describing everything."""
    data, runtime, model, train_cfg = (
        cfg["data"], cfg["runtime"], cfg["model"], cfg["train"]
    )
    one_hot = train_cfg["loss"] != "sparse_categorical_crossentropy"

    paths, labels, class_names, counts = scan_dataset(
        data["root"], data["variant"], data["class_subset"], data["max_per_class"],
        deduplicate=data["deduplicate"],
    )
    train_idx, val_idx, test_idx = split_dataset(
        paths, labels, data["val_split"], data["test_split"],
        data["split_strategy"], data["shuffle_seed"],
        group_same_leaf=data["group_same_leaf"],
    )

    def subset(indices):
        return [paths[i] for i in indices], [labels[i] for i in indices]

    train_paths, train_labels = subset(train_idx)
    val_paths, val_labels = subset(val_idx)
    test_paths, test_labels = subset(test_idx)

    runtime_ctx = dict(runtime)
    runtime_ctx["_num_classes"] = len(class_names)

    def make(p, l, training, split_name):
        return build_dataset(p, l, data, runtime_ctx, model,
                             training=training, one_hot=one_hot, split_name=split_name)

    train_ds = make(train_paths, train_labels, True, "train")
    val_ds = make(val_paths, val_labels, False, "val")
    test_ds = make(test_paths, test_labels, False, "test")

    bundle = {
        "class_names": class_names,
        "num_classes": len(class_names),
        "counts": counts,
        "paths": paths,
        "labels": labels,
        "train": {"paths": train_paths, "labels": train_labels, "ds": train_ds},
        "val": {"paths": val_paths, "labels": val_labels, "ds": val_ds},
        "test": {"paths": test_paths, "labels": test_labels, "ds": test_ds},
        "class_weights": compute_class_weights(
            train_labels, len(class_names),
            cfg["train"]["class_weight"], cfg["train"]["class_weight_manual"]
        ),
    }

    if verbose:
        print("\nDataset: %s / %s" % (os.path.abspath(data["root"]), data["variant"]))
        print("  kelas        : %d" % len(class_names))
        print("  total gambar : %d" % len(paths))
        print("  train/val/test: %d / %d / %d"
              % (len(train_paths), len(val_paths), len(test_paths)))
        sizes = sorted(counts.values())
        print("  per kelas    : min %d, max %d, median %d"
              % (sizes[0], sizes[-1], sizes[len(sizes) // 2]))
        print("  batch        : %d  -> %d batch train"
              % (data["batch_size"],
                 int(np.ceil(len(train_paths) / data["batch_size"]))))

    return bundle
