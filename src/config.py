"""Config: one declarative SPEC drives defaults, YAML, CLI overrides, the interactive
wizard, validation and --list-params. Adding a parameter = one line in SPEC."""
import copy
import os

import yaml

# kind: str | int | float | bool | choice | list | path
# fmt:  "dotted" names are what you pass to --set, e.g. --set model.arch=vgg16
SPEC = {
    "data": [
        ("root", "path", "data", None,
         "folder induk dataset (flat: berisi folder kelas, atau berisi color/)"),
        ("variant", "choice", "color", "color|grayscale|segmented",
         "subfolder citra yang dipakai"),
        ("img_height", "int", 128, None, "tinggi resize (px)"),
        ("img_width", "int", 128, None, "lebar resize (px)"),
        ("channels", "int", 3, None, "1=grayscale, 3=RGB"),
        ("class_subset", "list", [], None,
         "kelas yang dipakai; kosong = semua kelas"),
        ("max_per_class", "int", 0, None, "batas gambar per kelas, 0=semua"),
        ("val_split", "float", 0.15, None, "proporsi data validasi"),
        ("test_split", "float", 0.15, None, "proporsi data uji"),
        ("split_strategy", "choice", "stratified", "stratified|per_class_tail",
         "stratified=acak per kelas, per_class_tail=ambil ekor tiap kelas"),
        ("shuffle_seed", "int", 42, None, "seed urutan file sebelum split"),
        ("deduplicate", "bool", True, None,
         "buang duplikat file persis (MD5) in-memory sebelum split"),
        ("group_same_leaf", "bool", True, None,
         "kelompokkan foto daun fisik sama (Leaf X.Y) agar tidak terpecah antar split"),
        ("batch_size", "int", 32, None, "ukuran batch training"),
        ("cache", "choice", "none", "none|memory|disk",
         "cache tf.data (memory ~2.6GB utk 54k citra 128px, disk pakai space)"),
        ("prefetch", "int", 0, None, "jumlah batch di-prefetch (0=AUTOTUNE)"),
        ("augment", "bool", True, None, "aktifkan augmentasi citra"),
        ("aug_rotation", "float", 0.04, None, "rotasi, fraksi 2pi (0.04 = ~14 derajat)"),
        ("aug_width_shift", "float", 0.10, None, "geser horizontal, fraksi lebar"),
        ("aug_height_shift", "float", 0.10, None, "geser vertikal, fraksi tinggi"),
        ("aug_zoom", "float", 0.10, None, "zoom, fraksi"),
        ("aug_shear", "float", 0.0, None, "shear, fraksi"),
        ("aug_hflip", "bool", True, None, "flip horizontal"),
        ("aug_vflip", "bool", False, None, "flip vertikal"),
        ("aug_brightness", "float", 0.15, None, "variasi brightness, fraksi (0=off)"),
        ("aug_contrast", "float", 0.15, None, "variasi contrast, fraksi (0=off)"),
        ("aug_fill_mode", "choice", "reflect",
         "constant|reflect|wrap|nearest", "pengisian piksel saat transform"),
        ("rescale", "bool", True, None, "skala piksel 0-255 -> 0-1"),
        ("normalize", "choice", "imagenet", "none|div255|imagenet",
         "normalisasi; imagenet cocok untuk backbone pretrained"),
    ],
    "model": [
        ("arch", "choice", "mobilenetv2",
         "simple_cnn|vgg16|mobilenetv2|resnet50|efficientnetb0",
         "arsitektur CNN"),
        ("include_top", "bool", False, None, "pakai classifier ImageNet bawaan"),
        ("weights", "choice", "imagenet", "imagenet|none",
         "bobot awal backbone pretrained"),
        ("freeze_base", "bool", True, None, "bekukan backbone saat training"),
        ("fine_tune_at", "int", 0, None,
         "buka layer >= index ini (0=off, hanya berlaku bila freeze_base=true)"),
        ("dropout", "float", 0.3, None, "dropout sebelum layer output"),
        ("dense_units", "int", 256, None, "unit hidden layer classifier (0=tanpa hidden)"),
        ("batch_norm", "bool", True, None, "BatchNorm di classifier (simple_cnn: di tiap blok)"),
        ("l2_weight_decay", "float", 0.0, None, "L2 regularisasi Conv/Dense (0=off)"),
        ("conv_blocks", "int", 4, None, "jumlah blok konvolusi (khusus simple_cnn)"),
        ("filters_base", "int", 32, None, "filter blok pertama (khusus simple_cnn)"),
        ("kernel_size", "int", 3, None, "ukuran kernel (khusus simple_cnn)"),
        ("pool_size", "int", 2, None, "ukuran max-pooling (khusus simple_cnn)"),
    ],
    "train": [
        ("optimizer", "choice", "adam", "adam|adamw|sgd|rmsprop|nadam", "optimizer"),
        ("learning_rate", "float", 0.001, None, "learning rate awal"),
        ("momentum", "float", 0.9, None, "momentum (dipakai optimizer sgd)"),
        ("beta_1", "float", 0.9, None, "beta_1 (adam/adamw/nadam)"),
        ("beta_2", "float", 0.999, None, "beta_2 (adam/adamw/nadam)"),
        ("epsilon", "float", 1e-07, None, "epsilon optimizer"),
        ("loss", "choice", "categorical_crossentropy",
         "categorical_crossentropy|sparse_categorical_crossentropy|focal",
         "fungsi loss"),
        ("label_smoothing", "float", 0.0, None, "label smoothing (0=off)"),
        ("epochs", "int", 20, None, "jumlah epoch"),
        ("class_weight", "choice", "balanced", "none|balanced|manual",
         "bobot kelas; manual lewat train.class_weight_manual"),
        ("class_weight_manual", "list", [], None,
         "bobot manual per kelas (urut label), dipakai bila class_weight=manual"),
        ("early_stopping", "bool", True, None, "aktifkan early stopping"),
        ("es_monitor", "choice", "val_loss", "val_loss|val_accuracy",
         "metrik dipantau early stopping"),
        ("es_patience", "int", 5, None, "epoch tanpa perbaikan sebelum stop"),
        ("es_min_delta", "float", 0.0001, None, "ambang perbaikan minimum"),
        ("es_mode", "choice", "auto", "auto|min|max", "arah perbaikan metrik"),
        ("es_restore_best", "bool", True, None, "kembalikan bobot terbaik"),
        ("reduce_lr", "bool", True, None, "aktifkan penurunan learning rate"),
        ("lr_monitor", "choice", "val_loss", "val_loss|val_accuracy",
         "metrik dipantau reduce-lr"),
        ("lr_factor", "float", 0.5, None, "faktor pengali LR saat turun"),
        ("lr_patience", "int", 3, None, "epoch tanpa perbaikan sebelum LR turun"),
        ("lr_min_lr", "float", 1e-06, None, "batas bawah learning rate"),
        ("lr_cooldown", "int", 0, None, "epoch tunggu setelah LR turun"),
        ("checkpoint_monitor", "choice", "val_accuracy",
         "val_loss|val_accuracy", "metrik penentu checkpoint terbaik"),
        ("save_best_only", "bool", True, None, "simpan hanya model terbaik"),
        ("mixed_precision", "choice", "off", "off|mixed_float16",
         "presisi campuran (butuh GPU untuk manfaat nyata)"),
        ("steps_per_epoch", "int", 0, None, "batch per epoch, 0=otomatis"),
        ("validation_steps", "int", 0, None, "batch validasi per epoch, 0=otomatis"),
    ],
    "runtime": [
        ("device", "choice", "auto", "cpu|gpu|auto", "perangkat komputasi"),
        ("run_name", "str", "", None, "nama folder run (kosong=timestamp)"),
        ("runs_dir", "str", "runs", None, "folder induk output run"),
        ("seed", "int", 42, None, "seed global"),
        ("verbose", "int", 1, None, "0=silent, 1=progress bar, 2=satu baris per epoch"),
        ("log_csv", "bool", True, None, "tulis riwayat epoch ke CSV"),
        ("tensorboard", "bool", False, None, "tulis log TensorBoard"),
        ("resume_from", "str", "", None, "path .keras untuk lanjut training (kosong=off)"),
        ("num_parallel_calls", "int", 4, None,
         "paralelisme tf.data (0=AUTOTUNE, 4=aman CPU lokal)"),
        ("deterministic", "bool", False, None, "op deterministik (lebih lambat)"),
        ("strategy", "choice", "auto", "auto|none|mirrored",
         "distribusi multi-GPU: auto=MirroredStrategy jika >=2 GPU, none=1 device"),
    ],
    "eval": [
        ("eval_batch_size", "int", 32, None, "batch size saat evaluasi"),
        ("top_k", "int", 3, None, "jumlah prediksi teratas ditampilkan"),
        ("confusion_matrix", "bool", True, None, "gambar confusion matrix"),
        ("classification_report", "bool", True, None, "cetak precision/recall/f1"),
        ("per_class_metrics", "bool", True, None, "simpan metrik per kelas (JSON+CSV)"),
        ("sample_predictions", "int", 16, None, "jumlah gambar grid prediksi (0=off)"),
        ("save_plots", "bool", True, None, "simpan semua grafik PNG"),
        ("eval_split", "choice", "test", "val|test|train", "split yang dievaluasi"),
        ("error_analysis", "bool", True, None,
         "jalankan error analysis (confused pairs, misclassified, kalibrasi)"),
        ("misclassified_samples", "int", 16, None,
         "maks gambar salah prediksi di grid (0=off)"),
        ("top_confused_pairs", "int", 20, None,
         "maks pasangan kelas tertukar di laporan"),
        ("gradcam", "bool", True, None, "buat visualisasi Grad-CAM saat evaluasi"),
        ("gradcam_samples", "int", 8, None,
         "jumlah sampel pada grid Grad-CAM (0=off)"),
        ("gradcam_alpha", "float", 0.4, None,
         "opasitas overlay heatmap Grad-CAM (0..1]"),
    ],
}

CHOICE_KINDS = ("choice",)


def default_config() -> dict:
    """Fresh nested dict of every parameter at its default."""
    config = {}
    for section, params in SPEC.items():
        config[section] = {name: copy.deepcopy(default) for name, _, default, _, _ in params}
    return config


def _spec_map() -> dict:
    return {
        section: {p[0]: p for p in params}
        for section, params in SPEC.items()
    }


def coerce(section: str, name: str, raw):
    """Turn a string (from YAML or CLI) into the parameter's real type."""
    spec = _spec_map().get(section, {}).get(name)
    if spec is None:
        raise KeyError("parameter tidak dikenal: %s.%s" % (section, name))
    kind, choices = spec[1], spec[3]

    if kind == "bool":
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in ("1", "true", "yes", "y", "on", "ya"):
            return True
        if text in ("0", "false", "no", "n", "off", "tidak"):
            return False
        raise ValueError("%s.%s harus bool (true/false), dapat: %r" % (section, name, raw))
    if kind == "int":
        return int(str(raw).strip())
    if kind == "float":
        return float(str(raw).strip())
    if kind == "list":
        if isinstance(raw, (list, tuple)):
            return list(raw)
        text = str(raw).strip()
        if not text:
            return []
        return [item.strip() for item in text.split(",") if item.strip()]
    if kind == "choice":
        if isinstance(raw, bool):
            # YAML 1.1 parses unquoted off/on as booleans
            raw = "on" if raw else "off"
        text = str(raw).strip().lower()
        allowed = [c.strip() for c in choices.split("|")]
        if text not in allowed:
            raise ValueError("%s.%s harus salah satu dari %s, dapat: %r"
                             % (section, name, allowed, raw))
        return text
    return str(raw)


def from_yaml(path: str) -> dict:
    """Defaults, then whatever the YAML file sets (unknown keys are rejected)."""
    config = default_config()
    if not path:
        return config
    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    for section, values in loaded.items():
        if section not in config:
            raise KeyError("seksi config tidak dikenal: %r" % section)
        for name, raw in (values or {}).items():
            config[section][name] = coerce(section, name, raw)
    return config


def apply_overrides(config: dict, overrides) -> dict:
    """Apply ['model.arch=vgg16', ...] on top of an existing config."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError("override harus berformat seksi.nama=nilai, dapat: %r" % item)
        dotted, raw = item.split("=", 1)
        if "." not in dotted:
            raise ValueError("override butuh 'seksi.nama', dapat: %r" % dotted)
        section, name = dotted.split(".", 1)
        if section not in config:
            raise KeyError("seksi tidak dikenal: %r" % section)
        config[section][name] = coerce(section, name, raw)
    return config


def validate(config: dict) -> list:
    """Return a list of problem strings. Empty list = config is usable."""
    problems = []
    data, model, train, runtime, ev = (
        config["data"], config["model"], config["train"], config["runtime"], config["eval"]
    )

    root = data["root"]
    if not os.path.isdir(root):
        problems.append("data.root tidak ditemukan: %s" % os.path.abspath(root))
    else:
        variant_dir = os.path.join(root, data["variant"])
        flat = any(
            os.path.isdir(os.path.join(root, d))
            and any(f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                    for f in os.listdir(os.path.join(root, d)))
            for d in os.listdir(root)
        )
        if not os.path.isdir(variant_dir) and not flat:
            problems.append(
                "subfolder variant '%s' tidak ada di %s, dan folder itu juga tidak "
                "berisi folder kelas langsung" % (data["variant"], os.path.abspath(root)))

    if data["img_height"] < 16 or data["img_width"] < 16:
        problems.append("img_height/img_width minimal 16")
    if data["channels"] not in (1, 3):
        problems.append("data.channels harus 1 atau 3")
    if model["arch"] != "simple_cnn":
        if data["img_height"] % 32 or data["img_width"] % 32:
            problems.append("transfer learning butuh img_height & img_width kelipatan 32")
        if model["weights"] == "imagenet" and data["channels"] != 3:
            problems.append("weights=imagenet butuh data.channels=3")
    if data["val_split"] < 0 or data["test_split"] < 0:
        problems.append("val_split/test_split tidak boleh negatif")
    if data["val_split"] + data["test_split"] >= 1.0:
        problems.append("val_split + test_split harus < 1.0")
    if data["batch_size"] < 1:
        problems.append("data.batch_size minimal 1")
    if data["max_per_class"] < 0:
        problems.append("data.max_per_class tidak boleh negatif")
    if data["prefetch"] < 0:
        problems.append("data.prefetch tidak boleh negatif")

    if train["learning_rate"] <= 0:
        problems.append("train.learning_rate harus > 0")
    if train["epochs"] < 1:
        problems.append("train.epochs minimal 1")
    if not 0.0 <= train["label_smoothing"] < 1.0:
        problems.append("train.label_smoothing harus di [0, 1)")
    if not 0.0 <= model["dropout"] < 1.0:
        problems.append("model.dropout harus di [0, 1)")
    if model["conv_blocks"] < 1:
        problems.append("model.conv_blocks minimal 1")
    if train["class_weight"] == "manual" and not train["class_weight_manual"]:
        problems.append("class_weight=manual tapi class_weight_manual kosong")

    if runtime["device"] == "gpu":
        try:
            import tensorflow as tf
            if not tf.config.list_physical_devices("GPU"):
                problems.append(
                    "runtime.device=gpu tapi TensorFlow tidak melihat GPU "
                    "(pakai cpu atau pasang wheel CUDA)")
        except Exception as exc:  # pragma: no cover
            problems.append("gagal cek GPU: %s" % exc)

    if runtime["num_parallel_calls"] < 0:
        problems.append("runtime.num_parallel_calls tidak boleh negatif")
    if ev["top_k"] < 1:
        problems.append("eval.top_k minimal 1")
    if ev["eval_batch_size"] < 1:
        problems.append("eval.eval_batch_size minimal 1")
    if ev["misclassified_samples"] < 0:
        problems.append("eval.misclassified_samples tidak boleh negatif")
    if ev["top_confused_pairs"] < 0:
        problems.append("eval.top_confused_pairs tidak boleh negatif")
    if ev["gradcam_samples"] < 0:
        problems.append("eval.gradcam_samples tidak boleh negatif")
    if not 0.0 < ev["gradcam_alpha"] <= 1.0:
        problems.append("eval.gradcam_alpha harus di (0, 1]")

    return problems


def list_params() -> str:
    """Human-readable dump of every parameter, used by --list-params."""
    lines = []
    for section, params in SPEC.items():
        lines.append("\n[%s]" % section)
        for name, kind, default, choices, help_text in params:
            allowed = ""
            if kind == "choice":
                allowed = "  {%s}" % choices
            lines.append("  %-24s %-6s default=%-10s %s%s"
                         % ("%s.%s" % (section, name), kind, repr(default),
                            help_text, allowed))
    total = sum(len(p) for p in SPEC.values())
    lines.append("\nTotal parameter: %d" % total)
    return "\n".join(lines)


def interactive_wizard(config: dict) -> dict:
    """Walk every parameter section by section. Enter keeps the default."""
    print("\n=== Wizard konfigurasi ===")
    print("Enter = pakai default. Ketik 'skip' untuk melewati sisa seksi ini.")
    print("Ketik 'q' kapan saja untuk membatalkan.\n")

    for section, params in SPEC.items():
        print("--- %s ---" % section)
        for name, kind, _, choices, help_text in params:
            current = config[section][name]
            hint = ""
            if kind == "choice":
                hint = " {%s}" % choices
            prompt = "  %-22s %s%s [%s]: " % (name, help_text, hint, current)
            try:
                raw = input(prompt)
            except EOFError:
                print("\n(input berakhir, lanjut dengan nilai saat ini)")
                return config
            raw = raw.strip()
            if raw == "q":
                raise KeyboardInterrupt("dibatalkan oleh user")
            if raw == "skip":
                print("  (seksi %s dilewati)" % section)
                break
            if not raw:
                continue
            try:
                config[section][name] = coerce(section, name, raw)
            except (ValueError, KeyError) as exc:
                print("  ! %s -> nilai lama dipertahankan" % exc)
        print()

    return config