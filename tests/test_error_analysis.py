"""Deterministic unit tests for error analysis, config validation, and dataset audit."""
import warnings
import numpy as np
from PIL import Image

from scripts.audit_dataset import audit_directory
from src import config as cfgmod
from src.evaluate import (
    compute_calibration,
    compute_top_k_accuracy,
    extract_confused_pairs,
    extract_misclassified,
)


def test_compute_top_k_accuracy_and_clamping():
    y_true = np.array([0, 1, 2])
    probs = np.array([
        [0.7, 0.2, 0.1],
        [0.6, 0.3, 0.1],  # true=1 is top-2
        [0.6, 0.3, 0.1],  # true=2 is top-3
    ])
    assert abs(compute_top_k_accuracy(y_true, probs, k=1) - (1 / 3)) < 1e-9
    assert abs(compute_top_k_accuracy(y_true, probs, k=2) - (2 / 3)) < 1e-9
    assert abs(compute_top_k_accuracy(y_true, probs, k=10) - 1.0) < 1e-9


def test_compute_calibration_perfect_and_overconfident():
    y_true = np.array([0, 1])
    probs_perfect = np.array([[1.0, 0.0], [0.0, 1.0]])
    cal_perf = compute_calibration(y_true, probs_perfect, n_bins=10)
    assert cal_perf["ece"] == 0.0
    assert cal_perf["mean_conf_wrong"] is None

    probs_bad = np.array([[0.95, 0.05], [0.95, 0.05]])  # second sample wrong at 0.95
    cal_bad = compute_calibration(y_true, probs_bad, n_bins=10)
    assert abs(cal_bad["ece"] - 0.45) < 1e-6
    assert abs(cal_bad["mean_conf_correct"] - 0.95) < 1e-6
    assert abs(cal_bad["mean_conf_wrong"] - 0.95) < 1e-6


def test_extract_confused_pairs_and_zero_errors():
    class_names = ["A", "B", "C"]
    y_true = np.array([0, 0, 1, 1, 2])
    y_hat = np.array([0, 1, 1, 0, 2])
    pairs = extract_confused_pairs(y_true, y_hat, class_names, top_n=10)
    assert len(pairs) == 2
    assert pairs[0]["count"] == 1
    assert pairs[0]["pct_of_true_class"] == 50.0

    assert extract_confused_pairs(y_true, y_true, class_names, top_n=10) == []


def test_extract_misclassified_sorted_by_confidence():
    class_names = ["A", "B", "C"]
    paths = ["p0.jpg", "p1.jpg", "p2.jpg"]
    y_true = np.array([0, 1, 2])
    probs = np.array([
        [0.8, 0.1, 0.1],   # correct
        [0.65, 0.25, 0.1], # wrong, conf=0.65
        [0.92, 0.05, 0.03] # wrong, conf=0.92
    ])
    mis = extract_misclassified(paths, y_true, probs, class_names, top_k=2)
    assert [m["path"] for m in mis] == ["p2.jpg", "p1.jpg"]
    assert mis[0]["true_class"] == "C"
    assert mis[0]["pred_class"] == "A"
    assert abs(mis[0]["true_class_prob"] - 0.03) < 1e-9
    assert len(mis[0]["top_k_preds"]) == 2


def test_config_validate_eval_error_analysis_and_gradcam_bounds():
    bad = cfgmod.apply_overrides(cfgmod.default_config(), [
        "eval.gradcam_alpha=0.0",
        "eval.gradcam_samples=-1",
        "eval.misclassified_samples=-2",
        "eval.top_confused_pairs=-1",
    ])
    problems = cfgmod.validate(bad)
    assert any("gradcam_alpha" in p for p in problems)
    assert any("gradcam_samples" in p for p in problems)
    assert any("misclassified_samples" in p for p in problems)
    assert any("top_confused_pairs" in p for p in problems)


def test_audit_directory_detects_duplicates_non_images_and_nested(tmp_path):
    cls_a = tmp_path / "ClassA"
    cls_b = tmp_path / "ClassB"
    nested = tmp_path / "PlantVillage"
    cls_a.mkdir()
    cls_b.mkdir()
    nested.mkdir()

    arr = np.full((32, 32, 3), 120, dtype=np.uint8)
    img_path_1 = cls_a / "img1.jpg"
    img_path_2 = cls_a / "img2_dup.jpg"
    Image.fromarray(arr).save(img_path_1, quality=95)
    img_path_2.write_bytes(img_path_1.read_bytes())

    arr_b = np.full((32, 32, 3), 40, dtype=np.uint8)
    Image.fromarray(arr_b).save(cls_b / "img_b.jpg", quality=95)
    (cls_b / "svn-r6Yb5c").write_bytes(b"")

    report = audit_directory(str(tmp_path), sample_verify=10, seed=42)
    assert report["classes"] == 2
    assert report["valid_images"] == 3
    assert report["skipped_subdirs"] == ["PlantVillage"]
    assert len(report["non_image_files"]) == 1
    assert report["exact_md5_duplicates"]["groups"] == 1
    assert report["exact_md5_duplicates"]["within_class_groups"] == 1
    assert report["pil_verification"]["corrupt_count"] == 0


def test_kaggle_t4x2_yaml_and_runtime_validation():
    kaggle_cfg = cfgmod.from_yaml("configs/kaggle_t4x2.yaml")
    assert kaggle_cfg["train"]["mixed_precision"] == "mixed_float16"
    assert kaggle_cfg["runtime"]["strategy"] == "auto"
    assert kaggle_cfg["data"]["cache"] == "memory"
    assert kaggle_cfg["data"]["prefetch"] == 0
    assert kaggle_cfg["eval"]["eval_split"] == "val"

    bad = cfgmod.apply_overrides(cfgmod.default_config(), [
        "data.prefetch=-1",
        "runtime.num_parallel_calls=-1",
    ])
    problems = cfgmod.validate(bad)
    assert any("data.prefetch" in p for p in problems)
    assert any("runtime.num_parallel_calls" in p for p in problems)


def test_get_distribution_strategy_cpu_modes():
    from src.utils import configure_device, get_distribution_strategy

    configure_device("cpu")
    strat_auto, desc_auto = get_distribution_strategy("auto")
    strat_none, desc_none = get_distribution_strategy("none")
    strat_mir, desc_mir = get_distribution_strategy("mirrored")
    assert strat_auto.num_replicas_in_sync == 1
    assert strat_none.num_replicas_in_sync == 1
    assert strat_mir.num_replicas_in_sync == 1
    assert "single device" in desc_auto
    assert "single device" in desc_none
    assert "single device" in desc_mir


def test_scan_dataset_deduplicate_md5(tmp_path):
    from src.data import scan_dataset

    c1 = tmp_path / "ClassA"
    c2 = tmp_path / "ClassB"
    c1.mkdir()
    c2.mkdir()

    arr_a = np.full((32, 32, 3), 100, dtype=np.uint8)
    p1 = c1 / "a1.jpg"
    p2 = c1 / "a2_dup.jpg"
    Image.fromarray(arr_a).save(p1, quality=95)
    p2.write_bytes(p1.read_bytes())

    arr_b = np.full((32, 32, 3), 200, dtype=np.uint8)
    Image.fromarray(arr_b).save(c2 / "b1.jpg", quality=95)

    p_raw, _, _, _ = scan_dataset(str(tmp_path), "color", deduplicate=False)
    p_dedup, _, _, counts_dedup = scan_dataset(str(tmp_path), "color", deduplicate=True)
    assert len(p_raw) == 3
    assert len(p_dedup) == 2
    assert counts_dedup == {0: 1, 1: 1}


def test_split_dataset_group_same_leaf_no_leakage():
    from src.data import _leaf_group_key, split_dataset

    paths = []
    labels = []
    # Class 0: 4 groups of 3 photos each (12 images)
    for g in range(1, 5):
        for sub in range(1, 4):
            paths.append(f"/tmp/Tomato_healthy/uuid_{g}_{sub}___GH_HL Leaf {g}.{sub}.JPG")
            labels.append(0)
    # Class 1: 10 singleton images
    for i in range(10):
        paths.append(f"/tmp/Potato_healthy/uuid_{i}___RS_HL {i}.JPG")
        labels.append(1)

    tr, va, te = split_dataset(
        paths, labels, val_split=0.25, test_split=0.25,
        strategy="stratified", seed=42, group_same_leaf=True,
    )
    assert set(tr).isdisjoint(set(va))
    assert set(tr).isdisjoint(set(te))
    assert set(va).isdisjoint(set(te))
    assert len(tr) + len(va) + len(te) == len(paths)

    g_tr = {(labels[i], _leaf_group_key(paths[i])) for i in tr}
    g_va = {(labels[i], _leaf_group_key(paths[i])) for i in va}
    g_te = {(labels[i], _leaf_group_key(paths[i])) for i in te}
    assert not (g_tr & g_va)
    assert not (g_tr & g_te)
    assert not (g_va & g_te)


def test_all_five_optimizers_build_cleanly():
    from src.model import _build_optimizer

    for opt_name in ("adam", "adamw", "sgd", "rmsprop", "nadam"):
        cfg = cfgmod.apply_overrides(cfgmod.default_config(), [f"train.optimizer={opt_name}"])
        opt = _build_optimizer(cfg["train"])
        assert opt is not None


def test_mobilenetv3_large_build_and_gradcam(tmp_path):
    from src.gradcam import explain_image
    from src.model import build_model

    arr = np.full((64, 64, 3), 128, dtype=np.uint8)
    img_path = tmp_path / "leaf.jpg"
    Image.fromarray(arr).save(img_path, quality=95)

    cfg = cfgmod.apply_overrides(cfgmod.default_config(), [
        "model.arch=mobilenetv3_large",
        "model.weights=none",
        "model.freeze_base=true",
        "model.fine_tune_at=124",
        "data.img_height=64",
        "data.img_width=64",
    ])
    model = build_model(cfg, num_classes=3)
    assert model.name == "mobilenetv3_large"
    info = explain_image(model, img_path, cfg, ["A", "B", "C"])
    assert info["heatmap"].ndim == 2
    assert info["overlay_rgb"].shape == (64, 64, 3)


def test_config_validate_dangerous_combinations_and_warnings():
    bad_norm = cfgmod.apply_overrides(cfgmod.default_config(), [
        "model.arch=mobilenetv2",
        "model.weights=imagenet",
        "data.normalize=div255",
    ])
    probs_norm = cfgmod.validate(bad_norm)
    assert any("div255" in p for p in probs_norm)

    bad_aug = cfgmod.apply_overrides(cfgmod.default_config(), [
        "data.rescale=false",
        "data.augment=true",
        "data.aug_brightness=0.15",
    ])
    probs_aug = cfgmod.validate(bad_aug)
    assert any("rescale=false" in p for p in probs_aug)

    warn_cfg = cfgmod.apply_overrides(cfgmod.default_config(), [
        "train.loss=focal",
        "train.class_weight=balanced",
        "train.label_smoothing=0.05",
    ])
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        cfgmod.validate(warn_cfg)
        assert any(issubclass(w.category, UserWarning) for w in recorded)


def test_preprocess_image_matches_tf_data_decoder(tmp_path):
    from src.data import _make_decoder, _make_normalizer, preprocess_image

    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, size=(96, 80, 3), dtype=np.uint8)
    img_path = tmp_path / "sample.png"
    Image.fromarray(arr).save(img_path)

    cfg = cfgmod.apply_overrides(cfgmod.default_config(), [
        "data.img_height=64",
        "data.img_width=64",
        "data.rescale=true",
        "data.normalize=imagenet",
        "model.arch=mobilenetv2",
    ])
    _, batch_np = preprocess_image(img_path, cfg["data"], cfg["model"]["arch"])
    decode = _make_decoder(cfg["data"])
    norm = _make_normalizer(cfg["data"], cfg["model"]["arch"])
    decoded, _ = decode(str(img_path), 0)
    tf_batch = norm(decoded[None, ...]).numpy()
    assert float(np.max(np.abs(batch_np - tf_batch))) < 1e-5


def test_disk_cache_key_sensitive_to_seed_and_resize():
    from src.data import _disk_cache_key

    paths = ["/tmp/a.jpg", "/tmp/b.jpg"]
    labels = [0, 1]
    base = cfgmod.default_config()["data"]
    diff_seed = dict(base, shuffle_seed=99)
    diff_size = dict(base, img_height=224, img_width=224)

    k1 = _disk_cache_key(paths, labels, base, "train")
    k2 = _disk_cache_key(paths, labels, diff_seed, "train")
    k3 = _disk_cache_key(paths, labels, diff_size, "train")
    assert len({k1, k2, k3}) == 3


def test_eval_auto_loads_config_used_yaml(tmp_path):
    from unittest.mock import patch
    import keras
    import main as main_mod
    import yaml

    run_dir = tmp_path / "m_test"
    run_dir.mkdir()
    model = keras.Sequential([
        keras.Input(shape=(64, 64, 3)),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(3, activation="softmax"),
    ])
    ckpt = run_dir / "best.keras"
    model.save(ckpt)

    cfg = cfgmod.apply_overrides(cfgmod.default_config(), [
        "data.img_height=64",
        "data.img_width=64",
        "model.arch=mobilenetv3_large",
        "eval.eval_split=val",
        "runtime.device=cpu",
    ])
    with open(run_dir / "config_used.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    captured = {}

    def fake_evaluate(m, used_cfg, run_dir=None, tag=None, quiet=False, **kwargs):
        captured["arch"] = used_cfg["model"]["arch"]
        captured["height"] = used_cfg["data"]["img_height"]
        captured["tag"] = tag
        return {}

    import src.evaluate as eval_mod
    with patch.object(eval_mod, "evaluate", side_effect=fake_evaluate):
        rc = main_mod.main(["--eval", str(ckpt), "--quiet"])
    assert rc == 0
    assert captured == {"arch": "mobilenetv3_large", "height": 64, "tag": "val"}


def test_tflite_ptq_fp32_fp16_int8_roundtrip(tmp_path):
    import keras
    import tensorflow as tf
    from scripts.export_tflite import convert_keras_to_tflite

    model = keras.Sequential([
        keras.Input(shape=(32, 32, 3)),
        keras.layers.Conv2D(8, 3, activation="relu"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(3, activation="softmax"),
    ])
    rng = np.random.default_rng(42)
    calib = [rng.uniform(0.0, 1.0, size=(1, 32, 32, 3)).astype(np.float32) for _ in range(8)]

    def rep_fn():
        for batch in calib:
            yield [batch]

    sample = calib[0]
    for q in ("fp32", "fp16", "int8"):
        tbytes, _ = convert_keras_to_tflite(model, q, rep_dataset_fn=rep_fn if q == "int8" else None)
        tpath = tmp_path / f"model_{q}.tflite"
        tpath.write_bytes(tbytes)
        interp = tf.lite.Interpreter(model_path=str(tpath), num_threads=2)
        interp.allocate_tensors()
        in_idx = interp.get_input_details()[0]["index"]
        out_idx = interp.get_output_details()[0]["index"]
        interp.set_tensor(in_idx, sample)
        interp.invoke()
        out = interp.get_tensor(out_idx)
        assert out.shape == (1, 3)
        assert np.isfinite(out).all()


def test_gpu_augmented_model_trains_and_saves_clean_core_model(tmp_path):
    import keras
    import tensorflow as tf
    from src import config as cfgmod
    from src.data import _make_augmenter, _make_normalizer
    from src.model import build_model, compile_model
    from src.train import CoreModelCheckpoint, _GPUAugmentedModel

    prev_policy = keras.mixed_precision.global_policy().name
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        cfg = cfgmod.default_config()
        cfg["data"]["img_height"] = 32
        cfg["data"]["img_width"] = 32
        cfg["data"]["augment"] = True
        cfg["data"]["normalize"] = "imagenet"
        cfg["model"]["arch"] = "simple_cnn"
        cfg["model"]["conv_blocks"] = 2
        cfg["model"]["filters_base"] = 8
        cfg["model"]["dense_units"] = 16
        cfg["model"]["l2_weight_decay"] = 0.001

        core_model = build_model(cfg, num_classes=3)
        aug_layer = _make_augmenter(cfg["data"], dtype="float32")
        assert aug_layer is not None
        norm_fn = _make_normalizer(cfg["data"], cfg["model"]["arch"])
        train_model = _GPUAugmentedModel(core_model, aug_layer, norm_fn)
        compile_model(train_model, cfg["train"])

        rng = np.random.default_rng(7)
        x_train = rng.uniform(0.0, 1.0, size=(8, 32, 32, 3)).astype(np.float32)
        y_train = np.eye(3, dtype=np.float32)[np.arange(8) % 3]
        x_val_raw = rng.uniform(0.0, 1.0, size=(4, 32, 32, 3)).astype(np.float32)
        x_val_norm = norm_fn(tf.constant(x_val_raw)).numpy()
        y_val = np.eye(3, dtype=np.float32)[np.arange(4) % 3]

        _ = train_model(tf.constant(x_train[:2]), training=True)
        assert len(train_model.trainable_weights) == len(core_model.trainable_weights)
        assert len(train_model.losses) == len(core_model.losses) > 0

        ckpt_path = tmp_path / "best.keras"
        ckpt_cb = CoreModelCheckpoint(
            target_model=core_model,
            filepath=str(ckpt_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        )
        train_model.fit(
            x_train,
            y_train,
            validation_data=(x_val_norm, y_val),
            batch_size=4,
            epochs=1,
            callbacks=[ckpt_cb],
            verbose=0,
        )

        assert ckpt_path.is_file()
        loaded = keras.models.load_model(ckpt_path, compile=False)
        layer_names = [layer.name for layer in loaded.layers]
        assert "augment" not in layer_names
        assert not any(name.startswith("random_") for name in layer_names)
        np.testing.assert_allclose(
            loaded.predict(x_val_norm, verbose=0),
            core_model.predict(x_val_norm, verbose=0),
            rtol=1e-4,
            atol=1e-4,
        )
    finally:
        keras.mixed_precision.set_global_policy(prev_policy)


def test_step_history_logger_throttles_lr_reads(tmp_path):
    import csv
    from src.train import StepHistoryLogger

    csv_path = tmp_path / "step_history.csv"
    logger = StepHistoryLogger(str(csv_path), every=5)
    logger.set_params({"steps": 10})

    calls = {"n": 0}

    def fake_lr() -> float:
        calls["n"] += 1
        return 0.001

    logger._current_lr = fake_lr
    logger.on_train_begin()
    logger.on_epoch_begin(0)
    for batch in range(10):
        logger.on_train_batch_end(batch, logs={"loss": 1.0 - batch * 0.05, "accuracy": 0.5 + batch * 0.03})
    logger.on_epoch_end(0)
    logger.on_train_end()

    assert calls["n"] == 3
    with open(csv_path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [int(r["global_step"]) for r in rows] == [1, 5, 10]
