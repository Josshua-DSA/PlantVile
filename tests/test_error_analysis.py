"""Deterministic unit tests for error analysis, config validation, and dataset audit."""
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
    cfg = cfgmod.default_config()
    cfg["data"]["root"] = "."  # avoid missing dir noise? no, '.' has no class folders
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
