"""Local CPU end-to-end smoke test (offline, temporary directory, zero workspace residue).

Verifies:
  1. Synthetic dataset creation + nested non-image folder tolerance
  2. scripts/audit_dataset.py JSON report
  3. main.py --check on simple_cnn and mobilenetv2 (weights=none)
  4. 1-epoch mini training via main.py --yes (eval_split=val) + all error analysis & Grad-CAM artifacts
  5. main.py --eval reload on best.keras with config_used.yaml
  6. src.gradcam.explain_image on a nested frozen MobileNetV2 model

Run:
    python scripts/smoke_test.py
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> None:
    print("\n[CMD]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _create_smoke_dataset(data_dir: Path) -> None:
    rng = np.random.default_rng(42)
    classes = ["Tomato_healthy", "Tomato_Late_blight", "Potato___healthy"]
    for idx, cls in enumerate(classes):
        cdir = data_dir / cls
        cdir.mkdir(parents=True, exist_ok=True)
        base_color = np.array([40 + idx * 60, 120, 60], dtype=np.int16)
        for i in range(12):
            noise = rng.integers(-25, 26, size=(64, 64, 3), dtype=np.int16)
            arr = np.clip(base_color + noise, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(cdir / f"img_{i:03d}.jpg", quality=90)
    # Simulate nested non-image folder (like Kaggle's double-nested PlantVillage/)
    (data_dir / "PlantVillage").mkdir(parents=True, exist_ok=True)


def _assert_files(base_dir: Path, filenames: list[str]) -> None:
    for name in filenames:
        fpath = base_dir / name
        assert fpath.is_file() and fpath.stat().st_size > 0, f"Missing or empty artifact: {fpath}"


def main() -> int:
    sys.path.insert(0, str(ROOT))
    py = sys.executable

    with tempfile.TemporaryDirectory(prefix="plantvile_smoke_") as tmp:
        tmp_path = Path(tmp)
        data_dir = tmp_path / "data"
        runs_dir = tmp_path / "runs"
        audit_json = tmp_path / "audit.json"

        _create_smoke_dataset(data_dir)

        # 1. Dataset audit
        _run([
            py, "scripts/audit_dataset.py",
            "--root", str(data_dir), "--out", str(audit_json),
        ])
        audit = json.loads(audit_json.read_text(encoding="utf-8"))
        assert audit["classes"] == 3, audit
        assert audit["valid_images"] == 36, audit
        assert audit["pil_verification"]["corrupt_count"] == 0, audit
        assert "PlantVillage" in audit["skipped_subdirs"], audit

        # 2. --check on simple_cnn and mobilenetv2 (offline)
        _run([
            py, "main.py", "--check", "--config", "configs/default.yaml",
            "--set", f"data.root={data_dir}",
            "--set", "model.arch=simple_cnn",
            "--set", "data.img_height=64", "--set", "data.img_width=64",
            "--set", "data.normalize=div255", "--set", "runtime.device=cpu",
        ])
        _run([
            py, "main.py", "--check", "--config", "configs/default.yaml",
            "--set", f"data.root={data_dir}",
            "--set", "model.arch=mobilenetv2", "--set", "model.weights=none",
            "--set", "data.img_height=64", "--set", "data.img_width=64",
            "--set", "runtime.device=cpu",
        ])

        # 3. 1-epoch mini training (with batched augment + memory cache) + val evaluation + Grad-CAM
        run_name = "smoke_run"
        _run([
            py, "main.py", "--yes", "--config", "configs/default.yaml",
            "--set", f"data.root={data_dir}",
            "--set", f"runtime.runs_dir={runs_dir}",
            "--set", f"runtime.run_name={run_name}",
            "--set", "runtime.device=cpu",
            "--set", "model.arch=simple_cnn",
            "--set", "model.conv_blocks=2", "--set", "model.filters_base=8",
            "--set", "model.dense_units=16",
            "--set", "data.img_height=64", "--set", "data.img_width=64",
            "--set", "data.augment=true", "--set", "data.cache=memory",
            "--set", "data.prefetch=0", "--set", "data.normalize=div255",
            "--set", "data.batch_size=8", "--set", "eval.eval_batch_size=8",
            "--set", "train.epochs=1", "--set", "eval.eval_split=val",
            "--set", "eval.gradcam_samples=4", "--set", "eval.misclassified_samples=4",
        ])

        run_dir = runs_dir / run_name
        _assert_files(run_dir, [
            "best.keras", "final.keras", "history.csv", "history.json",
            "config_used.yaml", "config_used.json", "class_names.json", "device.json",
            "val_metrics.json", "val_per_class.json", "val_per_class.csv",
            "val_classification_report.txt", "val_confusion_matrix.png",
            "val_loss_accuracy.png", "val_sample_predictions.png",
            "val_confused_pairs.json", "val_confused_pairs.csv",
            "val_misclassified.csv", "val_confidence_calibration.png",
            "val_gradcam_grid.png",
        ])
        val_metrics = json.loads((run_dir / "val_metrics.json").read_text(encoding="utf-8"))
        assert "top_k_accuracy" in val_metrics
        assert "ece" in val_metrics["calibration"]
        assert "gradcam" in val_metrics["plots"]

        # 4. --eval reload on test split
        _run([
            py, "main.py",
            "--eval", str(run_dir / "best.keras"),
            "--config", str(run_dir / "config_used.yaml"),
            "--set", "eval.eval_split=test",
            "--set", "runtime.device=cpu",
        ])
        _assert_files(run_dir, ["reload_metrics.json", "reload_gradcam_grid.png"])

        # 5. Direct Grad-CAM test on nested frozen MobileNetV2 (offline)
        from src import config as cfgmod
        from src.gradcam import explain_image
        from src.model import build_model
        from src.utils import configure_device

        configure_device("cpu")
        cfg = cfgmod.from_yaml("configs/default.yaml")
        cfg = cfgmod.apply_overrides(cfg, [
            "model.arch=mobilenetv2", "model.weights=none",
            "model.freeze_base=true", "data.img_height=64", "data.img_width=64",
        ])
        mn_model = build_model(cfg, num_classes=3)
        sample_img = next((data_dir / "Tomato_healthy").glob("*.jpg"))
        info = explain_image(
            mn_model, sample_img, cfg,
            ["Potato___healthy", "Tomato_Late_blight", "Tomato_healthy"],
        )
        assert info["heatmap"].ndim == 2
        assert 0.0 <= float(info["heatmap"].min()) <= float(info["heatmap"].max()) <= 1.0
        assert info["overlay_rgb"].shape == (64, 64, 3)
        assert info["overlay_rgb"].dtype == np.uint8

        # 6. Verify mixed_float16 output dtype='float32' & Grad-CAM cast compatibility
        import keras

        try:
            keras.mixed_precision.set_global_policy("mixed_float16")
            fp16_cfg = cfgmod.apply_overrides(cfgmod.from_yaml("configs/default.yaml"), [
                "model.arch=simple_cnn", "model.conv_blocks=2", "model.filters_base=8",
                "model.dense_units=16", "data.img_height=64", "data.img_width=64",
                "data.normalize=div255", "train.mixed_precision=mixed_float16",
            ])
            fp16_model = build_model(fp16_cfg, num_classes=3)
            assert fp16_model.layers[-1].dtype == "float32", fp16_model.layers[-1].dtype
            fp16_info = explain_image(
                fp16_model, sample_img, fp16_cfg,
                ["Potato___healthy", "Tomato_Late_blight", "Tomato_healthy"],
            )
            assert fp16_info["heatmap"].ndim == 2
            assert fp16_info["overlay_rgb"].shape == (64, 64, 3)
        finally:
            keras.mixed_precision.set_global_policy("float32")

    print("\nSMOKE TEST PASSED: seluruh alur lokal -> audit -> check -> train -> error analysis -> Grad-CAM -> reload -> mixed_float16 OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
