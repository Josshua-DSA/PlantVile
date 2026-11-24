"""Plot combined Stage 1 -> Stage 2 gradient descent curves, architecture overlays, and step-level EMA curves.

Run from repository root:
    python -m scripts.plot_gradient_descent --runs-dir results/runs --out-dir results/gradient_descent_plots
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARCH_PAIRS: dict[str, tuple[str, str]] = {
    "mobilenetv2": ("m1_mobilenetv2_head", "m2_mobilenetv2_ft"),
    "efficientnetb0": ("m3a_effnetb0_head", "m3b_effnetb0_ft"),
    "mobilenetv3_large": ("m4a_mobilenetv3l_head", "m4b_mobilenetv3l_ft"),
}

ARCH_LABELS: dict[str, str] = {
    "mobilenetv2": "MobileNetV2",
    "efficientnetb0": "EfficientNet-B0",
    "mobilenetv3_large": "MobileNetV3-Large",
}

ARCH_COLORS: dict[str, str] = {
    "mobilenetv2": "#1b9e77",
    "efficientnetb0": "#d95f02",
    "mobilenetv3_large": "#7570b3",
}


def compute_entropy_floor(eps: float, num_classes: int) -> float | None:
    """Compute theoretical cross-entropy floor H(q) for Keras uniform label smoothing."""
    if not (0.0 < eps < 1.0) or num_classes <= 1:
        return None
    p_other = eps / num_classes
    p_main = (1.0 - eps) + p_other
    return float(-(p_main * math.log(p_main) + (num_classes - 1) * p_other * math.log(p_other)))


def compute_ema(values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Compute exponential moving average for mini-batch loss sequences."""
    if len(values) == 0:
        return values
    out = np.empty_like(values, dtype=np.float64)
    out[0] = float(values[0])
    for idx in range(1, len(values)):
        out[idx] = alpha * float(values[idx]) + (1.0 - alpha) * out[idx - 1]
    return out


def _load_history(run_dir: Path) -> dict[str, list[float]]:
    json_path = run_dir / "history.json"
    if json_path.is_file():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return {k: [float(v) for v in vals] for k, vals in data.items() if isinstance(vals, list)}

    csv_path = run_dir / "history.csv"
    if csv_path.is_file():
        cols: dict[str, list[float]] = {}
        with open(csv_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                for k, v in row.items():
                    if k == "epoch":
                        continue
                    cols.setdefault(k, []).append(float(v))
        return cols
    return {}


def _load_meta(run_dir: Path) -> tuple[float, int]:
    """Return (label_smoothing, num_classes) for a run directory."""
    ls = 0.0
    cfg_json = run_dir / "config_used.json"
    if cfg_json.is_file():
        cfg = json.loads(cfg_json.read_text(encoding="utf-8"))
        if cfg.get("train", {}).get("loss") == "categorical_crossentropy":
            ls = float(cfg.get("train", {}).get("label_smoothing", 0.0))

    num_classes = 15
    cls_json = run_dir / "class_names.json"
    if cls_json.is_file():
        num_classes = len(json.loads(cls_json.read_text(encoding="utf-8")))
    return ls, num_classes


def _lr_series(hist: dict[str, list[float]]) -> list[float]:
    if "learning_rate" in hist:
        return hist["learning_rate"]
    if "lr" in hist:
        return hist["lr"]
    return []


def _last_k_stats(val_loss: list[float], k: int = 5) -> tuple[float | None, float | None]:
    if len(val_loss) < 2:
        return None, None
    tail = np.asarray(val_loss[-min(k, len(val_loss)) :], dtype=np.float64)
    x = np.arange(len(tail), dtype=np.float64)
    slope = float(np.polyfit(x, tail, 1)[0])
    diff_std = float(np.std(np.diff(tail), ddof=1)) if len(tail) >= 3 else float(np.std(np.diff(tail)))
    return slope, diff_std


def plot_combined_arch(
    arch: str,
    s1_dir: Path | None,
    s2_dir: Path | None,
    out_path: Path,
) -> dict | None:
    """Plot continuous Stage 1 -> Stage 2 Loss, Accuracy, and LR for one architecture."""
    h1 = _load_history(s1_dir) if (s1_dir and s1_dir.is_dir()) else {}
    h2 = _load_history(s2_dir) if (s2_dir and s2_dir.is_dir()) else {}
    if not h1 and not h2:
        return None

    n1 = len(h1.get("loss", []))
    n2 = len(h2.get("loss", []))
    total_ep = n1 + n2
    epochs = list(range(1, total_ep + 1))

    def concat(key: str) -> list[float]:
        return list(h1.get(key, [])) + list(h2.get(key, []))

    train_loss = concat("loss")
    val_loss = concat("val_loss")
    train_acc = concat("accuracy")
    val_acc = concat("val_accuracy")
    lr_vals = _lr_series(h1) + _lr_series(h2)

    ref_dir = s2_dir if (s2_dir and s2_dir.is_dir()) else s1_dir
    ls, num_classes = _load_meta(ref_dir) if ref_dir else (0.0, 15)
    floor = compute_entropy_floor(ls, num_classes)

    s2_val_loss = h2.get("val_loss", val_loss)
    s2_val_acc = h2.get("val_accuracy", val_acc)
    slope, diff_std = _last_k_stats(s2_val_loss, k=5)

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 10.0), sharex=True)
    title_label = ARCH_LABELS.get(arch, arch)
    fig.suptitle(f"{title_label} — Kurva Gradient Descent Gabungan (Stage 1 → Stage 2)", fontsize=13)

    # Panel 1: Loss
    ax = axes[0]
    ax.plot(epochs, train_loss, color="#1f77b4", linewidth=1.8, label="train_loss")
    if len(val_loss) == total_ep:
        ax.plot(epochs, val_loss, color="#d95f02", linewidth=1.8, label="val_loss")
        best_ep = int(np.argmin(val_loss)) + 1
        ax.scatter([best_ep], [min(val_loss)], color="#b02a37", zorder=5,
                   label=f"best val_loss={min(val_loss):.4f} (ep {best_ep})")
    if floor is not None:
        ax.axhline(floor, color="#555555", linestyle="--", linewidth=1.2,
                   label=f"LS entropy floor H(q)={floor:.4f}")
    if n1 > 0 and n2 > 0:
        ax.axvline(n1 + 0.5, color="#333333", linestyle=":", linewidth=1.4,
                   label="Transisi Stage 2 (unfreeze)")
    if slope is not None and diff_std is not None:
        ax.text(
            0.99, 0.94,
            f"Last-5 ep val_loss slope={slope:+.5f}/ep | std(Δ)={diff_std:.5f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "#f7f7f7", "edgecolor": "#cccccc"},
        )
    ax.set_ylabel("Cross-Entropy Loss")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right" if slope is None else "center right", fontsize=8.5)

    # Panel 2: Accuracy
    ax = axes[1]
    ax.plot(epochs, train_acc, color="#1f77b4", linewidth=1.8, label="train_accuracy")
    if len(val_acc) == total_ep:
        ax.plot(epochs, val_acc, color="#2f7d5d", linewidth=1.8, label="val_accuracy")
        best_ep_a = int(np.argmax(val_acc)) + 1
        ax.scatter([best_ep_a], [max(val_acc)], color="#145a32", zorder=5,
                   label=f"best val_acc={max(val_acc):.4f} (ep {best_ep_a})")
    if n1 > 0 and n2 > 0:
        ax.axvline(n1 + 0.5, color="#333333", linestyle=":", linewidth=1.4)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(bottom=max(0.0, min(train_acc + val_acc) - 0.03), top=1.01)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5)

    # Panel 3: Learning Rate
    ax = axes[2]
    if len(lr_vals) == total_ep:
        ax.plot(epochs, lr_vals, color="#7570b3", marker="o", markersize=3.5, linewidth=1.6, label="learning_rate")
        ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=8.5)
    else:
        ax.text(0.5, 0.5, "Learning rate log tidak tersedia", ha="center", va="center", transform=ax.transAxes)
    if n1 > 0 and n2 > 0:
        ax.axvline(n1 + 0.5, color="#333333", linestyle=":", linewidth=1.4)
    ax.set_ylabel("Learning Rate (log)")
    ax.set_xlabel("Epoch Kontinu (Stage 1 + Stage 2)")
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "arch": arch,
        "stage1_epochs": n1,
        "stage2_epochs": n2,
        "best_val_loss": round(float(min(s2_val_loss)), 6) if s2_val_loss else None,
        "best_val_accuracy": round(float(max(s2_val_acc)), 6) if s2_val_acc else None,
        "entropy_floor": round(floor, 6) if floor is not None else None,
        "last5_val_loss_slope": round(slope, 6) if slope is not None else None,
        "last5_val_loss_diff_std": round(diff_std, 6) if diff_std is not None else None,
        "plot_path": str(out_path),
    }


def plot_stage2_comparison(runs_dir: Path, out_path: Path) -> bool:
    """Overlay Stage 2 train_loss, val_loss, and val_accuracy across available architectures."""
    loaded: list[tuple[str, str, dict[str, list[float]], float | None]] = []
    for arch, (_, s2_id) in ARCH_PAIRS.items():
        s2_dir = runs_dir / s2_id
        if not s2_dir.is_dir():
            continue
        hist = _load_history(s2_dir)
        if not hist.get("loss"):
            continue
        ls, num_cls = _load_meta(s2_dir)
        floor = compute_entropy_floor(ls, num_cls)
        loaded.append((arch, s2_id, hist, floor))

    if not loaded:
        return False

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    fig.suptitle("Perbandingan Kurva Gradient Descent Stage 2 (Fine-Tuning)", fontsize=13)

    floors = [f for _, _, _, f in loaded if f is not None]
    common_floor = floors[0] if floors else None

    for arch, s2_id, hist, _ in loaded:
        label = f"{ARCH_LABELS.get(arch, arch)} ({s2_id})"
        color = ARCH_COLORS.get(arch, None)
        ep = list(range(1, len(hist["loss"]) + 1))
        axes[0].plot(ep, hist["loss"], color=color, linewidth=1.9, label=label)
        if "val_loss" in hist:
            axes[1].plot(ep, hist["val_loss"], color=color, linewidth=1.9, label=label)
        if "val_accuracy" in hist:
            axes[2].plot(ep, hist["val_accuracy"], color=color, linewidth=1.9, label=label)

    if common_floor is not None:
        axes[0].axhline(common_floor, color="#555555", linestyle="--", linewidth=1.1,
                        label=f"H(q)={common_floor:.4f}")
        axes[1].axhline(common_floor, color="#555555", linestyle="--", linewidth=1.1,
                        label=f"H(q)={common_floor:.4f}")

    axes[0].set_title("Stage 2 Train Loss")
    axes[0].set_xlabel("Stage 2 Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Stage 2 Validation Loss")
    axes[1].set_xlabel("Stage 2 Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    axes[2].set_title("Stage 2 Validation Accuracy")
    axes[2].set_xlabel("Stage 2 Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(top=1.005)
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="lower right", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_step_curve(run_dir: Path, out_path: Path, alpha: float = 0.05) -> dict | None:
    """Plot mini-batch step loss + EMA + epoch-end val_loss for one run."""
    step_csv = run_dir / "step_history.csv"
    if not step_csv.is_file():
        return None

    steps, epochs, losses = [], [], []
    with open(step_csv, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            steps.append(int(row["global_step"]))
            epochs.append(int(row["epoch"]))
            losses.append(float(row["loss"]))
    if not steps:
        return None

    steps_arr = np.asarray(steps, dtype=np.int64)
    epochs_arr = np.asarray(epochs, dtype=np.int64)
    losses_arr = np.asarray(losses, dtype=np.float64)
    ema_arr = compute_ema(losses_arr, alpha=alpha)

    hist = _load_history(run_dir)
    ls, num_cls = _load_meta(run_dir)
    floor = compute_entropy_floor(ls, num_cls)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(steps_arr, losses_arr, color="#5b8db8", alpha=0.35, linewidth=1.0,
            label="Batch train loss (raw)")
    ax.plot(steps_arr, ema_arr, color="#1f4e79", linewidth=2.0,
            label=f"EMA (alpha={alpha:.2f})")

    if hist.get("val_loss"):
        ep_steps, ep_vals = [], []
        for ep_idx, vloss in enumerate(hist["val_loss"], start=1):
            mask = epochs_arr == ep_idx
            if np.any(mask):
                ep_steps.append(int(steps_arr[mask][-1]))
                ep_vals.append(float(vloss))
        if ep_steps:
            ax.plot(ep_steps, ep_vals, "o--", color="#d95f02", markersize=4.5,
                    linewidth=1.5, label="Epoch-end val_loss")

    if floor is not None:
        ax.axhline(floor, color="#555555", linestyle="--", linewidth=1.1,
                   label=f"LS entropy floor H(q)={floor:.4f}")

    ax.set_title(f"{run_dir.name} — Mini-Batch Gradient Descent (Per Step)")
    ax.set_xlabel("Global Step (mini-batch)")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "run_id": run_dir.name,
        "logged_points": int(len(steps_arr)),
        "max_global_step": int(steps_arr[-1]),
        "final_ema_loss": round(float(ema_arr[-1]), 6),
        "plot_path": str(out_path),
    }


def generate_all_plots(runs_dir: Path, out_dir: Path, ema_alpha: float = 0.05) -> dict:
    """Generate combined Stage 1->2 curves, Stage 2 overlay, and step curves."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"combined": {}, "stage2_comparison": None, "step_curves": {}}

    for arch, (s1_id, s2_id) in ARCH_PAIRS.items():
        res = plot_combined_arch(
            arch=arch,
            s1_dir=runs_dir / s1_id,
            s2_dir=runs_dir / s2_id,
            out_path=out_dir / f"gd_combined_{arch}.png",
        )
        if res is not None:
            summary["combined"][arch] = res

    comp_path = out_dir / "gd_comparison_stage2.png"
    if plot_stage2_comparison(runs_dir, comp_path):
        summary["stage2_comparison"] = str(comp_path)

    for run_dir in sorted(runs_dir.iterdir()) if runs_dir.is_dir() else []:
        if not run_dir.is_dir():
            continue
        step_res = plot_step_curve(
            run_dir=run_dir,
            out_path=out_dir / f"gd_step_{run_dir.name}.png",
            alpha=ema_alpha,
        )
        if step_res is not None:
            summary["step_curves"][run_dir.name] = step_res

    summary_path = out_dir / "gd_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PlantVile gradient descent comparison plots.")
    parser.add_argument("--runs-dir", type=Path, default=Path("results/runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/gradient_descent_plots"))
    parser.add_argument("--ema-alpha", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    summary = generate_all_plots(args.runs_dir, args.out_dir, ema_alpha=args.ema_alpha)
    print(
        f"Gradient descent plots saved -> {args.out_dir} "
        f"(combined={len(summary['combined'])}, "
        f"stage2_overlay={'yes' if summary['stage2_comparison'] else 'no'}, "
        f"step_curves={len(summary['step_curves'])})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
