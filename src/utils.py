"""Shared helpers: seeding, device selection, run directories, small IO."""
import datetime
import json
import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and TF so runs are repeatable."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf

    tf.random.set_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()


def configure_device(device: str) -> dict:
    """Honour runtime.device: 'cpu' hides the GPU, 'gpu' requires one, 'auto' takes what exists."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")

    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
        active = "cpu (GPU hidden)"
    elif device == "gpu":
        if not gpus:
            raise RuntimeError(
                "runtime.device=gpu but TensorFlow sees no GPU.\n"
                "  TF build: %s / CUDA %s\n"
                "  Either install the matching nvidia-*-cu12 wheels, or set "
                "runtime.device=cpu." % (
                    tf.__version__,
                    tf.sysconfig.get_build_info().get("cuda_version"),
                )
            )
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        active = "gpu (%s)" % ", ".join(g.name for g in gpus)
    else:  # auto
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass
            active = "gpu (%s)" % ", ".join(g.name for g in gpus)
        else:
            active = "cpu (no GPU visible to TensorFlow)"

    return {"requested": device, "active": active, "gpus": [g.name for g in gpus]}


def get_distribution_strategy(strategy_name: str):
    """Select MirroredStrategy for multi-GPU when enabled, else default single-device."""
    import tensorflow as tf

    visible_gpus = tf.config.get_visible_devices("GPU")
    if strategy_name == "none" or len(visible_gpus) <= 1:
        if strategy_name == "mirrored" and len(visible_gpus) < 2:
            print("[WARN] runtime.strategy=mirrored tetapi hanya %d GPU terlihat; "
                  "memakai single-device strategy" % len(visible_gpus))
        return tf.distribute.get_strategy(), "default (single device)"

    strategy = tf.distribute.MirroredStrategy()
    return strategy, "MirroredStrategy (%d GPUs)" % len(visible_gpus)


def make_run_dir(runs_dir: str, run_name: str) -> str:
    """Create (or reuse) runs/<run_name> and return its absolute path."""
    if not run_name:
        run_name = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    path = os.path.abspath(os.path.join(runs_dir, run_name))
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)


def print_kv_table(rows, title: str = "") -> None:
    """Left-aligned two-column dump. Rows = [(key, value), ...]."""
    if title:
        print("\n" + title)
        print("-" * len(title))
    width = max((len(str(k)) for k, _ in rows), default=0)
    for key, value in rows:
        print("  %-*s : %s" % (width, key, value))
