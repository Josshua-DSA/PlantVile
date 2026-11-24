"""Training: callbacks, fit loop, and the artifacts written to runs/<run_name>/."""
import os

import keras

from src.data import prepare_datasets
from src.model import build_model, compile_model
from src.utils import make_run_dir, print_kv_table, save_json, set_seed


def build_callbacks(cfg, run_dir: str):
    """ModelCheckpoint + optional EarlyStopping / ReduceLROnPlateau / CSVLogger."""
    train_cfg = cfg["train"]
    callbacks = []

    monitor = train_cfg["checkpoint_monitor"]
    callbacks.append(keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(run_dir, "best.keras"),
        monitor=monitor,
        mode="min" if "loss" in monitor else "max",
        save_best_only=train_cfg["save_best_only"],
        verbose=1,
    ))

    if train_cfg["early_stopping"]:
        mode = train_cfg["es_mode"]
        if mode == "auto":
            mode = "min" if "loss" in train_cfg["es_monitor"] else "max"
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor=train_cfg["es_monitor"],
            patience=train_cfg["es_patience"],
            min_delta=train_cfg["es_min_delta"],
            mode=mode,
            restore_best_weights=train_cfg["es_restore_best"],
            verbose=1,
        ))

    if train_cfg["reduce_lr"]:
        mode = "min" if "loss" in train_cfg["lr_monitor"] else "max"
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor=train_cfg["lr_monitor"],
            factor=train_cfg["lr_factor"],
            patience=train_cfg["lr_patience"],
            min_lr=train_cfg["lr_min_lr"],
            cooldown=train_cfg["lr_cooldown"],
            mode=mode,
            verbose=1,
        ))

    if cfg["runtime"]["log_csv"]:
        callbacks.append(keras.callbacks.CSVLogger(
            os.path.join(run_dir, "history.csv"), append=False))

    if cfg["runtime"]["tensorboard"]:
        callbacks.append(keras.callbacks.TensorBoard(
            log_dir=os.path.join(run_dir, "tensorboard"), histogram_freq=0))

    return callbacks


def run_training(cfg):
    """Full training run. Returns (model, history_dict, run_dir, bundle)."""
    from src.utils import configure_device, get_distribution_strategy

    runtime = cfg["runtime"]
    set_seed(runtime["seed"], runtime["deterministic"])
    device_info = configure_device(runtime["device"])

    policy = "float32" if cfg["train"]["mixed_precision"] == "off" else cfg["train"]["mixed_precision"]
    keras.mixed_precision.set_global_policy(policy)

    strategy, strategy_desc = get_distribution_strategy(runtime["strategy"])
    device_info["strategy"] = strategy_desc
    device_info["num_replicas"] = int(strategy.num_replicas_in_sync)

    run_dir = make_run_dir(runtime["runs_dir"], runtime["run_name"])
    print("\nRun dir : %s" % run_dir)
    print("Device  : %s" % device_info["active"])
    print("Strategy: %s" % strategy_desc)

    bundle = prepare_datasets(cfg)
    with strategy.scope():
        model = compile_model(build_model(cfg, bundle["num_classes"]), cfg["train"])

        if runtime["resume_from"]:
            if not os.path.isfile(runtime["resume_from"]):
                raise FileNotFoundError("runtime.resume_from tidak ada: %s"
                                        % runtime["resume_from"])
            model.load_weights(runtime["resume_from"])
            print("Resumed weights dari %s" % runtime["resume_from"])

    fit_kwargs = {
        "epochs": cfg["train"]["epochs"],
        "callbacks": build_callbacks(cfg, run_dir),
        "verbose": runtime["verbose"],
    }
    if cfg["train"]["steps_per_epoch"] > 0:
        fit_kwargs["steps_per_epoch"] = cfg["train"]["steps_per_epoch"]
    if cfg["train"]["validation_steps"] > 0:
        fit_kwargs["validation_steps"] = cfg["train"]["validation_steps"]
    if bundle["class_weights"]:
        fit_kwargs["class_weight"] = bundle["class_weights"]

    print_kv_table(
        [("train", len(bundle["train"]["paths"])),
         ("val", len(bundle["val"]["paths"])),
         ("test", len(bundle["test"]["paths"])),
         ("class_weight", cfg["train"]["class_weight"]),
         ("mixed_precision", policy),
         ("strategy", strategy_desc)],
        title="Training plan",
    )

    history = model.fit(bundle["train"]["ds"],
                        validation_data=bundle["val"]["ds"],
                        **fit_kwargs)

    # Final artifacts: last-epoch weights in final.keras, best checkpoint in best.keras
    model.save(os.path.join(run_dir, "final.keras"))
    best_path = os.path.join(run_dir, "best.keras")
    if os.path.isfile(best_path):
        with strategy.scope():
            model.load_weights(best_path)
    save_json(os.path.join(run_dir, "history.json"),
              {k: [float(v) for v in vals] for k, vals in history.history.items()})
    save_json(os.path.join(run_dir, "class_names.json"), bundle["class_names"])
    save_json(os.path.join(run_dir, "config_used.json"), cfg)
    save_json(os.path.join(run_dir, "device.json"), device_info)

    with open(os.path.join(run_dir, "config_used.yaml"), "w", encoding="utf-8") as handle:
        import yaml
        yaml.safe_dump(cfg, handle, sort_keys=False)

    print("\nTraining selesai. Output: %s" % run_dir)
    return model, history.history, run_dir, bundle
