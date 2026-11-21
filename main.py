#!/usr/bin/env python3
"""UAS Neuro Computing - PlantVillage plant disease classification (CNN).

Run modes
---------
  python main.py                                  menu interaktif penuh (semua parameter)
  python main.py --config configs/default.yaml    langsung dari YAML
  python main.py --config X --set model.arch=vgg16 --set train.epochs=5
  python main.py --check                          validasi config + dataset + model (TANPA training)
  python main.py --eval runs/<run>/best.keras     evaluasi model tersimpan
  python main.py --list-params                    dump semua parameter
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config as cfgmod  # noqa: E402
from src.utils import print_kv_table  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="PlantVillage CNN trainer (UAS Neuro Computing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default="", help="path YAML config")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="seksi.nama=nilai", help="override satu parameter (boleh berulang)")
    parser.add_argument("--check", action="store_true",
                        help="validasi config + scan dataset + build model, lalu keluar")
    parser.add_argument("--list-params", action="store_true", help="dump semua parameter")
    parser.add_argument("--eval", dest="model_path", default="",
                        help="path model .keras untuk evaluasi saja")
    parser.add_argument("--no-prompt", action="store_true",
                        help="jangan tampilkan wizard walau tanpa --config")
    return parser.parse_args(argv)


def print_config(cfg):
    rows = []
    for section, params in cfgmod.SPEC.items():
        for name, *_ in params:
            rows.append(("%s.%s" % (section, name), cfg[section][name]))
    print_kv_table(rows, title="Konfigurasi aktif")

def confirm_start(cfg):
    print("\n" + "=" * 66)
    print("  Siap training: %s | %s kelas | %s epoch | device=%s"
          % (cfg["model"]["arch"], "semua", cfg["train"]["epochs"],
             cfg["runtime"]["device"]))
    print("=" * 66)
    try:
        answer = input("Mulai training? [y/N]: ").strip().lower()
    except EOFError:
        answer = "n"
    return answer in ("y", "ya", "yes")


def do_check(cfg):
    """Validate everything short of training: config, dataset, model graph."""
    print("\n=== MODE CHECK (tidak ada training) ===")
    problems = cfgmod.validate(cfg)
    if problems:
        print("\nConfig bermasalah:")
        for item in problems:
            print("  ! %s" % item)
        return 1
    print("Config  : OK")

    from src.data import prepare_datasets
    from src.model import build_model
    from src.utils import configure_device

    device_info = configure_device(cfg["runtime"]["device"])
    print("Device  : %s" % device_info["active"])

    bundle = prepare_datasets(cfg)
    model = build_model(cfg, bundle["num_classes"])
    print("\nModel summary (layer pertama & terakhir):")
    for layer in list(model.layers)[:4]:
        print("  %-28s %s" % (layer.name, layer.__class__.__name__))
    print("  ...")
    for layer in list(model.layers)[-3:]:
        print("  %-28s %s" % (layer.name, layer.__class__.__name__))

    batch = next(iter(bundle["train"]["ds"]))
    images, labels = batch
    print("\nSample batch: images=%s labels=%s" % (images.shape, labels.shape))
    print("Pixel range : %.3f .. %.3f" % (float(images.numpy().min()),
                                          float(images.numpy().max())))
    print("\nSemua siap. Tidak ada training dijalankan.")
    return 0


def do_eval(cfg, model_path):
    import keras

    from src.evaluate import evaluate
    from src.utils import configure_device

    if not os.path.isfile(model_path):
        print("Model tidak ditemukan: %s" % model_path)
        return 1
    configure_device(cfg["runtime"]["device"])
    model = keras.models.load_model(model_path, compile=False)
    run_dir = os.path.dirname(os.path.abspath(model_path))
    print("Model   : %s" % model_path)
    evaluate(model, cfg, run_dir=run_dir, tag="reload")
    return 0


def main(argv=None):
    args = parse_args(argv)

    if args.list_params:
        print(cfgmod.list_params())
        return 0

    cfg = cfgmod.default_config()
    try:
        if args.config:
            if not os.path.isfile(args.config):
                print("Config tidak ditemukan: %s" % args.config)
                return 1
            cfg = cfgmod.from_yaml(args.config)
        elif not args.check and not args.model_path and not args.no_prompt:
            cfg = cfgmod.interactive_wizard(cfg)

        if args.overrides:
            cfg = cfgmod.apply_overrides(cfg, args.overrides)
    except (KeyError, ValueError) as exc:
        print("Config tidak valid: %s" % exc)
        print("Lihat daftar parameter dengan: python main.py --list-params")
        return 1
    except KeyboardInterrupt:
        print("\nDibatalkan.")
        return 1

    if not args.check and not args.model_path:
        print_config(cfg)

    if args.check:
        return do_check(cfg)

    if args.model_path:
        return do_eval(cfg, args.model_path)

    problems = cfgmod.validate(cfg)
    if problems:
        print("\nConfig bermasalah:")
        for item in problems:
            print("  ! %s" % item)
        return 1

    if not confirm_start(cfg):
        print("Dibatalkan. Tidak ada training dijalankan.")
        return 0

    from src.evaluate import evaluate
    from src.train import run_training

    model, history, run_dir, bundle = run_training(cfg)
    evaluate(model, cfg, bundle=bundle, run_dir=run_dir, history=history, tag="test")
    print("\nSelesai. Semua output ada di: %s" % run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
