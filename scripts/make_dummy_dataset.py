"""Make a tiny fake PlantVillage tree so --check can be verified without the real 342 MB dataset.

    python scripts/make_dummy_dataset.py --out data/dummy --classes 3 --per-class 12
"""
import argparse
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/dummy")
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--per-class", type=int, default=12)
    parser.add_argument("--size", type=int, default=96)
    args = parser.parse_args()

    from PIL import Image

    names = ["Tomato___healthy", "Tomato___Late_blight", "Potato___Early_blight",
             "Apple___Apple_scab", "Corn___Common_rust_"][:args.classes]
    rng = np.random.default_rng(0)
    variant_dir = os.path.join(args.out, "color")
    for name in names:
        cls_dir = os.path.join(variant_dir, name)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(args.per_class):
            pixels = rng.integers(0, 255, (args.size, args.size, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(os.path.join(cls_dir, "img_%03d.jpg" % i),
                                         quality=85)
        print("%-28s %d gambar" % (name, args.per_class))
    print("\nDummy dataset siap: %s" % os.path.abspath(args.out))


if __name__ == "__main__":
    main()
