"""Read-only dataset audit for PlantVillage (local or Kaggle).

Checks directory layout (including nested duplicate directories), file counts,
extensions, non-image files, exact MD5 duplicates, PIL image integrity,
resolution/mode consistency, pre-augmentation tags, and class imbalance.

Example:
    python scripts/audit_dataset.py --root data/PlantVillage --out runs/audit.json
"""
import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")
AUG_PATTERNS = ("_flip", "_rot", "180deg", "90deg", "270deg", "final_masked")


def _resolve_dir(root: str, variant: str) -> str:
    candidates = [
        os.path.join(root, variant),
        os.path.join(root, "PlantVillage", variant),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    if os.path.isdir(root):
        for entry in sorted(os.listdir(root)):
            sub = os.path.join(root, entry)
            if os.path.isdir(sub) and any(
                f.endswith(IMG_EXTS) and os.path.isfile(os.path.join(sub, f))
                for f in os.listdir(sub)
            ):
                return root
    raise FileNotFoundError("Dataset directory not found under %s" % os.path.abspath(root))


def _md5_file(item):
    cls, fname, fpath = item
    with open(fpath, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest(), cls, fname


def _verify_pil(item):
    cls, fname, fpath, fsize = item
    try:
        with Image.open(fpath) as im:
            im.verify()
        with Image.open(fpath) as im:
            return True, im.mode, "%dx%d" % im.size, cls, fname, fsize
    except Exception as exc:
        return False, str(exc), None, cls, fname, fsize


def audit_directory(
    root: str,
    variant: str = "color",
    sample_verify: int = 100,
    seed: int = 42,
) -> dict:
    """Audit a PlantVillage-style directory and return a structured summary dict."""
    variant_dir = _resolve_dir(root, variant)
    raw_dirs = sorted(
        d for d in os.listdir(variant_dir)
        if os.path.isdir(os.path.join(variant_dir, d))
    )

    class_dirs, skipped_dirs = [], []
    for d in raw_dirs:
        sub = os.path.join(variant_dir, d)
        has_images = any(
            f.endswith(IMG_EXTS) and os.path.isfile(os.path.join(sub, f))
            for f in os.listdir(sub)
        )
        if has_images:
            class_dirs.append(d)
        else:
            skipped_dirs.append(d)

    ext_counts = Counter()
    non_image_files = []
    all_records = []
    class_counts = {}
    size_groups = defaultdict(list)
    pre_aug_files = 0
    same_leaf_groups_total = 0
    same_leaf_files_total = 0

    for cls in class_dirs:
        cdir = os.path.join(variant_dir, cls)
        n_valid = 0
        leaf_groups = defaultdict(list)
        with os.scandir(cdir) as it:
            entries = sorted((e for e in it if e.is_file()), key=lambda e: e.name)
        for entry in entries:
            fname = entry.name
            fsize = entry.stat().st_size
            _, ext = os.path.splitext(fname)
            ext_counts[ext] += 1
            if not fname.endswith(IMG_EXTS):
                non_image_files.append({"class": cls, "file": fname, "bytes": fsize})
                continue
            n_valid += 1
            all_records.append((cls, fname, entry.path, fsize))
            size_groups[fsize].append((cls, fname, entry.path))

            lower = fname.lower()
            if any(p in lower for p in AUG_PATTERNS):
                pre_aug_files += 1
            rhs = fname.split("___", 1)[1] if "___" in fname else fname
            m_leaf = re.search(r"(Leaf\s+\d+)\.\d+", rhs, flags=re.I)
            if m_leaf:
                leaf_groups[m_leaf.group(1)].append(fname)

        class_counts[cls] = n_valid
        multi = [v for v in leaf_groups.values() if len(v) > 1]
        same_leaf_groups_total += len(multi)
        same_leaf_files_total += sum(len(v) for v in multi)

    byte_arr = np.array([r[3] for r in all_records], dtype=np.int64)
    dup_candidates = [item for grp in size_groups.values() if len(grp) > 1 for item in grp]
    hashes = defaultdict(list)
    if dup_candidates:
        with ThreadPoolExecutor(max_workers=16) as pool:
            for digest, cls, fname in pool.map(_md5_file, dup_candidates):
                hashes[digest].append((cls, fname))

    dup_groups = {h: items for h, items in hashes.items() if len(items) > 1}
    within_dups = [items for items in dup_groups.values() if len({c for c, _ in items}) == 1]
    cross_dups = [items for items in dup_groups.values() if len({c for c, _ in items}) > 1]

    by_cls = defaultdict(list)
    for rec in all_records:
        by_cls[rec[0]].append(rec)
    rng = random.Random(seed)
    verify_map = {r[2]: r for r in all_records if r[3] < 5000 or r[3] > 40000}
    if sample_verify > 0:
        for cls in class_dirs:
            items = by_cls[cls]
            for rec in rng.sample(items, min(sample_verify, len(items))):
                verify_map[rec[2]] = rec

    modes, resolutions, corrupt_files = Counter(), Counter(), []
    if verify_map:
        with ThreadPoolExecutor(max_workers=16) as pool:
            for ok, mode_or_err, size_str, cls, fname, fsize in pool.map(
                _verify_pil, verify_map.values()
            ):
                if ok:
                    modes[mode_or_err] += 1
                    resolutions[size_str] += 1
                else:
                    corrupt_files.append({
                        "class": cls, "file": fname, "bytes": fsize, "error": mode_or_err
                    })

    counts_list = list(class_counts.values())
    imbalance_ratio = (
        float(max(counts_list) / min(counts_list)) if counts_list and min(counts_list) > 0 else 0.0
    )

    return {
        "resolved_dir": os.path.abspath(variant_dir),
        "classes": len(class_dirs),
        "skipped_subdirs": skipped_dirs,
        "valid_images": len(all_records),
        "total_mb": round(float(byte_arr.sum() / (1024 * 1024)) if len(byte_arr) else 0.0, 2),
        "file_bytes": {
            "min": int(byte_arr.min()) if len(byte_arr) else 0,
            "median": int(np.median(byte_arr)) if len(byte_arr) else 0,
            "max": int(byte_arr.max()) if len(byte_arr) else 0,
        },
        "extensions": dict(ext_counts),
        "non_image_files": non_image_files,
        "class_counts": class_counts,
        "imbalance_ratio": round(imbalance_ratio, 2),
        "exact_md5_duplicates": {
            "groups": len(dup_groups),
            "extra_files": sum(len(items) - 1 for items in dup_groups.values()),
            "within_class_groups": len(within_dups),
            "cross_class_groups": len(cross_dups),
            "examples": [
                {"files": [{"class": c, "file": f} for c, f in items]}
                for items in list(dup_groups.values())[:5]
            ],
        },
        "pil_verification": {
            "checked_files": len(verify_map),
            "corrupt_count": len(corrupt_files),
            "corrupt_files": corrupt_files[:10],
            "modes": dict(modes),
            "resolutions": dict(resolutions),
        },
        "pre_aug_files": pre_aug_files,
        "same_leaf_groups": same_leaf_groups_total,
        "same_leaf_files": same_leaf_files_total,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audit PlantVillage dataset directory")
    parser.add_argument("--root", default="data/PlantVillage", help="Dataset root path")
    parser.add_argument("--variant", default="color", help="Variant subfolder if hierarchical")
    parser.add_argument("--sample-verify", type=int, default=100,
                        help="Stratified samples per class for PIL verification")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="", help="Optional JSON output path")
    args = parser.parse_args(argv)

    report = audit_directory(
        args.root, variant=args.variant, sample_verify=args.sample_verify, seed=args.seed
    )

    print("\n=== Dataset Audit ===")
    print("  Resolved dir     :", report["resolved_dir"])
    print("  Classes          :", report["classes"])
    print("  Valid images     :", report["valid_images"])
    print("  Total MB         :", report["total_mb"])
    print("  Imbalance ratio  : %.2fx" % report["imbalance_ratio"])
    print("  Skipped subdirs  :", report["skipped_subdirs"] or "none")
    print("  Non-image files  :", len(report["non_image_files"]))
    print("  MD5 dup groups   : %d (within=%d, cross=%d)" % (
        report["exact_md5_duplicates"]["groups"],
        report["exact_md5_duplicates"]["within_class_groups"],
        report["exact_md5_duplicates"]["cross_class_groups"],
    ))
    print("  PIL verified     : %d | corrupt=%d | modes=%s | sizes=%s" % (
        report["pil_verification"]["checked_files"],
        report["pil_verification"]["corrupt_count"],
        report["pil_verification"]["modes"],
        report["pil_verification"]["resolutions"],
    ))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print("  Saved report     :", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
