# Spatial-domain impact of the embedding (modified pixels / bits) per resolution group.
# Compares each original image with its already-produced watermarked PNG, so no re-embedding is needed.
#
# Usage (from the Version_22 folder):
#   python spatial_impact.py --dataset <dataset_root> --watermarked <folder with watermarked_* outputs>
#
# The dataset root holds one folder per group (R, 720p, 1024p, 1080p, 1440p), directly or under PNG/.
# Watermarked files are found recursively by name: watermarked_<image name>*.png
import argparse, csv, os
import numpy as np
from PIL import Image

GROUPS = ["1440p", "1080p", "1024p", "720p", "R"]
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def group_folder(root, group):
    for c in (os.path.join(root, group), os.path.join(root, "PNG", group)):
        if os.path.isdir(c):
            return c
    return None


def index_watermarked(root):
    found = {}
    for d, _, files in os.walk(root):
        for f in files:
            if f.startswith("watermarked_") and f.lower().endswith(".png"):
                found.setdefault(f, os.path.join(d, f))
    return found


def find_watermarked(index, image_name):
    stem = os.path.splitext(image_name)[0]
    for key in ("watermarked_" + image_name + ".png", "watermarked_" + stem + ".png"):
        if key in index:
            return index[key]
    for key, path in index.items():
        if key.startswith("watermarked_" + stem):
            return path
    return None


def stats(original_path, watermarked_path):
    o = np.array(Image.open(original_path).convert("RGB"), dtype=np.uint8)
    w = np.array(Image.open(watermarked_path).convert("RGB"), dtype=np.uint8)
    if o.shape != w.shape:
        raise ValueError(f"shape mismatch {o.shape} vs {w.shape}")
    total_pixels = o.shape[0] * o.shape[1]
    changed_pixels = int(np.count_nonzero(np.any(o != w, axis=2)))
    total_bits = o.size * 8
    changed_bits = int(np.unpackbits(np.bitwise_xor(o, w)).sum())
    return total_pixels, 100 * changed_pixels / total_pixels, total_bits, 100 * changed_bits / total_bits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--watermarked", required=True)
    ap.add_argument("--output", default="spatial_impact.csv")
    a = ap.parse_args()

    index = index_watermarked(a.watermarked)
    rows, summary = [], []
    for g in GROUPS:
        folder = group_folder(a.dataset, g)
        if folder is None:
            print(f"[skip] no folder for group {g}")
            continue
        vals = []
        for name in sorted(os.listdir(folder)):
            if not name.lower().endswith(EXTS):
                continue
            wpath = find_watermarked(index, name)
            if wpath is None:
                print(f"[missing] {g}/{name}: no watermarked file found")
                continue
            tp, pp, tb, pb = stats(os.path.join(folder, name), wpath)
            rows.append([g, name, tp, round(pp, 4), tb, round(pb, 4)])
            vals.append((tp, pp, tb, pb))
        if vals:
            m = np.mean(np.array(vals, dtype=np.float64), axis=0)
            summary.append((g, m))

    with open(a.output, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["group", "image", "total_pixels", "mod_pixels_pct", "total_bits", "mod_bits_pct"])
        wr.writerows(rows)

    print("\nGroup averages (LaTeX rows):")
    allm = []
    for g, m in summary:
        label = "\\textit{Randomized}" if g == "R" else f"${g}$"
        print(f"{label} & ${int(round(m[0])):,}$ & ${m[1]:.2f}$ & ${int(round(m[2])):,}$ & ${m[3]:.2f}$ \\\\")
        allm.append(m)
    if allm:
        m = np.mean(np.array(allm), axis=0)
        print(f"\\textbf{{Average}} & $\\mathbf{{{int(round(m[0])):,}}}$ & $\\mathbf{{{m[1]:.2f}}}$ & "
              f"$\\mathbf{{{int(round(m[2])):,}}}$ & $\\mathbf{{{m[3]:.2f}}}$ \\\\")
    print(f"\nPer-image values written to {a.output}")


if __name__ == "__main__":
    main()
