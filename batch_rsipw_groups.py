import os
import re
import csv
import sys
import time
import argparse
import subprocess
from statistics import mean


GROUPS = ["R", "720p", "1024p", "1080p", "1440p"]
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def find_group_folder(dataset_root, group_name):
    candidates = [
        os.path.join(dataset_root, group_name),
        os.path.join(dataset_root, "PNG", group_name),
        os.path.join(dataset_root, "JPG", group_name),
        os.path.join(dataset_root, "JPEG", group_name),
    ]

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    return None


def get_images_from_folder(folder):
    image_paths = []

    for filename in os.listdir(folder):
        extension = os.path.splitext(filename)[1].lower()

        if extension in VALID_EXTENSIONS:
            image_paths.append(os.path.join(folder, filename))

    image_paths.sort()
    return image_paths


def parse_float(pattern, text):
    match = re.search(pattern, text)

    if match:
        return float(match.group(1))

    return None


def parse_metrics(stdout):
    """
    rsipw.py prints block-level PSNR/SSIM first.
    The final full-image PSNR/SSIM is the LAST pair printed.
    """

    psnr_values = re.findall(r"PSNR taken\s+([0-9.]+)", stdout)
    ssim_values = re.findall(r"SSIM taken\s+([0-9.]+)", stdout)

    final_psnr = float(psnr_values[-1]) if psnr_values else None
    final_ssim = float(ssim_values[-1]) if ssim_values else None

    extraction_rate = parse_float(
        r"Extraction percentage\s*=\s*([0-9.]+)%",
        stdout
    )

    elapsed_mins = parse_float(
        r"Elapsed time\s*=\s*([0-9.]+)\s*mins",
        stdout
    )

    elapsed_secs = parse_float(
        r"Elapsed time\s*=\s*([0-9.]+)\s*seconds",
        stdout
    )

    return {
        "psnr": final_psnr,
        "ssim": final_ssim,
        "extraction_rate": extraction_rate,
        "elapsed_mins": elapsed_mins,
        "elapsed_secs": elapsed_secs,
    }


def run_single_image(rsipw_path, image_path, code, mode):
    cmd = [
        sys.executable,
        rsipw_path,
        image_path,
        code,
        mode
    ]

    start = time.time()

    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    runtime_secs = time.time() - start

    output = completed.stdout + "\n" + completed.stderr
    metrics = parse_metrics(output)

    metrics["image"] = os.path.basename(image_path)
    metrics["image_path"] = image_path
    metrics["return_code"] = completed.returncode
    metrics["runtime_secs_by_batch"] = runtime_secs
    metrics["status"] = "OK" if completed.returncode == 0 else "FAILED"

    return metrics, output


def safe_mean(values):
    clean_values = [v for v in values if v is not None]

    if not clean_values:
        return None

    return mean(clean_values)


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Run rsipw.py for all image groups and save average metrics per group."
    )

    parser.add_argument(
        "dataset_root",
        help="Root dataset folder, for example: generic-dataset or generic-dataset/PNG"
    )

    parser.add_argument(
        "code",
        help="Code to embed, for example: 123456707"
    )

    parser.add_argument(
        "mode",
        choices=["FAST", "FULL"],
        help="Embedding mode"
    )

    parser.add_argument(
        "--rsipw",
        default="rsipw.py",
        help="Path to rsipw.py. Default: rsipw.py"
    )

    parser.add_argument(
        "--output",
        default="batch_results",
        help="Output folder for CSV files and logs. Default: batch_results"
    )

    args = parser.parse_args()

    per_image_rows = []
    summary_rows = []

    for group in GROUPS:
        group_folder = find_group_folder(args.dataset_root, group)

        if group_folder is None:
            print(f"[SKIP] Group folder not found: {group}")
            continue

        image_paths = get_images_from_folder(group_folder)

        if not image_paths:
            print(f"[SKIP] No images found in: {group_folder}")
            continue

        print("\n======================================")
        print(f"GROUP: {group}")
        print(f"FOLDER: {group_folder}")
        print(f"IMAGES: {len(image_paths)}")
        print("======================================\n")

        group_rows = []

        for image_path in image_paths:
            print(f"Running: {image_path}")

            metrics, output = run_single_image(
                args.rsipw,
                image_path,
                args.code,
                args.mode
            )

            metrics["group"] = group
            group_rows.append(metrics)
            per_image_rows.append(metrics)

            log_dir = os.path.join(args.output, "logs", group)
            os.makedirs(log_dir, exist_ok=True)

            log_name = os.path.splitext(os.path.basename(image_path))[0] + ".log"
            log_path = os.path.join(log_dir, log_name)

            with open(log_path, "w", encoding="utf-8", errors="ignore") as f:
                f.write(output)

            print(
                f"  Status={metrics['status']} | "
                f"PSNR={metrics['psnr']} | "
                f"SSIM={metrics['ssim']} | "
                f"ER={metrics['extraction_rate']} | "
                f"Time={metrics['elapsed_secs']} sec"
            )

        summary = {
            "group": group,
            "images": len(group_rows),
            "successful_runs": sum(1 for row in group_rows if row["status"] == "OK"),
            "avg_psnr": safe_mean([row["psnr"] for row in group_rows]),
            "avg_ssim": safe_mean([row["ssim"] for row in group_rows]),
            "avg_extraction_rate": safe_mean([row["extraction_rate"] for row in group_rows]),
            "avg_elapsed_mins": safe_mean([row["elapsed_mins"] for row in group_rows]),
            "avg_elapsed_secs": safe_mean([row["elapsed_secs"] for row in group_rows]),
            "avg_runtime_secs_by_batch": safe_mean([row["runtime_secs_by_batch"] for row in group_rows]),
        }

        summary_rows.append(summary)

    per_image_csv = os.path.join(args.output, "per_image_results.csv")
    summary_csv = os.path.join(args.output, "group_summary.csv")

    write_csv(
        per_image_csv,
        per_image_rows,
        [
            "group",
            "image",
            "image_path",
            "status",
            "return_code",
            "psnr",
            "ssim",
            "extraction_rate",
            "elapsed_mins",
            "elapsed_secs",
            "runtime_secs_by_batch",
        ]
    )

    write_csv(
        summary_csv,
        summary_rows,
        [
            "group",
            "images",
            "successful_runs",
            "avg_psnr",
            "avg_ssim",
            "avg_extraction_rate",
            "avg_elapsed_mins",
            "avg_elapsed_secs",
            "avg_runtime_secs_by_batch",
        ]
    )

    print("\nDONE")
    print(f"Per-image results: {per_image_csv}")
    print(f"Group summary:     {summary_csv}")


if __name__ == "__main__":
    main()
