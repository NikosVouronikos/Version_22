import os
import re
import csv
import json
import sys
import subprocess
from datetime import datetime

# ============================================================
# MVDW Extract Attacked Images
# Runs validator.py on every attacked image.
#
# Usage:
#   python extract_attacked_mvdw.py 112230765
#
# Optional:
#   python extract_attacked_mvdw.py 112230765 --validator validator.py --attacked attacked --output attacked_extraction_results --reco 0
#
# Note: --reco is passed as a third CLI argument only if your validator.py supports it.
# ============================================================

GROUPS = ["R", "720p", "1024p", "1080p", "1440p"]
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_float(pattern, text):
    match = re.search(pattern, text)

    if match:
        value = match.group(1).replace(",", ".")
        return float(value)

    return None


def find_attacked_images(attacked_root):
    attacked_images = []
    for group in GROUPS:
        group_dir = os.path.join(attacked_root, group)
        if not os.path.isdir(group_dir):
            print(f"[SKIP] Missing attacked group: {group_dir}")
            continue
        for root, _, files in os.walk(group_dir):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext in VALID_EXTENSIONS:
                    attacked_images.append((group, os.path.join(root, filename), filename))
    attacked_images.sort(key=lambda x: (x[0], x[1]))
    return attacked_images


def extract_attack_info(path):
    normalized = str(path).replace("\\", "/")
    lower = normalized.lower()

    category = "Unknown"

    if "/crops/" in lower:
        category = "Crops"
    elif "/filters/" in lower:
        category = "Filters"
    elif "/resize/" in lower:
        category = "Resize"
    elif "/geometric/" in lower:
        category = "Geometric"

    name = os.path.basename(path)

    patterns = [
        ("jpeg", r"Compressed_q([0-9]+)_"),
        ("gaussian_noise", r"GN_lvl([0-9.]+)_"),
        ("salt_pepper", r"SaP_p([0-9.]+)_"),
        ("gaussian_blur", r"GB_k([0-9]+)_"),
        ("median_filter", r"MF_k([0-9]+)_"),
        ("motion_blur", r"MB_k([0-9]+)_"),
        ("histogram_equalization", r"HEQ_"),
        ("gamma", r"Gamma_"),
        ("sharpening", r"SHARP_"),
        ("resize", r"RESIZE_([0-9]+)_"),
        ("rotation", r"ROT_(-?[0-9]+)_"),
        ("translation", r"TRANS_x([0-9]+)_y([0-9]+)_"),
        ("horizontal_flip", r"HFLIP_"),
        ("mean_filter", r"MEAN_k([0-9]+)_"),
        ("crop", r"Croped_([A-Za-z]+)_([0-9]+)_"),
    ]

    attack_type = "unknown"
    attack_level = ""

    for atype, pattern in patterns:
        match = re.search(pattern, name)

        if match:
            attack_type = atype

            if atype == "translation":
                attack_level = f"x{match.group(1)}_y{match.group(2)}"
            elif atype == "crop":
                attack_level = f"{match.group(1)}_{match.group(2)}"
            elif match.groups():
                attack_level = match.group(1)
            else:
                attack_level = ""

            break

    return category, attack_type, attack_level


def parse_validator_output(output):
    ber = None
    extraction_rate = None
    extracted_code = None

    for line in output.splitlines():
        clean = line.strip()

        # BER line
        if "ber" in clean.lower():
            nums = re.findall(r"[0-9]+(?:[.,][0-9]+)?", clean)
            if nums:
                ber = float(nums[-1].replace(",", "."))

        # Extraction rate line
        if "%" in clean:
            nums = re.findall(r"[0-9]+(?:[.,][0-9]+)?", clean)
            if nums:
                extraction_rate = float(nums[-1].replace(",", "."))

    matches = re.findall(r"\[[^\]]+\]", output)
    if matches:
        extracted_code = matches[-1]

    return extraction_rate, ber, extracted_code


def run_validator(validator_path, image_path, code, reco=None):
    validator_abs = os.path.abspath(validator_path)
    validator_dir = os.path.dirname(validator_abs)
    image_abs = os.path.abspath(image_path)

    cmd = [sys.executable, validator_abs, image_abs, code]

    if reco is not None:
        cmd.append(str(reco))

    completed = subprocess.run(
        cmd,
        cwd=validator_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    output = completed.stdout + "\n" + completed.stderr
    extraction_rate, ber, extracted_code = parse_validator_output(output)

    return {
        "return_code": completed.returncode,
        "extraction_rate_percent": extraction_rate,
        "ber": ber,
        "extracted_code": extracted_code,
        "output": output
    }


def result_file_paths(output_root):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_root, f"attacked_extraction_results_{timestamp}.csv")
    json_path = os.path.join(output_root, f"attacked_extraction_results_{timestamp}.json")
    txt_path = os.path.join(output_root, f"attacked_extraction_summary_{timestamp}.txt")
    return csv_path, json_path, txt_path


def write_csv(csv_path, rows):
    header = [
        "timestamp",
        "group",
        "category",
        "attack_type",
        "attack_level",
        "attacked_image_name",
        "attacked_image_path",
        "return_code",
        "extraction_rate_percent",
        "ber",
        "extracted_code"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def safe_average(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def aggregate_results(rows):
    by_group_attack = {}
    for row in rows:
        key = (row["group"], row["category"], row["attack_type"], row["attack_level"])
        if key not in by_group_attack:
            by_group_attack[key] = {
                "group": row["group"],
                "category": row["category"],
                "attack_type": row["attack_type"],
                "attack_level": row["attack_level"],
                "count": 0,
                "avg_extraction_rate_percent": None,
                "avg_ber": None,
                "_rates": [],
                "_bers": []
            }
        by_group_attack[key]["count"] += 1
        by_group_attack[key]["_rates"].append(row["extraction_rate_percent"])
        by_group_attack[key]["_bers"].append(row["ber"])

    aggregate = []
    for item in by_group_attack.values():
        item["avg_extraction_rate_percent"] = safe_average(item["_rates"])
        item["avg_ber"] = safe_average(item["_bers"])
        del item["_rates"]
        del item["_bers"]
        aggregate.append(item)

    aggregate.sort(key=lambda x: (x["group"], x["category"], x["attack_type"], str(x["attack_level"])))
    return aggregate


def write_json(json_path, rows, aggregate):
    payload = {"generated_at": datetime.now().isoformat(), "results": rows, "aggregate": aggregate}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_summary(txt_path, rows, aggregate):
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("MVDW ATTACKED IMAGE EXTRACTION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated at: {datetime.now().isoformat()}\n")
        f.write(f"Total attacked images processed: {len(rows)}\n\n")
        f.write("Overall averages\n")
        f.write("-" * 40 + "\n")
        f.write(f"Average extraction rate (%): {safe_average([r['extraction_rate_percent'] for r in rows])}\n")
        f.write(f"Average BER: {safe_average([r['ber'] for r in rows])}\n\n")
        f.write("Aggregate by group and attack\n")
        f.write("-" * 40 + "\n")
        for item in aggregate:
            f.write(
                f"{item['group']} | {item['category']} | {item['attack_type']} {item['attack_level']} | "
                f"count={item['count']} | avg_ER={item['avg_extraction_rate_percent']} | avg_BER={item['avg_ber']}\n"
            )


def parse_args(argv):
    code = None
    validator = "validator.py"
    attacked = "attacked"
    output = "attacked_extraction_results"
    reco = None

    if len(argv) > 1:
        code = argv[1]

    i = 2
    while i < len(argv):
        if argv[i] == "--validator":
            validator = argv[i + 1]
            i += 2
        elif argv[i] == "--attacked":
            attacked = argv[i + 1]
            i += 2
        elif argv[i] == "--output":
            output = argv[i + 1]
            i += 2
        elif argv[i] == "--reco":
            reco = argv[i + 1]
            i += 2
        else:
            i += 1

    if code is None:
        print("Usage: python extract_attacked_mvdw.py <code> [--validator validator.py] [--attacked attacked] [--output attacked_extraction_results] [--reco 0|1]")
        sys.exit(1)

    return code, validator, attacked, output, reco

def write_aggregate_csv(csv_path, aggregate):

    header = [
        "group",
        "category",
        "attack_type",
        "attack_level",
        "count",
        "avg_extraction_rate_percent",
        "avg_ber"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for row in aggregate:

            row["avg_extraction_rate_percent"] = (
                round(row["avg_extraction_rate_percent"], 2)
                if row["avg_extraction_rate_percent"] is not None
                else None
            )

            row["avg_ber"] = (
                round(row["avg_ber"], 4)
                if row["avg_ber"] is not None
                else None
            )

            writer.writerow(row)


def main():
    code, validator_path, attacked_root, output_root, reco = parse_args(sys.argv)
    ensure_dir(output_root)

    images = find_attacked_images(attacked_root)
    print(f"Found {len(images)} attacked images")

    csv_path, json_path, txt_path = result_file_paths(output_root)
    logs_dir = os.path.join(output_root, "logs")
    ensure_dir(logs_dir)

    rows = []
    for idx, (group, image_path, image_name) in enumerate(images, start=1):
        category, attack_type, attack_level = extract_attack_info(image_path)
        print(f"[{idx}/{len(images)}] {group} | {category} | {attack_type} {attack_level} | {image_name}")

        result = run_validator(validator_path, image_path, code, reco=reco)

        log_name = f"{idx:05d}_{group}_{category}_{attack_type}_{attack_level}_{os.path.splitext(image_name)[0]}.log"
        log_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", log_name)
        log_path = os.path.join(logs_dir, log_name)
        with open(log_path, "w", encoding="utf-8", errors="ignore") as f:
            f.write(result["output"])

        row = {
            "timestamp": datetime.now().isoformat(),
            "group": group,
            "category": category,
            "attack_type": attack_type,
            "attack_level": attack_level,
            "attacked_image_name": image_name,
            "attacked_image_path": os.path.abspath(image_path),
            "return_code": result["return_code"],
            "extraction_rate_percent":
                round(result["extraction_rate_percent"], 2)
                if result["extraction_rate_percent"] is not None
                else None,
            "ber":
                round(result["ber"], 4)
                if result["ber"] is not None
                else None,
                    "extracted_code": result["extracted_code"]
        }
        rows.append(row)
        er_str = (
            f"{row['extraction_rate_percent']:.2f}"
            if row['extraction_rate_percent'] is not None
            else "None"
        )

        ber_str = (
            f"{row['ber']:.4f}"
            if row['ber'] is not None
            else "None"
        )

        print(
            f"  return={row['return_code']} | "
            f"ER={er_str}% | "
            f"BER={ber_str}"
        )

    aggregate = aggregate_results(rows)
    aggregate_csv = os.path.join(output_root,"aggregate_attack_results.csv")
    write_aggregate_csv(aggregate_csv, aggregate)
    write_csv(csv_path, rows)
    write_json(json_path, rows, aggregate)
    write_summary(txt_path, rows, aggregate)

    print("\nSaved:")
    print(f"CSV : {csv_path}")
    print(f"JSON: {json_path}")
    print(f"TXT : {txt_path}")


if __name__ == "__main__":
    main()
