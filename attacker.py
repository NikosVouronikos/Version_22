import os
import sys
import cv2
import numpy as np
import shutil
from PIL import Image

# ============================================================
# MVDW Attacker
# Reads watermarked images from:
#   watermarked/R, watermarked/720p, watermarked/1024p,
#   watermarked/1080p, watermarked/1440p
# Writes attacked images to:
#   attacked/<group>/<image_stem>/<AttackCategory>/...
# ============================================================

GROUPS = ["R", "720p", "1024p", "1080p", "1440p"]
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

def copy_metadata_files(source_image_path, target_folder):
    source_folder = os.path.dirname(source_image_path)

    for filename in ["Code_Mapping.txt", "BasicValues.txt", "GridPositions.txt"]:
        src = os.path.join(source_folder, filename)
        dst = os.path.join(target_folder, filename)

        if os.path.exists(src):
            shutil.copy2(src, dst)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def save_png(array, output_path):
    array = np.asarray(array)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    Image.fromarray(array).save(output_path, format="PNG")


def save_jpeg(array, output_path, quality):
    array = np.asarray(array)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    Image.fromarray(array).save(output_path, format="JPEG", quality=quality, subsampling=1)


def get_watermarked_images(watermarked_root):
    images = []
    for group in GROUPS:
        group_dir = os.path.join(watermarked_root, group)
        if not os.path.isdir(group_dir):
            print(f"[SKIP] Missing group folder: {group_dir}")
            continue
        for root, _, files in os.walk(group_dir):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext in VALID_EXTENSIONS:
                    images.append((group, os.path.join(root, filename), filename))
    images.sort(key=lambda x: (x[0], x[1]))
    return images


# ================= Crop attacks =================

def crop_attack(image_path, image_name, output_dir, direction, percentage):
    arr = load_rgb(image_path).copy()
    h, w = arr.shape[:2]

    if direction == "Left":
        d = int((percentage / 100) * w)
        arr[:, :d] = 0
    elif direction == "Right":
        d = int((percentage / 100) * w)
        arr[:, w - d:] = 0
    elif direction == "Top":
        d = int((percentage / 100) * h)
        arr[:d, :] = 0
    elif direction == "Bottom":
        d = int((percentage / 100) * h)
        arr[h - d:, :] = 0
    elif direction == "BothVertical":
        d = int((percentage / 200) * w)
        arr[:, :d] = 0
        arr[:, w - d:] = 0
    elif direction == "BothHorizontal":
        d = int((percentage / 200) * h)
        arr[:d, :] = 0
        arr[h - d:, :] = 0
    else:
        raise ValueError(f"Unsupported crop direction: {direction}")

    out_name = f"CROP_{direction}_{percentage}_{image_name}"
    out_path = os.path.join(output_dir, out_name)
    save_png(arr, out_path)
    return out_path


# ================= Filter/noise/compression attacks =================

def compression_attack(image_path, image_name, output_dir, quality):
    arr = load_rgb(image_path)
    base = os.path.splitext(image_name)[0]
    out_name = f"JPEG_q{quality}_{base}.jpg"
    out_path = os.path.join(output_dir, out_name)
    save_jpeg(arr, out_path, quality=quality)
    return out_path


def gaussian_blur_attack(image_path, image_name, output_dir, kernel_size):
    arr = load_rgb(image_path)
    attacked = cv2.GaussianBlur(arr, (kernel_size, kernel_size), 0)
    out_path = os.path.join(output_dir, f"GB_k{kernel_size}_{image_name}")
    save_png(attacked, out_path)
    return out_path


def gaussian_noise_attack(image_path, image_name, output_dir, std):
    arr = load_rgb(image_path).astype(np.float32)
    noise = np.random.normal(0, std, arr.shape)
    attacked = np.clip(arr + noise, 0, 255).astype(np.uint8)
    out_path = os.path.join(output_dir, f"GN_std{std}_{image_name}")
    save_png(attacked, out_path)
    return out_path


def salt_and_pepper_attack(image_path, image_name, output_dir, probability):
    arr = load_rgb(image_path).copy()
    h, w = arr.shape[:2]
    random_matrix = np.random.random((h, w))
    salt = random_matrix < (probability / 2)
    pepper = (random_matrix >= (probability / 2)) & (random_matrix < probability)
    arr[salt] = 255
    arr[pepper] = 0
    out_path = os.path.join(output_dir, f"SaP_p{probability}_{image_name}")
    save_png(arr, out_path)
    return out_path


def histogram_equalization_attack(image_path, image_name, output_dir):
    arr = load_rgb(image_path)
    ycrcb = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    attacked = cv2.merge((y_eq, cr, cb))
    attacked = cv2.cvtColor(attacked, cv2.COLOR_YCrCb2RGB)
    out_path = os.path.join(output_dir, f"HEQ_{image_name}")
    save_png(attacked, out_path)
    return out_path


def gamma_attack(image_path, image_name, output_dir, gamma=1.625):
    arr = load_rgb(image_path).astype(np.float32)
    attacked = np.power(arr / 255.0, gamma) * 255.0
    attacked = np.clip(attacked, 0, 255).astype(np.uint8)
    out_path = os.path.join(output_dir, f"GAMMA_{gamma}_{image_name}")
    save_png(attacked, out_path)
    return out_path


def median_attack(image_path, image_name, output_dir, kernel_size):
    arr = load_rgb(image_path)
    attacked = cv2.medianBlur(arr, kernel_size)
    out_path = os.path.join(output_dir, f"MF_k{kernel_size}_{image_name}")
    save_png(attacked, out_path)
    return out_path


def motion_blur_attack(image_path, image_name, output_dir, kernel_size=25):
    arr = load_rgb(image_path)
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[(kernel_size - 1) // 2, :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    attacked = cv2.filter2D(arr, -1, kernel)
    out_path = os.path.join(output_dir, f"MB_k{kernel_size}_{image_name}")
    save_png(attacked, out_path)
    return out_path


def sharpening_attack(image_path, image_name, output_dir):
    arr = load_rgb(image_path)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    attacked = cv2.filter2D(arr, -1, kernel)
    out_path = os.path.join(output_dir, f"SHARP_{image_name}")
    save_png(attacked, out_path)
    return out_path


# ================= Resize attacks =================

def resize_attack(image_path, image_name, output_dir, scale_percent):
    arr = load_rgb(image_path)
    h, w = arr.shape[:2]
    new_w = max(1, int(w * scale_percent / 100))
    new_h = max(1, int(h * scale_percent / 100))
    resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    attacked = cv2.resize(resized, (w, h), interpolation=cv2.INTER_LINEAR)
    out_path = os.path.join(output_dir, f"RESIZE_{scale_percent}_{image_name}")
    save_png(attacked, out_path)
    return out_path


# ================= Geometric attacks =================

def rotation_attack(image_path, image_name, output_dir, angle):
    arr = load_rgb(image_path)
    h, w = arr.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    attacked = cv2.warpAffine(arr, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out_path = os.path.join(output_dir, f"ROT_{angle}_{image_name}")
    save_png(attacked, out_path)
    return out_path


def translation_attack(image_path, image_name, output_dir, shift_x, shift_y):
    arr = load_rgb(image_path)
    h, w = arr.shape[:2]
    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    attacked = cv2.warpAffine(arr, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out_path = os.path.join(output_dir, f"TRANS_x{shift_x}_y{shift_y}_{image_name}")
    save_png(attacked, out_path)
    return out_path


def horizontal_flip_attack(image_path, image_name, output_dir):
    arr = load_rgb(image_path)
    attacked = cv2.flip(arr, 1)
    out_path = os.path.join(output_dir, f"HFLIP_{image_name}")
    save_png(attacked, out_path)
    return out_path


# ================= Frequency attacks =================

def fourier_random_noise_attack(image_path, image_name, output_dir, noise_level):
    arr = load_rgb(image_path)
    attacked = np.zeros_like(arr)
    for c in range(3):
        channel = arr[:, :, c]
        dft = np.fft.fft2(channel)
        dft_shift = np.fft.fftshift(dft)
        magnitude = np.abs(dft_shift)
        phase = np.angle(dft_shift)
        noise = noise_level * np.random.normal(0, 1, channel.shape)
        noisy_magnitude = magnitude + noise * magnitude
        noisy_dft = noisy_magnitude * np.exp(1j * phase)
        inverse_shift = np.fft.ifftshift(noisy_dft)
        channel_back = np.abs(np.fft.ifft2(inverse_shift))
        attacked[:, :, c] = np.clip(channel_back, 0, 255).astype(np.uint8)
    out_path = os.path.join(output_dir, f"FT_RAND_NOISE_{noise_level}_{image_name}")
    save_png(attacked, out_path)
    return out_path


def phase_spectrum_attack(image_path, image_name, output_dir, phase_noise_level):
    arr = load_rgb(image_path)
    attacked = np.zeros_like(arr)
    for c in range(3):
        channel = arr[:, :, c].astype(np.float32)
        dft = np.fft.fft2(channel)
        dft_shift = np.fft.fftshift(dft)
        magnitude = np.abs(dft_shift)
        phase = np.angle(dft_shift)
        noise = np.random.uniform(-np.pi, np.pi, phase.shape) * phase_noise_level
        noisy_dft = magnitude * np.exp(1j * (phase + noise))
        inverse_shift = np.fft.ifftshift(noisy_dft)
        channel_back = np.abs(np.fft.ifft2(inverse_shift))
        attacked[:, :, c] = np.clip(channel_back, 0, 255).astype(np.uint8)
    out_path = os.path.join(output_dir, f"PHASE_NOISE_{phase_noise_level}_{image_name}")
    save_png(attacked, out_path)
    return out_path


# ================= Runner =================

def attack_one_image(group, image_path, image_name, attacked_root):
    image_stem = os.path.splitext(image_name)[0]
    base_output_dir = os.path.join(attacked_root, group, image_stem)
    ensure_dir(base_output_dir)

    crops_dir = os.path.join(base_output_dir, "Crops")
    ensure_dir(crops_dir)
    copy_metadata_files(image_path, crops_dir)
    
    for direction in ["Left", "Right", "Top", "Bottom", "BothVertical", "BothHorizontal"]:
        for percentage in [25, 50, 75]:
            crop_attack(image_path, image_name, crops_dir, direction, percentage)

    filters_dir = os.path.join(base_output_dir, "Filters")
    ensure_dir(filters_dir)
    copy_metadata_files(image_path, filters_dir)
    
    for quality in [50, 70, 90]:
        compression_attack(image_path, image_name, filters_dir, quality)
    for std in [5, 10, 15.5]:
        gaussian_noise_attack(image_path, image_name, filters_dir, std)
    for probability in [0.01, 0.03, 0.05]:
        salt_and_pepper_attack(image_path, image_name, filters_dir, probability)
    for kernel_size in [3, 5, 7]:
        gaussian_blur_attack(image_path, image_name, filters_dir, kernel_size)
    for kernel_size in [3, 5, 7]:
        median_attack(image_path, image_name, filters_dir, kernel_size)
    histogram_equalization_attack(image_path, image_name, filters_dir)
    gamma_attack(image_path, image_name, filters_dir)
    motion_blur_attack(image_path, image_name, filters_dir, kernel_size=25)
    sharpening_attack(image_path, image_name, filters_dir)

    resize_dir = os.path.join(base_output_dir, "Resize")
    ensure_dir(resize_dir)
    copy_metadata_files(image_path, resize_dir)
    
    for scale_percent in [90, 110]:
        resize_attack(image_path, image_name, resize_dir, scale_percent)

    geometric_dir = os.path.join(base_output_dir, "Geometric")
    ensure_dir(geometric_dir)
    copy_metadata_files(image_path, geometric_dir)
    
    for angle in [5, 10, -10]:
        rotation_attack(image_path, image_name, geometric_dir, angle)
    for shift in [5, 10]:
        translation_attack(image_path, image_name, geometric_dir, shift, shift)
    horizontal_flip_attack(image_path, image_name, geometric_dir)

    frequency_dir = os.path.join(base_output_dir, "Frequency")
    ensure_dir(frequency_dir)
    copy_metadata_files(image_path, frequency_dir)
    
    for noise_level in [0.1, 0.2]:
        fourier_random_noise_attack(image_path, image_name, frequency_dir, noise_level)
    for phase_noise_level in [0.3, 0.5, 1.0]:
        phase_spectrum_attack(image_path, image_name, frequency_dir, phase_noise_level)


def main():
    project_root = os.getcwd()
    watermarked_root = os.path.join(project_root, "watermarked")
    attacked_root = os.path.join(project_root, "attacked")

    if len(sys.argv) >= 2:
        watermarked_root = sys.argv[1]
    if len(sys.argv) >= 3:
        attacked_root = sys.argv[2]

    ensure_dir(attacked_root)
    images = get_watermarked_images(watermarked_root)
    print(f"Found {len(images)} watermarked images")

    for idx, (group, image_path, image_name) in enumerate(images, start=1):
        print(f"[{idx}/{len(images)}] Attacking {image_path}")
        try:
            attack_one_image(group, image_path, image_name, attacked_root)
        except Exception as e:
            print(f"  FAILED: {e}")

    print("\nDone.")
    print(f"Attacked images saved in: {attacked_root}")


if __name__ == "__main__":
    main()
