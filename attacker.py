# Author: Nikolaos Vouronikos
import numpy as np,cv2,sys,os,json,shutil,subprocess
from PIL import Image

def save_image_ffmpeg(img_array, out_path, quality=None):
    img_array = np.ascontiguousarray(img_array.astype(np.uint8))
    if out_path.lower().endswith(('.jpg', '.jpeg')):
        if img_array.ndim == 3 and img_array.shape[2] == 3:
            pil_img = Image.fromarray(img_array, 'RGB')
        elif img_array.ndim == 2:
            pil_img = Image.fromarray(img_array, 'L')
        else:
            raise ValueError(f"Unsupported image shape for JPEG save: {img_array.shape}")
        pil_img.save(out_path,'JPEG',quality=quality if quality is not None else 75)
    else:
        if img_array.ndim == 2:
            h, w = img_array.shape
            pix_fmt = "gray"
        elif img_array.shape[2] == 3:
            h, w = img_array.shape[:2]
            pix_fmt = "rgb24"
        else:
            raise ValueError(f"Unsupported image shape for FFmpeg save: {img_array.shape}")
        
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f", "rawvideo",
                "-pix_fmt", pix_fmt,
                "-s", f"{w}x{h}",
                "-i", "-",
                "-frames:v", "1",
                "-c:v", "png",
                "-compression_level", "0",
                out_path
            ],
            input=img_array.tobytes(),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

def load_image_ffmpeg(path):
    probe = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,pix_fmt",
            "-of", "json",
            path
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=True
    )

    meta = json.loads(probe.stdout)
    stream = meta["streams"][0]

    w = int(stream["width"])
    h = int(stream["height"])

    pix_fmt = stream.get("pix_fmt", "rgb24").lower()
    has_alpha = "a" in pix_fmt
    out_pix_fmt = "rgba" if has_alpha else "rgb24"
    channels = 4 if has_alpha else 3
    raw = subprocess.run(
        [
            "ffmpeg",
            "-i", path,
            "-f", "rawvideo",
            "-pix_fmt", out_pix_fmt,
            "-"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True
    ).stdout

    img = np.frombuffer(raw,dtype=np.uint8).reshape((h, w, channels))
    return img[:, :, :3]

# ============================================================
# Config
# ============================================================
GROUPS = ["R", "720p", "1024p", "1080p", "1440p"]
VALID_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tiff",
    ".webp"
}

# ============================================================
# Utils
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def copy_metadata_files(source_image_path, target_folder):
    source_folder = os.path.dirname(source_image_path)
    for filename in [
        "Code_Mapping.txt",
        "BasicValues.txt",
        "GridPositions.txt"
    ]:
        src = os.path.join(source_folder, filename)
        dst = os.path.join(target_folder, filename)

        if os.path.exists(src):
            shutil.copy2(src, dst)

def get_watermarked_images(watermarked_root):
    images = []
    for group in GROUPS:
        group_dir = os.path.join(watermarked_root, group)
        if not os.path.isdir(group_dir):
            continue
        for root, _, files in os.walk(group_dir):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext in VALID_EXTENSIONS:
                    full_path = os.path.join(root, filename)
                    images.append((group,full_path,filename))
    images.sort(key=lambda x: (x[0], x[1]))
    return images

# ============================================================
# Crop Attacks
# ============================================================

def crop_attack(image_path,image_name,output_dir,direction,percentage):
    wImage = load_image_ffmpeg(image_path)
    wImage = wImage.copy()

    w = wImage.shape[1]
    h = wImage.shape[0]

    if direction == "Left":
        d = int((percentage / 100) * w)
        wImage[:, :d] = 0
    elif direction == "Right":
        d = int((percentage / 100) * w)
        wImage[:, w - d:] = 0
    elif direction == "Top":
        d = int((percentage / 100) * h)
        wImage[:d, :] = 0
    elif direction == "Bottom":
        d = int((percentage / 100) * h)
        wImage[h - d:, :] = 0
    elif direction == "BothVertical":
        d = int((percentage / 200) * w)
        wImage[:, :d] = 0
        wImage[:, w - d:] = 0
    elif direction == "BothHorizontal":
        d = int((percentage / 200) * h)
        wImage[:d, :] = 0
        wImage[h - d:, :] = 0
    else:
        raise ValueError("Unsupported crop direction")

    attacked_name = (f"Croped_{direction}_{percentage}_{image_name}")
    out_path = os.path.join(output_dir, attacked_name)
    save_image_ffmpeg(wImage, out_path)
    return out_path

# ============================================================
# Compression
# ============================================================

def compression_attack(image_path,image_name,output_dir,quality):
    wImage = load_image_ffmpeg(image_path)
    wImage = wImage.copy()

    attacked_name = (f"Compressed_q{quality}_"+ image_name.replace(".png", ".jpg"))
    out_path = os.path.join(output_dir, attacked_name)
    save_image_ffmpeg(wImage,out_path,quality=quality)
    return out_path

# ============================================================
# Gaussian Noise
# ============================================================

def gaussian_noise_attack(image_path,image_name,output_dir,noise_level=0.1):
    attacked_name = (f"GN_lvl{noise_level}_{image_name}")
    wImage = load_image_ffmpeg(image_path)
    wImage = wImage.copy()

    w = wImage.shape[0]
    h = wImage.shape[1]

    gauss_noise = np.zeros((w, h, 3),dtype=np.uint8)
    cv2.randn(gauss_noise,0,noise_level)
    gauss_noise = (gauss_noise * 0.5).astype(np.uint8)
    noisy = cv2.add(wImage,gauss_noise)
    out_path = os.path.join(output_dir,attacked_name)
    save_image_ffmpeg(noisy, out_path)
    return out_path

# ============================================================
# Salt & Pepper
# ============================================================

def salt_pepper_attack(image_path,image_name,output_dir,noise_prob=0.05):
    attacked_name = (f"SaP_p{noise_prob}_{image_name}")
    wImage = load_image_ffmpeg(image_path)
    wImage = wImage.copy()
    w = wImage.shape[0]
    h = wImage.shape[1]
    for i in range(w):
        for j in range(h):
            random_num_1 = np.random.uniform(low=0.0,high=1.0)
            random_num_2 = np.random.uniform(low=0.0,high=1.0)
            if random_num_1 < noise_prob:
                wImage[i, j] = 255
            elif random_num_2 < noise_prob:
                wImage[i, j] = 0

    out_path = os.path.join(output_dir,attacked_name)
    save_image_ffmpeg(wImage, out_path)
    return out_path

# ============================================================
# Gaussian Blur
# ============================================================

def gaussian_blur_attack(image_path,image_name,output_dir,kernel_size=5):
    attacked_name = (f"GB_k{kernel_size}_{image_name}")
    wImage = load_image_ffmpeg(image_path)
    wImage = wImage.copy()
    
    imblur = cv2.GaussianBlur(wImage,(kernel_size, kernel_size),0)
    out_path = os.path.join(output_dir,attacked_name)
    save_image_ffmpeg(imblur, out_path)
    return out_path

# ============================================================
# Histogram EQ
# ============================================================

def histogram_equalization_attack(image_path,image_name,output_dir):
    attacked_name = ("HEQ_" + image_name)
    wImage = load_image_ffmpeg(image_path)
    wImage = wImage.copy()

    img = cv2.cvtColor(wImage,cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img)
    y_eq = cv2.equalizeHist(y)
    img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
    img_heq = cv2.cvtColor(img_y_cr_cb_eq,cv2.COLOR_YCR_CB2BGR)
    out_path = os.path.join(output_dir,attacked_name)
    save_image_ffmpeg(img_heq, out_path)
    return out_path

# ============================================================
# Gamma
# ============================================================

def gamma_attack(image_path,image_name,output_dir):
    attacked_name = ("Gamma_" + image_name)
    wImage = load_image_ffmpeg(image_path)
    wImage = wImage.copy()

    image_gamma = cv2.filter2D(wImage,-1,(1.5))
    out_path = os.path.join(output_dir,attacked_name)
    save_image_ffmpeg(image_gamma,out_path)
    return out_path

# ============================================================
# Mean
# ============================================================

def mean_filter_attack(image_path,image_name,output_dir,kernel_size):
    attacked_name = (f"MEAN_k{kernel_size}_{image_name}")
    wImage = load_image_ffmpeg(image_path)
    wImage = wImage.copy()

    im_mean = cv2.blur(wImage, (kernel_size, kernel_size))

    out_path = os.path.join(output_dir,attacked_name)
    save_image_ffmpeg(im_mean, out_path)
    return out_path

# ============================================================
# Median
# ============================================================

def median_attack(image_path,image_name,output_dir,kernel_size):
    attacked_name = (f"MF_k{kernel_size}_{image_name}")
    wImage = load_image_ffmpeg(image_path)
    wImage = wImage.copy()

    im_median = cv2.medianBlur(wImage, kernel_size)
    out_path = os.path.join(output_dir,attacked_name)
    save_image_ffmpeg(im_median, out_path)
    return out_path

# ============================================================
# Motion Blur
# ============================================================

def motion_blur_attack(image_path,image_name,output_dir,kernel_size=3):
    attacked_name = (f"MB_k{kernel_size}_{image_name}")
    wImage = load_image_ffmpeg(image_path)
    wImage = wImage.copy()
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2),:] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    im_motion = cv2.filter2D(wImage,-1,kernel)

    out_path = os.path.join(output_dir,attacked_name)
    save_image_ffmpeg(im_motion,out_path)
    return out_path

# ============================================================
# Sharpen
# ============================================================

def sharpen_attack(image_path,image_name,output_dir):
    attacked_name = ("SHARP_" + image_name)
    wImage = load_image_ffmpeg(image_path)
    wImage = wImage.copy()

    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])

    sharpened = cv2.filter2D(wImage,-1,kernel)
    out_path = os.path.join(output_dir,attacked_name)
    save_image_ffmpeg(sharpened,out_path)
    return out_path


# ============================================================
# Resize
# ============================================================

def resize_attack(image_path,image_name,output_dir,scale_percent):
    attacked_name = (f"RESIZE_{scale_percent}_{image_name}")
    wImage = load_image_ffmpeg(image_path)
    wImage = wImage.copy()
    height, width = wImage.shape[:2]
    new_width = max(1,int(width * scale_percent / 100))
    new_height = max(1,int(height * scale_percent / 100))

    downscaled = cv2.resize(wImage,(new_width, new_height),interpolation=cv2.INTER_LANCZOS4)
    upscaled = cv2.resize(downscaled,(width, height),interpolation=cv2.INTER_LANCZOS4)
    out_path = os.path.join(output_dir,attacked_name)
    save_image_ffmpeg(upscaled,out_path)
    return out_path

# ============================================================
# Rotation
# ============================================================

def rotation_attack(image_path,image_name,output_dir,angle):
    attacked_name = (f"ROT_{angle}_{image_name}")
    wImage = load_image_ffmpeg(image_path)
    wImage = wImage.copy()
    height, width = wImage.shape[:2]
    center = (width // 2,height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center,angle,1.0)
    rotated = cv2.warpAffine(wImage,rotation_matrix,(width, height),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REFLECT)

    out_path = os.path.join(output_dir,attacked_name)
    save_image_ffmpeg(rotated,out_path)
    return out_path

# ============================================================
# Horizontal Flip
# ============================================================

def horizontal_flip_attack(image_path,image_name,output_dir):
    attacked_name = ("HFLIP_" + image_name)
    wImage = load_image_ffmpeg(image_path)
    wImage = wImage.copy()
    flipped = cv2.flip(wImage,1)
    out_path = os.path.join(output_dir,attacked_name)
    save_image_ffmpeg(flipped,out_path)
    return out_path

# ============================================================
# Translation
# ============================================================

def translation_attack(image_path,image_name,output_dir,shift_x=5,shift_y=5):
    attacked_name = (f"TRANS_x{shift_x}_y{shift_y}_{image_name}")
    wImage = load_image_ffmpeg(image_path)
    wImage = wImage.copy()
    height, width = wImage.shape[:2]
    M = np.float32([
        [1, 0, shift_x],
        [0, 1, shift_y]
    ])

    translated = cv2.warpAffine(wImage,M,(width, height),borderMode=cv2.BORDER_REFLECT)
    out_path = os.path.join(output_dir,attacked_name)
    save_image_ffmpeg(translated,out_path)
    return out_path

# ============================================================
# Main Pipeline
# ============================================================

def attack_one_image(group,image_path,image_name,attacked_root):
    image_stem = os.path.splitext(image_name)[0]
    base_output_dir = os.path.join(attacked_root,group,image_stem)
    ensure_dir(base_output_dir)

    # --------------------------------------------------------
    # Crops
    # --------------------------------------------------------

    crops_dir = os.path.join(base_output_dir,"Crops")
    ensure_dir(crops_dir)
    copy_metadata_files(image_path,crops_dir)
    for direction in ["Left","Right","Top","Bottom","BothVertical","BothHorizontal"]:
        for percentage in [25, 50, 75]:
            crop_attack(image_path,image_name,crops_dir,direction,percentage)

    # --------------------------------------------------------
    # Filters
    # --------------------------------------------------------

    filters_dir = os.path.join(base_output_dir,"Filters")
    ensure_dir(filters_dir)
    copy_metadata_files(image_path,filters_dir)
    #Salt & Pepper
    for noise_prob in [0.05, 0.1, 0.2]:
        salt_pepper_attack(image_path,image_name,filters_dir,noise_prob)

    #Gaussina Noise
    for noise_level in [0.05, 0.1, 0.2]:
        gaussian_noise_attack(image_path,image_name,filters_dir,noise_level)

    #Compression
    for quality in [70, 80, 90]:
        compression_attack(image_path,image_name,filters_dir,quality)

    #Gaussian Blur
    for kernel_size in [3, 5, 7]:
        gaussian_blur_attack(image_path,image_name,filters_dir,kernel_size)
        
    for kernel_size in [3,5]:
        mean_filter_attack(image_path,image_name,filters_dir,kernel_size)

    #Median Filter
    for kernel_size in [3, 5, 7]:
        median_attack(image_path,image_name,filters_dir,kernel_size)

    #HEQ
    histogram_equalization_attack(image_path,image_name,filters_dir)
    
    #Gamma
    gamma_attack(image_path,image_name,filters_dir)
    
    #Motion Blur
    for kernel_size in [3, 5, 7]:
        motion_blur_attack(image_path,image_name,filters_dir,kernel_size)
    
    #Sharpening Filter
    sharpen_attack(image_path,image_name,filters_dir)

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    resize_dir = os.path.join(base_output_dir,"Resize")
    ensure_dir(resize_dir)
    copy_metadata_files(image_path,resize_dir)
    for scale_percent in [90, 110]:
        resize_attack(image_path,image_name,resize_dir,scale_percent)

    # --------------------------------------------------------
    # Geometric
    # --------------------------------------------------------

    geometric_dir = os.path.join(base_output_dir,"Geometric")
    ensure_dir(geometric_dir)
    copy_metadata_files(image_path,geometric_dir)
    for angle in [5, 10, -10]:
        rotation_attack(image_path,image_name,geometric_dir,angle)

    for shift in [2, 4]:
        translation_attack(image_path,image_name,geometric_dir,shift,shift)

    horizontal_flip_attack(image_path,image_name,geometric_dir)

def main():

    project_root = os.getcwd()
    watermarked_root = os.path.join(project_root,"watermarked")
    attacked_root = os.path.join(project_root,"attacked")
    ensure_dir(attacked_root)
    images = get_watermarked_images(watermarked_root)

    print(f"Found {len(images)} watermarked images")
    for idx, (group,image_path,image_name) in enumerate(images, start=1):
        print(f"[{idx}/{len(images)}] "f"Attacking {image_path}")
        try:
            attack_one_image(group,image_path,image_name,attacked_root)
        except Exception as e:
            print(f"FAILED: {e}")

    print("\nDone.")
    print(f"Attacked images saved in: "f"{attacked_root}")

if __name__ == "__main__":
    main()