import os
import sys

from attacker import (
    ensure_dir,
    copy_metadata_files,
    salt_pepper_attack,
    gaussian_noise_attack,
    median_attack,
    motion_blur_attack,
    resize_attack,
    rotation_attack,
    translation_attack,
    horizontal_flip_attack,
    crop_attack,
    sharpen_attack,
    histogram_equalization_attack,
    gamma_attack,
    gaussian_blur_attack,
    compression_attack
)


def main():

    if len(sys.argv) < 4:
        print(
            "Usage:\n"
            "python single_attack.py <image_path> <attack> <level>"
        )
        return

    image_path = sys.argv[1]
    attack_name = sys.argv[2]
    level = sys.argv[3]

    image_name = os.path.basename(image_path)

    output_dir = os.path.join(
        "single_attacks",
        attack_name
    )

    ensure_dir(output_dir)

    try:
        copy_metadata_files(image_path, output_dir)
    except:
        pass

    if attack_name == "compression":
        out = compression_attack(image_path, image_name, output_dir, int(level))
    elif attack_name == "median":
        out = median_attack(image_path,image_name,output_dir,int(level))
    elif attack_name == "motion":
        out = motion_blur_attack(image_path,image_name,output_dir,int(level))   
    elif attack_name == "gaussian_blur":
        out = gaussian_blur_attack(image_path,image_name,output_dir,int(level))
    elif attack_name == "noise":
        out = gaussian_noise_attack(image_path,image_name,output_dir,float(level))
    elif attack_name == "sap":
        out = salt_pepper_attack(image_path,image_name,output_dir,float(level))
    elif attack_name == "resize":
        out = resize_attack(image_path,image_name,output_dir,int(level))
    elif attack_name == "rotate":
        out = rotation_attack(image_path,image_name,output_dir,int(level))
    elif attack_name == "translate":
        out = translation_attack(image_path,image_name,output_dir,int(level),int(level))
    elif attack_name == "crop_left":
        out = crop_attack(image_path,image_name,output_dir,"Left",int(level))
    elif attack_name == "crop_right":
        out = crop_attack(image_path,image_name,output_dir,"Right",int(level))
    elif attack_name == "crop_top":
        out = crop_attack(image_path,image_name,output_dir,"Top",int(level))
    elif attack_name == "crop_bottom":
        out = crop_attack(image_path,image_name,output_dir,"Bottom",int(level))
    elif attack_name == "crop_both_vertical":
        out = crop_attack(image_path,image_name,output_dir,"BothVertical",int(level))
    elif attack_name == "crop_both_horizontal":
        out = crop_attack(image_path,image_name,output_dir,"BothHorizontal",int(level))
    elif attack_name == "hflip":
        out = horizontal_flip_attack(image_path,image_name,output_dir)
    elif attack_name == "sharp":
        out = sharpen_attack(image_path,image_name,output_dir)
    elif attack_name == "heq":
        out = histogram_equalization_attack(image_path,image_name,output_dir)
    elif attack_name == "gamma":
        out = gamma_attack(image_path,image_name,output_dir)
    else:
        raise ValueError(f"Unknown attack: {attack_name}")

    print("\nAttacked image:")
    print(out)

if __name__ == "__main__":
    main()