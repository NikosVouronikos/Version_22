from PIL import Image
import os

# Helping script to resize images -> currently for 1920x1080p images
# Author: Nikolaos Vouronikos
def resize_images(input_folder, output_folder, target_width = 1920, target_height = 1080):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through each file in the input folder
    for filename in os.listdir(input_folder):
        try:
            # Open image file
            with Image.open(os.path.join(input_folder, filename)) as img:
                # Resize image
                resized_img = img.resize((target_width, target_height), Image.LANCZOS)
                # Save resized image to output folder
                resized_img.save(os.path.join(output_folder, filename))
        except Exception as e:
            print(f"Error resizing {filename}: {e}")

if __name__ == "__main__":
    input_folder = "00"
    output_folder = "testImages"
    resize_images(input_folder, output_folder)