import numpy as np, cv2, sys, os
from PIL import Image

# Bluring filters: https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
# Median filter: https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaa7c0f8b1d3e2c5f6b9d8c2f3e5f6b6b7
# Gaussian noise: https://docs.opencv.org/4.x/d6/dc7/group__imgproc__filter.html#gaa7c0f8b1d3e2c5f6b9d8c2f3e5f6b6b7
# Salt and pepper noise: https://docs.opencv.org/4.x/d6/dc7/group__imgproc__filter.html#gaa7c0f8b1d3e2c5f6b9d8c2f3e5f6b6b7
# Histogram equalization: https://docs.opencv.org/4.x/d5/db5/tutorial_py_histogram_equalization.html
# Gamma correction: https://docs.opencv.org/4.x/d3/dc1/tutorial_basic_linear_transform.html
# Motion blur: https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
# Fourier transform: https://docs.opencv.org/4.x/d3/dc3/tutorial_py_fft.html
# Image resizing: https://docs.opencv.org/4.x/d4/d61/tutorial_warp_affine.html
# Image rotation: https://docs.opencv.org/4.x/d4/d61/tutorial_warp_affine.html
# Image sharpening: https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html

# Author: Nikolaos Vouronikos
# Description: Used for attack scenarios - Testing Embed Algorithm

EXTENSION_TO_FORMAT = {
    "jpg": "JPEG",
    "jpeg": "JPEG",
    "png": "PNG",
    "bmp": "BMP",
    "tiff": "TIFF",
    "webp": "WEBP"
}

# Author: Nikolaos Vouronikos
# Description: Function to attack watermarked image with various filters
def attackWatermarkedImageWithFilters(watermarkedImage, watermarkedImageName, mode, extension):
    filter_functions = {
        "Compression": lambda: compressWatermarkedImage(watermarkedImage, watermarkedImageName, extension),
        "Gaussian Blur": lambda: gaussianBlurWatermarkedImage(watermarkedImage, watermarkedImageName),
        "Gaussian Noise": lambda: gaussianNoiseWatermarkedImage(watermarkedImage, watermarkedImageName),
        "Salt and Pepper": lambda: saltAndPepperWatermarkedImage(watermarkedImage, watermarkedImageName),
        "Histogram Equalization": lambda: histogramEQWatermarkedImage(watermarkedImage, watermarkedImageName),
        "Gamma": lambda: gammaWatermarkedImage(watermarkedImage, watermarkedImageName),
        "Median": lambda: medianFilterWatermarkedImage(watermarkedImage, watermarkedImageName),
        "Motion Blur": lambda: motionBlurWatermarkedImage(watermarkedImage, watermarkedImageName),
        "Sharpening": lambda: sharpenWatermarkedImage(watermarkedImage, watermarkedImageName),
    }
    return filter_functions[mode]()

# Author: Nikolaos Vouronikos
# Description: Function to attack watermarked image with crops
# direction: Left, Right, Top, Bottom, BothVertical, BothHorizontal
# percentage: 25, 50, 75
def attackWatermarkedImageWithCrops(watermarkedImage, watermarkedImageName, direction, percentage):
	input_dir = os.path.dirname(watermarkedImage)
	wImage = np.array(Image.open(watermarkedImage))
	w,h = wImage.shape[1],wImage.shape[0]
	croppedImageName = f"Croped_{direction}_{percentage}_{watermarkedImageName}"
 
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
		exit(1)

	croppedImage = Image.fromarray(wImage)
	save_path = os.path.join(input_dir, croppedImageName)
	croppedImage.save(save_path, quality = 100)
	return croppedImage, croppedImageName

#Author: Nikolaos Vouronikos
# Description: Function to compress watermarked image by changing quality and subsampling
def compressWatermarkedImage(watermarkedImage, watermarkedImageName, extension):
	# Compress by changing quality and subsampling
	# Current settings are 65% quality and 4:2:0 subsampling (really heavy compression)
	image_format = EXTENSION_TO_FORMAT.get(extension)
	if(image_format == 'PNG'): # png is not affected by quality decrease and subsampling
		return '',''
	wImage = Image.open(watermarkedImage)
	compressedImageName = "Compressed_" + watermarkedImageName
	input_dir = os.path.dirname(watermarkedImage)
	save_path = os.path.join(input_dir, compressedImageName)
	wImage.save(save_path, format = image_format, quality = 65, subsampling = 1)
	compressedImage = Image.open(save_path)
	return compressedImage,compressedImageName

# Author: Nikolaos Vouronikos
# Description: Function to apply Gaussian noise to watermarked image
def gaussianNoiseWatermarkedImage(watermarkedImage, watermarkedImageName):
	std = 15.5
	gnName = "GN_" + watermarkedImageName
	wImage = np.array(Image.open(watermarkedImage))
	input_dir = os.path.dirname(watermarkedImage)
	save_path = os.path.join(input_dir, gnName)
	w,h = wImage.shape[0],wImage.shape[1]
	gauss_noise = np.zeros((w,h,3), dtype = np.uint8)
	cv2.randn(gauss_noise, 0, std)
	gauss_noise = (gauss_noise*0.5).astype(np.uint8)
	noisy = cv2.add(wImage, gauss_noise)
	noisy_img = Image.fromarray(noisy)
	noisy_img.save(save_path, quality = 100, subsampling = 1)
	gnImage = Image.open(save_path)
	return gnImage,gnName

# Author: Nikolaos Vouronikos
# Description: Function to apply salt and pepper noise to watermarked image
# salt and pepper noise is a form of noise that affects the image by randomly replacing pixels with either black or white
def saltAndPepperWatermarkedImage(watermarkedImage, watermarkedImageName):
	sapName = "SaP_" + watermarkedImageName
	input_dir = os.path.dirname(watermarkedImage)
	save_path = os.path.join(input_dir, sapName)
	wImage = np.array(Image.open(watermarkedImage))
	w,h = wImage.shape[0],wImage.shape[1]
	for i in range(w):
		for j in range(h):
			random_num_1 = np.random.uniform(low = 0.0, high = 1.0)
			random_num_2 = np.random.uniform(low = 0.0, high = 1.0)
			if(random_num_1 < 0.05):
				wImage[i,j] = 255
			elif(random_num_2 < 0.05):
				wImage[i,j] = 0
			else:
				pass
	sap = Image.fromarray(wImage)
	sap.save(save_path, quality = 100, subsampling = 1)
	sapImage = Image.open(save_path)
	return sapImage,sapName

# Author: Nikolaos Vouronikos
# Description: Function to apply Gaussian blur to watermarked image
def gaussianBlurWatermarkedImage(watermarkedImage, watermarkedImageName):
	gbName = "GB_" + watermarkedImageName
	input_dir = os.path.dirname(watermarkedImage)
	save_path = os.path.join(input_dir, gbName)
	wImage = np.array(Image.open(watermarkedImage))
	imblur = cv2.GaussianBlur(wImage, (5, 5), 0)
	imblur = Image.fromarray(imblur)
	imblur.save(save_path, quality = 100, subsampling = 1)
	gbImage = Image.open(save_path)
	return gbImage,gbName

# Author: Nikolaos Vouronikos
# Description: Function to apply histogram equalization to watermarked image
# Note: This function converts the image to YCrCb color space, applies histogram equalization to the Y channel, and then converts it back to BGR.
def histogramEQWatermarkedImage(watermarkedImage, watermarkedImageName):
	heqName = "HEQ_" + watermarkedImageName
	input_dir = os.path.dirname(watermarkedImage)
	save_path = os.path.join(input_dir, heqName)
	wImage = np.array(Image.open(watermarkedImage))
	img = cv2.cvtColor(wImage, cv2.COLOR_BGR2YCrCb)
	y, cr, cb = cv2.split(img)
	y_eq = cv2.equalizeHist(y)
	img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
	img_heq = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2BGR)
	img_heq = Image.fromarray(img_heq)
	img_heq.save(save_path, quality = 100, subsampling = 1)
	heqImage = Image.open(save_path)
	return heqImage,heqName

# Author: Nikolaos Vouronikos
# Description: Function to apply gamma correction to watermarked image
# Note: Gamma correction is a nonlinear operation used to encode and decode luminance or tristimulus values in images.
def gammaWatermarkedImage(watermarkedImage, watermarkedImageName):
	# gamma = 1.625
	gammaName = "Gamma_" + watermarkedImageName
	input_dir = os.path.dirname(watermarkedImage)
	save_path = os.path.join(input_dir, gammaName)
	wImage = np.array(Image.open(watermarkedImage))
	image_gamma = cv2.filter2D(wImage, -1, (1.625))
	image_gamma = Image.fromarray(image_gamma)
	image_gamma.save(save_path, quality = 100, subsampling = 1)
	gammaImage = Image.open(save_path)
	return gammaImage,gammaName

# Author: Nikolaos Vouronikos
# Description: Function to apply median filter to watermarked image
# Note: Median filtering is a non-linear digital filtering technique, often used to remove noise from an image.
def medianFilterWatermarkedImage(watermarkedImage, watermarkedImageName):
    mfName = "MF_" + watermarkedImageName
    input_dir = os.path.dirname(watermarkedImage)
    save_path = os.path.join(input_dir, mfName)
    wImage = np.array(Image.open(watermarkedImage))
    # Apply median filter (kernel size should be odd and >1)
    im_median = cv2.medianBlur(wImage, 21) # heavy filtering
    im_median = Image.fromarray(im_median)
    im_median.save(save_path, quality = 100)
    mfImage = Image.open(save_path)
    return mfImage, mfName

# Author: Nikolaos Vouronikos
# Description: Function to apply motion blur to watermarked image
# Note: Motion blur is a common effect in photography and video, where the image appears blurred in the direction of motion.
def motionBlurWatermarkedImage(watermarkedImage, watermarkedImageName):
	kernel_size = 25
	mbName = f"MB_k{kernel_size}_" + watermarkedImageName
	input_dir = os.path.dirname(watermarkedImage)
	save_path = os.path.join(input_dir, mbName)
	wImage = np.array(Image.open(watermarkedImage))
	kernel = np.zeros((kernel_size, kernel_size))
	kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
	kernel = kernel / kernel_size
	im_motion = cv2.filter2D(wImage, -1, kernel)
	im_motion = Image.fromarray(im_motion)
	im_motion.save(save_path, quality = 100)
	mbImage = Image.open(save_path)
	return mbImage, mbName

#Author: Nikolaos Vouronikos
# Description: Function to sharpen watermarked image using a kernel
# Note: Sharpening enhances the edges and fine details in an image.
def sharpenWatermarkedImage(watermarkedImage, watermarkedImageName):
    sharpName = "SHARP_" + watermarkedImageName
    input_dir = os.path.dirname(watermarkedImage)
    save_path = os.path.join(input_dir, sharpName)
    wImage = np.array(Image.open(watermarkedImage))
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])

    sharpened = cv2.filter2D(wImage, -1, kernel)
    sharpened = Image.fromarray(sharpened)
    sharpened.save(save_path, quality = 100)
    sharpImage = Image.open(save_path)
    return sharpImage, sharpName

# Author: Nikolaos Vouronikos
# Description: Function to resize watermarked image by downscaling and then upscaling
# scale_percent: 25, 50, 100 (downscale then upscale to original dimensions)
def resizeAttackWatermarkedImage(watermarkedImage, watermarkedImageName, scale_percent):
	# Downscale (25, 50, 100 %) then Upscale to original dimensions
	resizeName = f"RESIZE_{scale_percent}_" + watermarkedImageName
	input_dir = os.path.dirname(watermarkedImage)
	save_path = os.path.join(input_dir, resizeName)
	wImage = np.array(Image.open(watermarkedImage))
	height, width = wImage.shape[:2]
	new_width = int(width * scale_percent / 100)
	new_height = int(height * scale_percent / 100)
	downscaled = cv2.resize(wImage, (new_width, new_height), interpolation=cv2.INTER_AREA)
	upscaled = cv2.resize(downscaled, (width, height), interpolation=cv2.INTER_LINEAR)
	resized = Image.fromarray(upscaled)
	resized.save(save_path, quality = 100)
	resizeImage = Image.open(save_path)
	return resizeImage, resizeName

# Author: Nikolaos Vouronikos
# Description: Function to rotate watermarked image by a given angle
# angle: 45, 90, 180 degrees
# Note: The rotation is performed around the center of the image.
def rotationAttackWatermarkedImage(watermarkedImage, watermarkedImageName, angle):
    rotateName = f"ROT_{angle}_" + watermarkedImageName
    input_dir = os.path.dirname(watermarkedImage)
    save_path = os.path.join(input_dir, rotateName)
    # Load image
    wImage = np.array(Image.open(watermarkedImage))
    height, width = wImage.shape[:2]
    center = (width // 2, height // 2)
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Apply rotation
    rotated = cv2.warpAffine(wImage, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    rotated = Image.fromarray(rotated)
    rotated.save(save_path, quality=100, subsampling=1)
    rotateImage = Image.open(save_path)
    return rotateImage, rotateName

# Author: Nikolaos Vouronikos
# Description: Function to apply geometric distortion attack on watermarked image
# Note: This function applies an affine transformation to the image, distorting it geometrically.
def geometricDistortionAttack(watermarkedImage, watermarkedImageName):
    warpName = "WARP_" + watermarkedImageName
    input_dir = os.path.dirname(watermarkedImage)
    save_path = os.path.join(input_dir, warpName)
    # Load image
    wImage = np.array(Image.open(watermarkedImage))
    height, width = wImage.shape[:2]
    # Define source and destination points for affine transformation
    src_pts = np.float32([[0, 0], [width - 1, 0], [0, height - 1]])
    dst_pts = np.float32([[0, 0], [int(0.9 * (width - 1)), int(0.1 * height)], [int(0.1 * width), int(0.9 * (height - 1))]])
    # Compute affine transform matrix
    M = cv2.getAffineTransform(src_pts, dst_pts)
    # Apply warp
    warped = cv2.warpAffine(wImage, M, (width, height), borderMode=cv2.BORDER_REFLECT)
    warped = Image.fromarray(warped)
    warped.save(save_path, quality = 100)
    warpImage = Image.open(save_path)
    return warpImage, warpName

# Author: Nikolaos Vouronikos
# Description: Function to apply Fourier random noise attack on RGB watermarked image
# noise_level: 0.1, 0.2 (mild and heavy attack in frequency domain)
# Note: This function applies random noise to the magnitude spectrum of the Fourier transform of each RGB channel.
def fourierRandomNoiseAttackRGB(watermarkedImage, watermarkedImageName, noise_level):
    frName = f"FT_RAND_NOISE_{noise_level}_" + watermarkedImageName
    input_dir = os.path.dirname(watermarkedImage)
    save_path = os.path.join(input_dir, frName)
    img = cv2.imread(watermarkedImage)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    noisy_img = np.zeros_like(img)

    for c in range(3):
        channel = img[:, :, c]
        # Forward FFT
        dft = np.fft.fft2(channel)
        dft_shift = np.fft.fftshift(dft)
        # Magnitude and phase
        magnitude = np.abs(dft_shift)
        phase = np.angle(dft_shift)
        # Random frequency noise
        noise = noise_level * np.random.normal(loc=0.0, scale=1.0, size=channel.shape)
        noisy_magnitude = magnitude + noise * magnitude
        # Reconstruct noisy DFT
        noisy_dft = noisy_magnitude * np.exp(1j * phase)
        back_ishift = np.fft.ifftshift(noisy_dft)
        img_back = np.fft.ifft2(back_ishift)
        img_back = np.abs(img_back).clip(0, 255).astype(np.uint8)

        noisy_img[:, :, c] = img_back

    result_img = Image.fromarray(noisy_img)
    result_img.save(save_path, quality = 100)
    return result_img, frName

# Author: Nikolaos Vouronikos
# Description: Function to apply phase spectrum attack on RGB watermarked image
# phase_noise_level: 0.3, 0.5, 1.0 (mild, heavy and maximum attack in phase spectrum)
# Note: This function adds random noise to the phase spectrum of the Fourier transform of each RGB channel.	
def phaseSpectrumAttackRGB(watermarkedImage, watermarkedImageName, phase_noise_level):
    phName = f"PHASE_NOISE_{phase_noise_level}_" + watermarkedImageName
    input_dir = os.path.dirname(watermarkedImage)
    save_path = os.path.join(input_dir, phName)
    img = cv2.imread(watermarkedImage)
    if img is None:
        raise ValueError("Image not found")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    attacked_img = np.zeros_like(img)
    for c in range(3):
        channel = img[:, :, c].astype(np.float32)
        # Fourier transform
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)
        # Magnitude and phase
        magnitude = np.abs(fshift)
        phase = np.angle(fshift)
        # Add random noise to phase
        noise = np.random.uniform(-np.pi, np.pi, phase.shape) * phase_noise_level
        noisy_phase = phase + noise
        # Recombine and inverse FFT
        f_noisy = magnitude * np.exp(1j * noisy_phase)
        f_ishift = np.fft.ifftshift(f_noisy)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back).clip(0, 255).astype(np.uint8)

        attacked_img[:, :, c] = img_back

    result_img = Image.fromarray(attacked_img)
    result_img.save(save_path, quality = 100)
    return result_img, phName

# Author: Nikolaos Vouronikos
# Description: Function to convert between PNG and JPEG formats
# PNG → JPEG (lossy) or JPEG → PNG (freeze JPEG damage)
def formatConversionAttack(imagePath, imageName, extension):
    input_dir = os.path.dirname(imagePath)
    base_name = os.path.splitext(imageName)[0]
    if extension == 'png':
        # Convert PNG → JPEG (lossy)
        converted_name = f"JPEG_q_" + base_name + ".jpg"
        save_path = os.path.join(input_dir, converted_name)
        img = Image.open(imagePath)
        img.save(save_path, format = "JPEG", quality = 100)
        return save_path, converted_name

    elif extension == 'jpg' or extension == 'jpeg':
        # Convert JPEG → PNG (freeze JPEG damage)
        converted_name = "PNG_" + base_name + ".png"
        save_path = os.path.join(input_dir, converted_name)
        img = Image.open(imagePath)
        img.save(save_path, format="PNG")
        return save_path, converted_name

if __name__ == '__main__':
	# python3 attacker.py watermarked/watermarked_image_1/watermarked_image_1.jpg
	imagePath = sys.argv[1]
	extension = os.path.splitext(imagePath)[1][1:].lower()
	imageName = os.path.basename(imagePath)

	###################### FIRST ATTACK SCENARIO - CROP ATTACKS ######################
	for direction in ["Left", "Right", "Top", "Bottom", "BothVertical", "BothHorizontal"]:
		for percentage in [25, 50, 75]:
			croppedImage, croppedImageName = attackWatermarkedImageWithCrops(imagePath, imageName, direction, percentage)

	###################### SECOND ATTACK SCENARIO - FILTER ATTACKS ######################
	for mode in ["Compression", "Gaussian Blur", "Gaussian Noise", "Salt and Pepper", "Histogram Equalization", "Gamma", "Median", "Motion Blur", "Sharpening"]:
		attackedImage, attackedImageName = attackWatermarkedImageWithFilters(imagePath, imageName, mode, extension)

	###################### THIRD ATTACK SCENARIO - RESIZE ATTACKS ######################
	for scale_percent in [25, 50, 100]: # Downscale 25, 50 and 100% then upscale to original dimensions
		attackedImage, attackedImageName = resizeAttackWatermarkedImage(imagePath, imageName, scale_percent)

	###################### FOURTH ATTACK SCENARIO - GEOMETRIC ATTACKS ######################
	attackedImage, attackedImageName = geometricDistortionAttack(imagePath, imageName)
	for angle in [45, 90, 180]: # Rotate 45, 90 and 180 degrees
		attackedImage, attackedImageName = rotationAttackWatermarkedImage(imagePath, imageName, angle)

	###################### FIFTH ATTACK SCENARIO - FREQUENCY DOMAIN ATTACKS ######################
	for noise_level in [0.1, 0.2]: # mild and heavy attack in frequency domain
		attackedImage, attackedImageName = fourierRandomNoiseAttackRGB(imagePath, imageName, noise_level)

	for phase_noise_level in [0.3, 0.5, 1.0]: # mild, heavy and maximum (in phase spectrum)
		attackedImage, attackedImageName = phaseSpectrumAttackRGB(imagePath, imageName, phase_noise_level)

	###################### SIXTH ATTACK SCENARIO - JPG -> PNG / PNG -> JPG ######################
	save_path, converted_name = formatConversionAttack(imagePath, imageName, extension)

	# For validation run -> python3 validator.py watermarked/watermarked_image_1/Compressed_watermarked_image_1.jpg 56728192afd67fca