from skimage.metrics import structural_similarity as ssim
from math import log10, sqrt
from PIL import Image
import numpy as np

def sip_to_bits(sip,bits_len):
    sip_bits = []
    for val in sip:
        bin_value = format(val, f'0{bits_len}b')
        sip_bits.append(bin_value)
    return [bit for s in sip_bits for bit in s]


def SIP_to_BER(initial_sip, extracted_sip):
    initial_sip_bits = sip_to_bits(initial_sip,4)
    extracted_sip_bits = sip_to_bits(extracted_sip,4)

    error_bits = 0

    for i in range(len(initial_sip_bits)):
        if(initial_sip_bits[i] != extracted_sip_bits[i]):
            error_bits += 1
    
    return error_bits

def _to_array(x):
    if isinstance(x, np.ndarray):
        return x
    return np.array(Image.open(x))

def PSNR(original, compressed):
    original = _to_array(original)
    compressed = _to_array(compressed)

    mse = np.mean((original.astype(np.float64) - compressed.astype(np.float64)) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    return 20 * log10(max_pixel / sqrt(mse))

def SSIM(img1, img2):
    image1 = _to_array(img1)
    image2 = _to_array(img2)

    return ssim(
        image1,
        image2,
        data_range=image2.max() - image2.min(),
        channel_axis=-1
    )