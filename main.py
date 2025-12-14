import os
import numpy as np
import cv2
from astropy.io import fits

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def has_cuda():
    try:
        return hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except:
        return False

USE_CUDA = has_cuda()

def load_fits_as_uint8_with_meta(fits_path):
    hdul = fits.open(fits_path)
    data = hdul[0].data
    header = hdul[0].header
    hdul.close()

    if data.ndim > 2:
        data = data[0]

    data = np.nan_to_num(data).astype(np.float32)

    vmin = np.percentile(data, 1)
    vmax = np.percentile(data, 99)
    if vmax <= vmin:
        vmax = vmin + 1.0

    data = np.clip(data, vmin, vmax)
    norm = (data - vmin) / (vmax - vmin)
    img_uint8 = (norm * 255.0).astype(np.uint8)

    return img_uint8, vmin, vmax, header

def uint8_to_fits(img_uint8, vmin, vmax):
    norm = img_uint8.astype(np.float32) / 255.0
    return (norm * (vmax - vmin) + vmin).astype(np.float32)

def gaussian_filter(img, k, sigma):
    if USE_CUDA:
        g = cv2.cuda_GpuMat()
        g.upload(img)
        out = cv2.cuda.GaussianBlur(g, (k, k), sigma)
        return out.download()
    return cv2.GaussianBlur(img, (k, k), sigma)

def bilateral_filter(img, d, sigma_color, sigma_space):
    if USE_CUDA:
        g = cv2.cuda_GpuMat()
        g.upload(img)
        out = cv2.cuda.bilateralFilter(g, d, sigma_color, sigma_space)
        return out.download()
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

def nlm_filter(img, h, template, search):
    if USE_CUDA and hasattr(cv2.cuda, "fastNlMeansDenoising"):
        g = cv2.cuda_GpuMat()
        g.upload(img)
        out = cv2.cuda.fastNlMeansDenoising(g, None, h, search, template)
        return out.download()
    return cv2.fastNlMeansDenoising(img, None, h, template, search)

def process_fits(fits_path, out_dir):
    base = os.path.splitext(os.path.basename(fits_path))[0]
    img, vmin, vmax, header = load_fits_as_uint8_with_meta(fits_path)

    gaussian_kernels = [7, 11, 15]
    gaussian_sigmas = [2.0, 5.0, 8.0]

    bilateral_kernels = [7, 11, 15]
    bilateral_sigmas = [30.0, 75.0, 150.0]

    nlm_patch_sizes = [3, 5, 7]
    nlm_h_values = [10.0, 20.0, 30.0]

    for k in gaussian_kernels:
        for s in gaussian_sigmas:
            out = gaussian_filter(img, k, s)
            name = f"{base}_gaussian_k{k}_s{s}.fits"
            fits.writeto(
                os.path.join(out_dir, name),
                uint8_to_fits(out, vmin, vmax),
                header=header,
                overwrite=True
            )

    for k in bilateral_kernels:
        for s in bilateral_sigmas:
            out = bilateral_filter(img, k, s, s)
            name = f"{base}_bilateral_k{k}_s{s}.fits"
            fits.writeto(
                os.path.join(out_dir, name),
                uint8_to_fits(out, vmin, vmax),
                header=header,
                overwrite=True
            )

    for p in nlm_patch_sizes:
        for h in nlm_h_values:
            search = p * 5 + 1
            out = nlm_filter(img, h, p, search)
            name = f"{base}_nlm_p{p}_h{h}.fits"
            fits.writeto(
                os.path.join(out_dir, name),
                uint8_to_fits(out, vmin, vmax),
                header=header,
                overwrite=True
            )

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root, "data")
    out_dir = os.path.join(root, "results")

    ensure_dir(out_dir)

    fits_files = [
        f for f in os.listdir(data_dir)
        if f.lower().endswith((".fits", ".fit"))
    ]

    if not fits_files:
        raise RuntimeError("No FITS files found")

    for f in fits_files:
        process_fits(os.path.join(data_dir, f), out_dir)

if __name__ == "__main__":
    main()
