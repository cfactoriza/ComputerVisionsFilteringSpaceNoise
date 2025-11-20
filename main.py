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

    data_clipped = np.clip(data, vmin, vmax)
    norm = (data_clipped - vmin) / (vmax - vmin)
    img_uint8 = (norm * 255.0).astype(np.uint8)

    return img_uint8, vmin, vmax, header

def uint8_to_fits_data(img_uint8, vmin, vmax):
    norm = img_uint8.astype(np.float32) / 255.0
    data = norm * (vmax - vmin) + vmin
    return data.astype(np.float32)

def gaussian_filter_fast(img):
    ksize = (7, 7)
    sigma = 1.5
    if USE_CUDA:
        gpu = cv2.cuda_GpuMat()
        gpu.upload(img)
        gpu_blur = cv2.cuda.GaussianBlur(gpu, ksize, sigma)
        return gpu_blur.download()
    else:
        return cv2.GaussianBlur(img, ksize, sigma)

def bilateral_filter_fast(img):
    d = 7
    sigma_color = 50
    sigma_space = 30
    if USE_CUDA:
        gpu = cv2.cuda_GpuMat()
        gpu.upload(img)
        gpu_bilat = cv2.cuda.bilateralFilter(gpu, d, sigma_color, sigma_space)
        return gpu_bilat.download()
    else:
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

def nlm_filter_fast(img):
    h = 10
    template_window = 7
    search_window = 21

    if USE_CUDA and hasattr(cv2.cuda, "fastNlMeansDenoising"):
        gpu = cv2.cuda_GpuMat()
        gpu.upload(img)
        gpu_nlm = cv2.cuda.fastNlMeansDenoising(gpu, None, h, search_window, template_window)
        return gpu_nlm.download()
    else:
        return cv2.fastNlMeansDenoising(img, None, h, template_window, search_window)

def process_fits_image(fits_path, out_dir):
    base = os.path.splitext(os.path.basename(fits_path))[0]
    img_uint8, vmin, vmax, header = load_fits_as_uint8_with_meta(fits_path)

    ensure_dir(out_dir)

    cv2.imwrite(os.path.join(out_dir, f"{base}_original.png"), img_uint8)

    img_gauss = gaussian_filter_fast(img_uint8)
    cv2.imwrite(os.path.join(out_dir, f"{base}_gaussian.png"), img_gauss)

    img_bilat = bilateral_filter_fast(img_uint8)
    cv2.imwrite(os.path.join(out_dir, f"{base}_bilateral.png"), img_bilat)

    img_nlm = nlm_filter_fast(img_uint8)
    cv2.imwrite(os.path.join(out_dir, f"{base}_nlm.png"), img_nlm)

    data_orig = uint8_to_fits_data(img_uint8, vmin, vmax)
    data_gauss = uint8_to_fits_data(img_gauss, vmin, vmax)
    data_bilat = uint8_to_fits_data(img_bilat, vmin, vmax)
    data_nlm = uint8_to_fits_data(img_nlm, vmin, vmax)

    fits.writeto(os.path.join(out_dir, f"{base}_original_scaled.fits"), data_orig, header=header, overwrite=True)
    fits.writeto(os.path.join(out_dir, f"{base}_gaussian.fits"), data_gauss, header=header, overwrite=True)
    fits.writeto(os.path.join(out_dir, f"{base}_bilateral.fits"), data_bilat, header=header, overwrite=True)
    fits.writeto(os.path.join(out_dir, f"{base}_nlm.fits"), data_nlm, header=header, overwrite=True)

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, "data")
    results_dir = os.path.join(project_root, "results")

    ensure_dir(results_dir)

    fits_files = [f for f in os.listdir(data_dir) if f.lower().endswith((".fits", ".fit"))]

    if not fits_files:
        raise RuntimeError("No FITS files found in the 'data' folder.")

    for fname in fits_files:
        fits_path = os.path.join(data_dir, fname)
        process_fits_image(fits_path, results_dir)

if __name__ == "__main__":
    main()

