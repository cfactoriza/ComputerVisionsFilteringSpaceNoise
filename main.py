import os
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
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

def load_fits_as_uint8_with_meta(path):
    hdul = fits.open(path)
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

def uint8_to_fits(img, vmin, vmax):
    norm = img.astype(np.float32) / 255.0
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

def compute_stats(img_fits):
    return (
        float(np.mean(img_fits)),
        float(np.std(img_fits)),
        float(np.min(img_fits)),
        float(np.max(img_fits))
    )

def process_fits(path, out_dir, writer):
    base = os.path.splitext(os.path.basename(path))[0]
    img, vmin, vmax, header = load_fits_as_uint8_with_meta(path)

    gaussian_kernels = [7, 11, 15]
    gaussian_sigmas = [2.0, 5.0, 8.0]

    bilateral_kernels = [7, 11, 15]
    bilateral_sigmas = [30.0, 75.0, 150.0]

    nlm_patch_sizes = [3, 5, 7]
    nlm_h_values = [10.0, 20.0, 30.0]

    for k in gaussian_kernels:
        for s in gaussian_sigmas:
            out = gaussian_filter(img, k, s)
            fits_img = uint8_to_fits(out, vmin, vmax)
            mean, std, mn, mx = compute_stats(fits_img)
            fits.writeto(os.path.join(out_dir, f"{base}_gaussian_k{k}_s{s}.fits"), fits_img, header, overwrite=True)
            writer.writerow([base, "gaussian", f"k={k}", f"sigma={s}", mean, std, mn, mx])

    for k in bilateral_kernels:
        for s in bilateral_sigmas:
            out = bilateral_filter(img, k, s, s)
            fits_img = uint8_to_fits(out, vmin, vmax)
            mean, std, mn, mx = compute_stats(fits_img)
            fits.writeto(os.path.join(out_dir, f"{base}_bilateral_k{k}_s{s}.fits"), fits_img, header, overwrite=True)
            writer.writerow([base, "bilateral", f"k={k}", f"sigma={s}", mean, std, mn, mx])

    for p in nlm_patch_sizes:
        for h in nlm_h_values:
            search = p * 5 + 1
            out = nlm_filter(img, h, p, search)
            fits_img = uint8_to_fits(out, vmin, vmax)
            mean, std, mn, mx = compute_stats(fits_img)
            fits.writeto(os.path.join(out_dir, f"{base}_nlm_p{p}_h{h}.fits"), fits_img, header, overwrite=True)
            writer.writerow([base, "nlm", f"patch={p}", f"h={h}", mean, std, mn, mx])

def plot_results(csv_path, plot_dir):
    data = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = (r["filter"], r["param1"])
            data.setdefault(key, []).append((float(r["param2"].split("=")[1]), float(r["stddev"])))

    ensure_dir(plot_dir)

    for (filt, param1), values in data.items():
        values.sort()
        x, y = zip(*values)
        plt.plot(x, y, marker="o", label=param1)

    for filt in ["gaussian", "bilateral", "nlm"]:
        plt.figure()
        for (f, p), values in data.items():
            if f == filt:
                values.sort()
                x, y = zip(*values)
                plt.plot(x, y, marker="o", label=p)
        plt.xlabel("Filter Strength")
        plt.ylabel("Standard Deviation")
        plt.title(f"{filt.capitalize()} Noise Reduction")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{filt}_stddev.png"))
        plt.close()

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root, "data")
    results_dir = os.path.join(root, "results")
    plot_dir = os.path.join(results_dir, "plots")
    ensure_dir(results_dir)

    csv_path = os.path.join(results_dir, "results_statistics.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "filter", "param1", "param2", "mean", "stddev", "min", "max"])

        for file in os.listdir(data_dir):
            if file.lower().endswith((".fits", ".fit")):
                process_fits(os.path.join(data_dir, file), results_dir, writer)

    plot_results(csv_path, plot_dir)

if __name__ == "__main__":
    main()
