"""
CS 4391 Homework 2 Programming: Part 4 - non-local means filter
Implement the nlm_filtering() function in this python script
"""
 
import cv2
import numpy as np
import math

def nlm_filtering(
    img: np.uint8,
    intensity_variance: float,
    patch_size: int,
    window_size: int,
) -> np.uint8:
    """
    Homework 2 Part 4
    Compute the filtered image given an input image, kernel size of image patch, spatial variance, and intensity range variance
    """

    img = img / 255
    img = img.astype("float32")
    img_filtered = np.zeros(img.shape) # Placeholder of the filtered image
    
    # Todo: For each pixel position [i, j], you need to compute the filtered output: img_filtered[i, j] using a non-local means filter
    # step 1: compute window_sizexwindow_size filter weights of the non-local means filter in terms of intensity_variance. 
    # step 2: compute the filtered pixel img_filtered[i, j] using the obtained kernel weights and the pixel values in the search window
    # Please see slides 30 and 31 of lecture 6. Clarification: the patch_size refers to the size of small image patches (image content in yellow, 
    # red, and blue boxes in the slide 30); intensity_variance denotes sigma^2 in slide 30; the window_size is the size of the search window as illustrated in slide 31.
    # Tip: use zero-padding to address the black border issue. 

    # ********************************
    # Your code is here.

    pad_patch = patch_size // 2
    pad_window = window_size // 2
    sizeX, sizeY = img.shape
    img_pad = np.pad(img, pad_patch + pad_window, mode='constant')
    for i in range(sizeX):
        for j in range(sizeY):
            i0 = i + pad_patch + pad_window
            j0 = j + pad_patch + pad_window
            patch_ref = img_pad[i0 - pad_patch:i0 + pad_patch + 1, j0 - pad_patch:j0 + pad_patch + 1]
            weights = []
            patches = []
            for m in range(i0 - pad_window, i0 + pad_window + 1):
                for n in range(j0 - pad_window, j0 + pad_window + 1):
                    patch = img_pad[m - pad_patch:m + pad_patch + 1, n - pad_patch:n + pad_patch + 1]
                    dist2 = np.sum((patch_ref - patch) ** 2)
                    weight = np.exp(-dist2 / (2 * intensity_variance))
                    weights.append(weight)
                    patches.append(img_pad[m, n])
            weights = np.array(weights)
            patches = np.array(patches)
            weights_sum = np.sum(weights) + 1e-16
            img_filtered[i, j] = np.sum(weights * patches) / weights_sum

    # ********************************

            
    img_filtered = img_filtered * 255
    img_filtered = np.uint8(img_filtered)
    return img_filtered

 
if __name__ == "__main__":
    img = cv2.imread("data/img/butterfly.jpeg", 0) # read gray image
    img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA) # reduce image size for saving your computation time
    cv2.imwrite('results/im_original.png', img) # save image 
    
    # Generate Gaussian noise
    noise = np.random.normal(0,0.6,img.size)
    noise = noise.reshape(img.shape[0],img.shape[1]).astype('uint8')
   
    # Add the generated Gaussian noise to the image
    img_noise = cv2.add(img, noise)
    cv2.imwrite('results/im_noisy.png', img_noise)
    
    # Bilateral filtering
    intensity_variance = 1
    patch_size = 5 # small image patch size
    window_size = 15 # serach window size
    img_bi = nlm_filtering(img_noise, intensity_variance, patch_size, window_size)
    cv2.imwrite('results/im_nlm.png', img_bi)