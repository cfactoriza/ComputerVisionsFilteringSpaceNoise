
import cv2
import numpy as np
import math
import sys
 
def linear_local_filtering(
    img: np.uint8,
    filter_weights: np.ndarray,
) -> np.uint8:


    img = img / 255
    img = img.astype("float32") # input image
    img_filtered = np.zeros(img.shape) # Placeholder of the filtered image
    kernel_size = filter_weights.shape[0] # filter kernel size
    sizeX, sizeY = img.shape

    for i in range(kernel_size // 2, sizeX - kernel_size // 2):
        for j in range(kernel_size // 2, sizeY - kernel_size // 2):
            window = img[
                i - kernel_size // 2: i + kernel_size // 2 + 1,
                j - kernel_size // 2: j + kernel_size // 2 + 1,
            ]
            img_filtered[i, j] = np.sum(window * filter_weights)
    img_filtered = img_filtered * 255
    img_filtered = np.uint8(img_filtered)
    return img_filtered

def gauss_kernel_generator(kernel_size: int, spatial_variance: float) -> np.ndarray:
    kernel_weights = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    for k in range(kernel_size):
        for l in range(kernel_size):
            x = k - center
            y = l - center
            kernel_weights[k, l] = (
                    1.0 / (2.0 * math.pi * spatial_variance)
                    * math.exp(-(x ** 2 + y ** 2) / (2.0 * spatial_variance))
            )
    return kernel_weights
 
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
    
    # mean filtering
    box_filter = np.ones((7, 7))/49
    img_avg = linear_local_filtering(img_noise, box_filter) # apply the filter to process the image: img_noise
    cv2.imwrite('results/im_box.png', img_avg)

    # Gaussian filtering
    kernel_size = 7  
    spatial_var = 15 # sigma_s^2 
    gaussian_filter = gauss_kernel_generator(kernel_size, spatial_var)
    gaussian_filter_normlized = gaussian_filter / (np.sum(gaussian_filter)+1e-16) # normalization term
    im_g = linear_local_filtering(img_noise, gaussian_filter_normlized) # apply the filter to process the image: img_noise
    cv2.imwrite('results/im_gaussian.png', im_g)