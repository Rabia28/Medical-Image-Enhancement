import cv2
import torch
import torch.nn.functional as F
import numpy as np
from numpy import asarray

'''
YOU CAN USE 0,1 AND 2 INSTEAD OF 3 TO TEST OTHER IMAGES PRESENT INTO THE FOLDER
'''
imagePath = '3.jpg'


def sharpening_type1(image, p=0.5):
    kernel = np.array([[-p, -p, -p], [-p, 1 + 8 * p, -p], [-p, -p, -p]], np.float32)
    image = cv2.filter2D(image, -1, kernel)
    return image


def unsharp_masking(image, alpha=1.5, beta=-0.5):
    gaussian = cv2.GaussianBlur(image, (0, 0), 10)
    unsharp_image = cv2.addWeighted(image, alpha, gaussian, beta, 0, image)
    return unsharp_image


def clahe(image, clipLimit=3.0):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    lab_planes = cv2.split(lab)
    grids = 16
    clahe_image = cv2.createCLAHE(clipLimit, tileGridSize=(grids, grids))
    lab_planes[0] = clahe_image.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    clahe_image = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return clahe_image


def clahe_and_tophat(image, clipLimit=3.0, kernel_size=3):
    clahe_image = clahe(image, clipLimit)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv2.morphologyEx(clahe_image, cv2.MORPH_OPEN, kernel)
    tophat = clahe_image - opening
    final_image = clahe_image + tophat
    return final_image


def gauss(x, y, sigma=2.0):
    Z = 2 * np.pi * sigma ** 2
    return 1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))


def get_gaussian_filter(kernel_shape):
    x = np.zeros(kernel_shape, dtype='float64')
    mid = np.floor(kernel_shape[-1] / 2.)
    for kernel_idx in range(0, kernel_shape[1]):
        for i in range(0, kernel_shape[2]):
            for j in range(0, kernel_shape[3]):
                x[0, kernel_idx, i, j] = gauss(i - mid, j - mid)
    return x / np.sum(x)


def LocalContrastNorm(image, radius=9):
    if radius % 2 == 0:
        radius += 1

    n, c, h, w = image.shape[0], image.shape[1], image.shape[2], image.shape[3]

    gaussian_filter = torch.Tensor(get_gaussian_filter((1, c, radius, radius)))
    filtered_out = F.conv2d(image, gaussian_filter, padding=radius - 1)
    mid = int(np.floor(gaussian_filter.shape[2] / 2.))
    ### Subtractive Normalization
    image = image - filtered_out[:, :, mid:-mid, mid:-mid]

    ## Variance Calc
    sum_sqr_image = F.conv2d(image.pow(2), gaussian_filter, padding=radius - 1)
    s_deviation = sum_sqr_image[:, :, mid:-mid, mid:-mid].sqrt()
    denom = sum_sqr_image[:, :, mid:-mid, mid:-mid].sqrt()
    per_img_mean = denom.mean()

    ## Divisive Normalization
    divisor = np.maximum(per_img_mean.numpy(), s_deviation.numpy())
    divisor = np.maximum(divisor, 1e-4)
    new_image = image / torch.Tensor(divisor)
    return new_image


def lcm_clahe(image):
    data = asarray(image)
    image_tensor = torch.Tensor([np.array(data).transpose((2, 0, 1))])

    ret = LocalContrastNorm(image_tensor, radius=9)
    ret = ret[0].numpy().transpose((1, 2, 0))
    scaled_ret = (ret - ret.min()) / (ret.max() - ret.min())  ## Scaled between 0 to 1 to see properly

    lab = cv2.cvtColor(scaled_ret, cv2.COLOR_BGR2Lab)
    lab_planes = cv2.split(lab)
    lab = cv2.merge(lab_planes)
    final = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return final


printsize = 600
original_image = cv2.imread(imagePath)
resized_image = cv2.resize(original_image, (printsize, printsize))

sharpened_image = sharpening_type1(original_image)
resized_sharpened = cv2.resize(sharpened_image, (printsize, printsize))

cv2.imshow("original", resized_image)
cv2.imshow("sharpened", resized_sharpened)
cv2.waitKey(0)

unsharp_image = unsharp_masking(original_image)
resized_unsharp = cv2.resize(unsharp_image, (printsize, printsize))

cv2.imshow("original", resized_image)
cv2.imshow("unsharp masking", resized_unsharp)
cv2.waitKey(0)

clahe_image = clahe(original_image)
resized_clahe = cv2.resize(clahe_image, (printsize, printsize))

cv2.imshow("original", resized_image)
cv2.imshow("clahe", resized_clahe)
cv2.waitKey(0)

clahe_and_tophat_image = clahe_and_tophat(original_image)
resized_clahe_and_tophat = cv2.resize(clahe_and_tophat_image, (printsize, printsize))

cv2.imshow("original", resized_image)
cv2.imshow("clahe and tophat", resized_clahe_and_tophat)
cv2.waitKey(0)

lcm_clahe_image = lcm_clahe(original_image)
resized_lcm_clahe = cv2.resize(lcm_clahe_image, (printsize, printsize))

cv2.imshow("original", resized_image)
cv2.imshow("lcm clahe", resized_lcm_clahe)
cv2.waitKey(0)

cv2.destroyAllWindows()
