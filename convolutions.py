from skimage.exposure import rescale_intensity
import numpy as np
import argparse
from cv2 import cv2
import os


def convolve(image, kernel):
    # Take the spatial dimensions of the image and kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    # Allocate memory for the output image, taking care to "pad"
    # the borders of the input image so the spatial size are
    # not reduced
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float")

    # Loop over the input image, "sliding" the kernel across
    # each (x, y)-coordinate from left-to-right and top-to-bottom
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # Extract the ROI of the image by extracting the
            # *center* region of the current (x, y)-coordinates
            # dimensions
            ROI = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            # Perform the actual convolution by taking the
            # element-wise multiplication between the ROI and the
            # kernel, then summing the matrix
            k = (ROI * kernel).sum()

            # Store the convolved value in the output (x, y)-
            # coordinate of the output image
            output[y - pad, x - pad] = k

    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    return output


PATH = os.path.abspath(os.getcwd())
DIRECTORY = "Image"
IMAGE = "cute.jpg"

# Construct average blurring kernels used to smooth an image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

# Construct a sharpening filter
sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")

# Construct a Laplacian kernel used to detect edge-like
# region of the image
laplacian = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, -1, 0]), dtype="int")

# Construct the Sobel x-axis kernel
sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")

# Construct the Sobel Y-axis kernel
sobelY = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype="int")

# Construct the emboss kernel
emboss = np.array((
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]), dtype="int")

# Construct the kernel back, a list of kernels we're going to apply
# using both our custom 'convolve' function and OpenCV's 'filter2D'
# function
kernelBank = (
    ("small_blur", smallBlur),
    ("large_blur", largeBlur),
    ("sharpen", sharpen),
    ("laplacian", laplacian),
    ("sobel_x", sobelX),
    ("sobel_y", sobelY),
    ("emboss", emboss)
)

image = cv2.imread(os.path.join(PATH, DIRECTORY, IMAGE1))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

config_kernel = np.array((
    [1, -9, 4],
    [1, -1, 1],
    [2, 0, 1]), dtype="int")

# To run with config kernel
convolveOutput = convolve(gray, config_kernel)
cv2.imshow("Original", gray)
cv2.imshow("Convole", convolveOutput)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Uncomment these line to run with kernel Bank
# for (kernelName, K) in kernelBank:
#     # apply the kernel to the grayscale image using both our custom
#     # ‘convolve‘ function and OpenCV’s ‘filter2D‘ function
#     print("[INFO] applying {} kernel".format(kernelName))
#     convolveOutput = convolve(gray, K)
#     opencvOutput = cv2.filter2D(gray, -1, K)
#
#     # show the output images
#     cv2.imshow("Original", gray)
#     cv2.imshow("{} - convole".format(kernelName), convolveOutput)
#     cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
