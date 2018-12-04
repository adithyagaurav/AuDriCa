import numpy as np
import cv2
import matplotlib.pyplot as plt
yellowHSV_min = np.array([0,70,70])
yellowHSV_max = np.array([50,255,255])
whiteHSV_min = np.array([0,0,0])
whiteHSV_max = np.array([0,0,255])
def HSV_thresh(image, min, max):
    """converts an image to to hsv color space and then segments the input range"""
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    above_thresh = np.all(HSV > min, axis =2)
    below_thresh = np.all(HSV < max, axis =2)
    output = np.logical_and(above_thresh, below_thresh)
    return output

def sobel_thresh(image, kernel_size=9):
    """Converts input image to edge image using sobel operator"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = kernel_size)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = kernel_size)

    smag = np.sqrt(sx ** 2 + sy ** 2)
    smag = np.uint8((smag / np.max(smag))*255)
    _, smag = cv2.threshold(smag, 50, 1, cv2.THRESH_BINARY)

    return smag.astype(bool)

def convert_to_binary(image):
    """Converts an image to binary image"""
    binary = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    HSV_mask = HSV_thresh(image, yellowHSV_min, yellowHSV_max)
    binary = np.logical_or(binary, HSV_mask)

    white_mask = HSV_thresh(image, whiteHSV_min, whiteHSV_max)
    binary = np.logical_or(binary, white_mask)

    sobel_mask = sobel_thresh(image, kernel_size = 9)
    binary = np.logical_or(binary, sobel_mask)

    kernel = np.ones((5,5), np.uint8)
    close_image = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    return close_image

"""img = cv2.imread("road_images/test6.jpg")
after_binary = convert_to_binary(img)
plt.imshow(after_binary, cmap='gray')
plt.show()
cv2.imwrite('road_images/result.jpg', after_binary)
"""
