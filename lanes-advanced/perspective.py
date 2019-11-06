import numpy as np
import cv2
from binarization import convert_to_binary
import matplotlib.pyplot as plt
def convert_to_birdeye(image):
    """Converts the image view to bird's eye view using perspective transformation"""
    h, w = image.shape[:2]
    points1 = np.float32([[w,h-10],[0,h-10],[487,459],[780,459]])
    points2 = np.float32([[w,h],[0,h],[0,0],[w,0]])
    M = cv2.getPerspectiveTransform(points1, points2)
    Minv = cv2.getPerspectiveTransform(points2, points1)
    warped =  cv2.warpPerspective(image, M, (w,h),flags=cv2.INTER_LINEAR)
    #print(warped.shape)
    return warped, M, Minv

"""img = cv2.imread("road_images/test6.jpg")
after_binary = convert_to_binary(img)
after_warp, M, Minv = convert_to_birdeye(after_binary)
print(after_warp.shape)
plt.subplot(1,2,1)
plt.imshow(after_binary, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(after_warp, cmap = 'gray')
plt.show()
"""
