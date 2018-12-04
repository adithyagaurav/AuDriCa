import numpy as np
import cv2
import os
from glob import glob
import matplotlib.pyplot as plt

def calibrate_camera(calib_dir):
    """Generates the calibration matrix to eliminate camera distortions"""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    images = glob(os.path.join(calib_dir, '*.jpg'))
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret_, corners = cv2.findChessboardCorners(gray, (9,6), None)
        if ret_ == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners,(11,11),(-1,-1), criteria)
            imgpoints.append(corners2)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    return ret, mtx, dist, rvecs, tvecs

def undistort(frame, mtx, dist):
    """Undistorts the input image using a calibration matrix"""
    undistorted_frame = cv2.undistort(frame, mtx, dist)
    return undistorted_frame


"""ret, mtx, dist, rvecs, tvecs = calibrate_camera("calibration_images")
img = cv2.imread("calibration_images/calibration2.jpg")
img_undistorted = undistort(img, mtx, dist)
show_img = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2RGB)
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(show_img)
plt.show()

"""