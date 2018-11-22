import numpy as np
import matplotlib.pyplot as plt
import cv2
from calibration import undistort, calibrate_camera
from binarization import convert_to_binary
from perspective import convert_to_birdeye
from projection import final_image_projections
from finding_lines import Line, from_previous_fits, from_sliding_windows, compute_offset, blend
from moviepy.editor import VideoFileClip
line_left = Line()
line_right = Line()
processed_frames = 0
def pipeline(frame):
    global line_left, line_right, processed_frames
    img_undistorted = undistort(frame,mtx,dist)
    after_binary = convert_to_binary(img_undistorted)
    after_warp, M, Minv = convert_to_birdeye(after_binary)
    #if processed_frames > 0 and line_left.detected and line_right.detected:
    #    line_left, line_right, img_fit, curvature = from_previous_fits(after_warp, line_left, line_right)
    #else:
    line_left, line_right, img_fit, curvature = from_sliding_windows(after_warp, line_left, line_right, processed_frames, nwindows=9)
    offset = compute_offset(line_left, line_right, img_fit)
    road_image = final_image_projections(img_undistorted, Minv, line_left, line_right)
    final_blend = blend(road_image, after_binary, after_warp, img_fit, curvature, offset)
    processed_frames += 1

    return final_blend

if __name__ == '__main__':
    ret, mtx, dist, rvecs, tvecs = calibrate_camera("calibration_images")
    clip = VideoFileClip('project_video.mp4').fl_image(pipeline)
    clip.write_videofile('out_project_video.mp4', audio=False)
    """cap = cv2.VideoCapture('project_video_.mp4')
    while(cap.isOpened()):
        ret, frame = cap.read()
        out_frame = pipeline(frame)
        out_frame = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
        plt.imshow(out_frame)
        plt.show()
    #img = cv2.imread("road_images/test9.jpg")
    #blend = cv2.cvtColor(pipeline(img), cv2.COLOR_BGR2RGB)
    #plt.imshow(blend)
    #plt.show()"""


