import numpy as np
import matplotlib.pyplot as plt
import cv2
from calibration import undistort, calibrate_camera
from binarization import convert_to_binary
from perspective import convert_to_birdeye
from projection import final_image_projections

class Line:
    def __init__(self):
        self.detected=False
        self.last_fit=None
        self.all_x=None
        self.all_y=None
        self.curvature=None
        self.last_n_fits = [[0,0,0]]
        self.average_pixel_fit = []
    
    def line_update(self, new_fit_pixel, detected):
        self.detected=detected
        #print(new_fit_pixel)
        self.last_fit = new_fit_pixel
        self.last_n_fits=np.vstack([self.last_n_fits, new_fit_pixel])
        #print(self.last_n_fits)
    
    def average_fit(self):
        self.average_pixel_fit = np.mean(self.last_n_fits[:,:], axis=0)

def compute_offset(line_left, line_right, image):
    h,w = image.shape[:2]
    line_left_bottom = line_left.all_x[line_left.all_y == max(line_left.all_y)][0]
    line_right_bottom = line_right.all_x[line_right.all_y == max(line_right.all_y)][0]
    midpoint = np.int((line_right_bottom - line_left_bottom)/2)
    centre_of_image = w/2
    deviation = centre_of_image - midpoint
    real_world_deviation = deviation * (3.7/700)
    return real_world_deviation

def blend(image, binary, warp, fit, curvature, offset):
    h,w = image.shape[:2]
    postcard_h = int(h*0.2)
    postcard_w = int(w*0.2)
    info_offsetx, info_offsety = 25,20
    mask = image.copy()
    cv2.rectangle(mask, pt1 = (0,0), pt2 = (w, postcard_h+2*info_offsety), color=(0,0,0), thickness = cv2.FILLED)
    blend_image = cv2.addWeighted(image, 1, mask,  0.2, 0.75,0)

    binary_postcard = cv2.resize(binary, dsize = (postcard_w, postcard_h))
    binary_postcard = np.dstack([binary_postcard, binary_postcard, binary_postcard]) * 255
    blend_image[info_offsety:info_offsety + postcard_h, info_offsetx:info_offsetx + postcard_w,:] = binary_postcard

    warp_postcard = cv2.resize(warp, dsize=(postcard_w, postcard_h))
    warp_postcard = np.dstack([warp_postcard, warp_postcard, warp_postcard]) * 255
    blend_image[info_offsety: info_offsety + postcard_h, 2*info_offsetx + postcard_w : 2 * (info_offsetx + postcard_w),:] = warp_postcard

    fit_postcard = cv2.resize(fit, dsize = (postcard_w, postcard_h))
    blend_image[info_offsety: info_offsety + postcard_h, 3*info_offsetx + 2*postcard_w : 3*(info_offsetx + postcard_w), :] = fit_postcard

    cv2.putText(blend_image, 'Curvature : {}'.format(curvature), (860,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(blend_image, "Offset : {}".format(offset), (860,130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255),2,cv2.LINE_AA)

    return blend_image
def from_sliding_windows(warped, line_left, line_right, processed_frames, nwindows=9):
    h, w = warped.shape[:2]
    
    histogram = np.sum(warped[h//2:, :], axis=0)
    #plt.plot(histogram)
    #plt.show()
    output = np.dstack((warped, warped, warped)) * 255
    
    midpoint = len(histogram) // 2
    base_left = np.argmax(histogram[:midpoint])
    base_right = np.argmax(histogram[midpoint:]) + midpoint
    window_height = np.int(h // nwindows)

    xleft = base_left
    xright = base_right
    
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    pix_thresh = 50

    left_inds = []
    right_inds = []
    left_positive_window_count = 0
    right_positive_window_count = 0
    for window in range(nwindows):
        win_y_low = h - (window + 1)*window_height
        win_y_high = h - window*window_height
        win_xleft_left = xleft - margin
        win_xleft_right = xleft + margin
        win_xright_left = xright - margin
        win_xright_right = xright + margin

        valid_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_left) & (nonzerox < win_xleft_right)).nonzero()[0]
        valid_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_left) & (nonzerox < win_xright_right)).nonzero()[0]
        
        left_inds.append(valid_left_inds)
        right_inds.append(valid_right_inds)
        if len(valid_left_inds) > pix_thresh:
            xleft = np.int(np.mean(nonzerox[valid_left_inds]))
            left_positive_window_count += 1
        if len(valid_right_inds) > pix_thresh:
            xright = np.int(np.mean(nonzerox[valid_right_inds]))
            right_positive_window_count +=1
        
    left_inds = np.concatenate(left_inds)
    right_inds = np.concatenate(right_inds)
    line_left.all_x = nonzerox[left_inds]
    line_left.all_y = nonzeroy[left_inds] 
    line_right.all_x = nonzerox[right_inds]
    line_right.all_y = nonzeroy[right_inds]
    left_detected=True
    right_detected=True
    if left_positive_window_count < 5:
        line_left.average_fit()
        print(line_left.average_pixel_fit)
        line_left_pixel = line_left.average_pixel_fit
        left_detected=False
    else:
        line_left_pixel = np.polyfit(line_left.all_y, line_left.all_x, 2)
    #if not list(line_left.all_x) or not list(line_left.all_y):
    #    line_left_pixel = line_left.last_fit
    #    detected=False
    #else:
    #    line_left_pixel = np.polyfit(line_left.all_y, line_left.all_x, 2)
    if right_positive_window_count < 5:
        line_right_pixel = line_right.average_fit
        right_detected=False
    else:
        line_right_pixel = np.polyfit(line_right.all_y, line_right.all_x, 2)
    #if not list(line_right.all_x) or not list(line_right.all_y):
    #    line_right_pixel = line_right.last_fit
    #    detected=False
    #else:
    #    line_right_pixel = np.polyfit(line_right.all_y, line_right.all_x, 2)
    #print(detected)
    line_left.line_update(line_left_pixel, detected=left_detected)
    
    line_right.line_update(line_right_pixel, detected=right_detected)
    

    ploty=np.linspace(0, h - 1, h)
    left_fitx = line_left_pixel[0] * ploty ** 2 + line_left_pixel[1] * ploty + line_left_pixel[2]
    right_fitx = line_right_pixel[0] * ploty ** 2 + line_right_pixel[1] * ploty + line_right_pixel[2]

    ym_per_pix = 30/720
    xm_per_pix = 3.7/700

    y_eval = np.max(ploty)
    fit_cr_left = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    curve_left = ((1+(2*line_left_pixel[0] * y_eval/2. + fit_cr_left[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr_left[0])
    fit_cr_right = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    curve_right = ((1+(2*line_right_pixel[0] * y_eval/2. + fit_cr_right[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr_right[0])
    curvature = (curve_left + curve_right)/2

    output[nonzeroy[left_inds], nonzerox[left_inds]] = [255,0,0]
    output[nonzeroy[right_inds], nonzerox[right_inds]] = [0,0,255]

    return line_left, line_right, output, curvature

def from_previous_fits(warped, line_left, line_right):
    h,w = warped.shape[:2]
    line_left_pixel = line_left.last_fit
    line_right_pixel = line_right.last_fit

    nonzero = warped.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    margin=100
    left_inds = ((nonzero_x > (line_left_pixel[0] * (nonzero_y ** 2) + line_left_pixel[1] * nonzero_y + line_left_pixel[2] - margin)) & (nonzero_x < (line_left_pixel[0] * (nonzero_y ** 2) + line_left_pixel[1] * nonzero_y + line_left_pixel[2] + margin)))
    right_inds = ((nonzero_x > (line_right_pixel[0] * (nonzero_y ** 2) + line_right_pixel[1] * nonzero_y + line_right_pixel[2] - margin)) & (nonzero_x < (line_right_pixel[0] * (nonzero_y ** 2) + line_right_pixel[1] * nonzero_y + line_right_pixel[2] + margin)))

    line_left.all_x, line_left.all_y = nonzero_x[left_inds], nonzero_y[left_inds]
    line_right.all_x, line_right.all_y = nonzero_x[right_inds], nonzero_y[right_inds]
    detected = True
    
    if not list(line_left.all_x) or not list(line_left.all_y):
        line_left_pixel = line_left.last_fit
        detected=False
    else:
        line_left_pixel = np.polyfit(line_left.all_y, line_left.all_x, 2)
    
    if not list(line_right.all_x) or not list(line_right.all_y):
        line_right_pixel = line_right.last_fit
        detected=False
    else:
        line_right_pixel = np.polyfit(line_right.all_y, line_right.all_x, 2)

    line_left.line_update(line_left_pixel, detected)
    line_right.line_update(line_right_pixel, detected)


    ploty=np.linspace(0, h - 1, h)
    left_fitx = line_left_pixel[0] * ploty ** 2 + line_left_pixel[1] * ploty + line_left_pixel[2]
    right_fitx = line_right_pixel[0] * ploty ** 2 + line_right_pixel[1] * ploty + line_right_pixel[2]
    points = np.empty([len(left_fitx),2])
    points[:,1]=ploty
    points[:,0]=left_fitx
    
    
    ym_per_pix = 30/720
    xm_per_pix = 3.7/700

    y_eval = np.max(ploty)
    fit_cr_left = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    curve_left = ((1+(2*line_left_pixel[0] * y_eval/2. + fit_cr_left[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr_left[0])
    fit_cr_right = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    curve_right = ((1+(2*line_right_pixel[0] * y_eval/2. + fit_cr_right[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr_right[0])
    curvature = (curve_left + curve_right)/2
    img_out = np.dstack((warped, warped, warped)) * 255

    img_out[nonzero_y[left_inds], nonzero_x[left_inds]] = [255,0,0]
    img_out[nonzero_y[right_inds], nonzero_x[right_inds]] = [0,0,255]

    return line_left, line_right, img_out, curvature

            
"""line_left = Line()
line_right = Line()
processed_frames = 0
ret, mtx, dist, rvecs, tvecs = calibrate_camera("calibration_images")
img = cv2.imread("road_images/test2.jpg")
img_undistorted = undistort(img,mtx,dist)
after_binary = convert_to_binary(img_undistorted)
after_warp, M, Minv = convert_to_birdeye(after_binary)
while(processed_frames<2):
    if processed_frames > 0 and line_left.detected and line_right.detected:
        line_left, line_right, img_fit, curvature = from_previous_fits(after_warp, line_left, line_right)
        processed_frames=2
    else:
        line_left, line_right, img_fit, curvature = from_sliding_windows(after_warp, line_left, line_right, nwindows=9)
        processed_frames=1
    plt.imshow(img_fit)
    plt.show()
    offset = compute_offset(line_left, line_right, img_fit)
    road_image = final_image_projections(img_undistorted, Minv, line_left, line_right)
    final_blend = blend(road_image, after_binary, after_warp, img_fit, curvature, offset)
    plt.imshow(cv2.cvtColor(final_blend, cv2.COLOR_BGR2RGB))
    plt.show()
"""
