import numpy as np
import cv2
from finding_lines import Line
def compute_offset(line_left, line_right, image):
    h,w = image.shape[:2]
    line_left_bottom = line_left.all_x[line_left.all_y == max(line_left.all_y)]
    line_right_bottom = line_right.all_x[line_right.all_y == max(line_right.all_y)]
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
    mask = np.zeros_like(image, dtype=np.uint8)
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

