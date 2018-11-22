import numpy as np
import cv2
def final_image_projections(image, Minv, line_left, line_right):
    binary = np.zeros_like(image, dtype=np.uint8)
    print(binary.shape)
    h,w = image.shape[:2]
    line_left_pixel = line_left.last_fit
    line_right_pixel = line_right.last_fit
    
    ploty = np.linspace(0, h -1, h)
    left_fitx = line_left_pixel[0] * ploty ** 2 + line_left_pixel[1] * ploty + line_left_pixel[2]
    right_fitx = line_right_pixel[0] * ploty ** 2 + line_right_pixel[1] * ploty + line_right_pixel[2]

    
    pts_left = np.empty([len(left_fitx),2])
    pts_left[:,0] = left_fitx
    pts_left[:,1] = ploty
    pts_left_copy = pts_left 
    pts_left = pts_left.reshape(1,len(left_fitx),2)

    pts_right = np.empty([len(right_fitx),2])
    pts_right[:,0] = right_fitx
    pts_right[:,1] = ploty
    pts_right = np.flipud(pts_right)
    pts_right_copy = pts_right
    pts_right = pts_right.reshape(1,len(right_fitx),2)

    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(binary, np.int_([pts]), (0, 255, 0))
    dewarp = cv2.warpPerspective(binary, Minv, (w,h))
    projected_image = cv2.addWeighted(image, 1, dewarp, 0.2,gamma=0)

    line_projection = np.zeros_like(image, dtype=np.uint8)
    left_line_poly_coords = np.array([find_line_poly(pts_left_copy)], dtype=np.int32)
    right_line_poly_coords = np.array([find_line_poly(pts_right_copy)], dtype=np.int32)
    cv2.fillPoly(line_projection, left_line_poly_coords,(0,0,255))
    cv2.fillPoly(line_projection, right_line_poly_coords,(255,0,0))
    dewarp_line = cv2.warpPerspective(line_projection, Minv,(w,h))

    on_road_image = cv2.addWeighted(projected_image, 1, dewarp_line,1,gamma=0)

    return on_road_image

def find_line_poly(pts):
    line_width=50
    left_edge = np.empty([len(pts),2])
    right_edge = np.empty([len(pts),2])
    left_edge[:,0] = pts[:,0] - 25
    left_edge[:,1] = pts[:,1]

    right_edge[:,0] = np.flipud(pts[:,0]) + 25
    right_edge[:,1] = np.flipud(pts[:,1])
    poly_coords = np.vstack([left_edge, right_edge])
    return poly_coords