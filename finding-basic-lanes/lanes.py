import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio
def make_points(image, line):
    #print(line[0], line[1])
    slope, intercept = line
    y1 = int(image.shape[0])
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return ([x1, y1, x2, y2])

def line_fits(image, lines):
    left_fit = []
    right_fit = []
    
    for line in lines:
        x1, y1, x2, y2 = np.reshape(line, 4)
        fit = np.polyfit((x1, x2), (y1, y2), 1)
        slope = fit[0]
        intercept = fit[1]
        if slope < 0:
            left_fit.append([slope, intercept])
        else:
            right_fit.append([slope, intercept])
    return left_fit, right_fit
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(blur, 50, 150)
    return edge

def display_lines(image, lines):
    #left_line, right_line = np.reshape(lines,2)
    line_image = np.zeros_like(image)
    poly_points = []
    for line in lines:
        x1,y1,x2,y2 = np.reshape(line, 4)
        points = np.array([(x1,y1),(x2,y2)]).reshape(1,2,2)
        cv2.line(line_image, (x1,y1), (x2,y2), color = (255,0,0), thickness = 10)
        poly_points.append(points.reshape(1,2,2))
    
    poly_points=np.array(poly_points).reshape(-1,4,2)
    #print(poly_points)
    buff = np.copy(poly_points[0,2])
    poly_points[0,2]=poly_points[0,3]
    poly_points[0,3]=buff
    #print(poly_points)
    cv2.fillPoly(line_image, poly_points, color=(0,255,0))
    return line_image

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    polygon = np.array([[(0, height), (width, height), (int(0.55*width), int(0.6*height)), (int(0.45*width), int(0.6*height))]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image
input_video = 'test2.mp4'
output_video = 'output2.mp4'
video_reader = imageio.get_reader(input_video)
fps = video_reader.get_meta_data()['fps']
video_writer = imageio.get_writer(output_video, fps=fps, codec='mpeg4')
#image = cv2.imread('C:\\Users\Aadi\\finding-lanes\\test_image.jpg')
frame_count=0
old_left_line_average = [0,0]
old_right_line_average = [0,0]
for frame in video_reader:
    frame_count += 1
    print(frame_count)
    image = frame
    edge = canny(image)
    
    roi = region_of_interest(edge)
    lines = cv2.HoughLinesP(roi, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
    left_fit, right_fit = line_fits(image, lines)
    if not len(left_fit) == 0:
        left_line_average = np.average(left_fit, axis = 0)
    else:
        left_line_average = old_left_line_average
    if not len(right_fit) == 0:
        right_line_average = np.average(right_fit, axis = 0)
    else:
        right_line_average = old_right_line_average
    left_line = make_points(image, left_line_average)
    right_line = make_points(image, right_line_average)
    old_left_line_average = left_line_average
    old_right_line_average = right_line_average
    averaged_lines = ([left_line, right_line])
    line_image = display_lines(image, averaged_lines)
    output_image = cv2.addWeighted(image, 1, line_image, 0.4,1,0)
    video_writer.append_data(output_image)