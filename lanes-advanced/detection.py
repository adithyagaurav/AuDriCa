import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from keras import backend as K
from keras.optimizers import Adam
import numpy as np
import cv2
import time

from support.keras_ssd300 import ssd_300
from support.keras_ssd_loss import SSDLoss

# Set the image size.
img_height = 300
img_width = 300

# 1: Build the Keras model
K.clear_session() # Clear previous models from memory.
model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=20,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)

# 2: Load the trained weights into the model.
weights_path = 'VGG_VOC0712Plus_SSD_300x300_iter_240000.h5'
model.load_weights(weights_path, by_name=True)

# 3: Compile the model with Optimizer and the loss function.
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
def detect_and_depth(image, M):
	input_images = []
	image_ = cv2.resize(image, (img_height, img_width))
	input_images.append(image_)
	input_images = np.array(input_images)

	y_pred = model.predict(input_images)
	confidence_threshold = 0.5
	y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

	for box in y_pred_thresh[0]:
		# Transform the predicted bounding boxes coordinates for the 512x512 image to the original image dimensions.
		xmin = box[-4] * image.shape[1] / img_width
		ymin = box[-3] * image.shape[0] / img_height
		xmax = box[-2] * image.shape[1] / img_width
		ymax = box[-1] * image.shape[0] / img_height
		depth_point_x, depth_point_y = xmin+0.5*(xmax-xmin), ymax
		point_in_image = np.array([[depth_point_x, depth_point_y]], dtype="float64")
		point_in_depth = cv2.perspectiveTransform(np.array([point_in_image]), M)
		#Draw each bounding box on frame and write text at (left, top) coordinate
		image = cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,255,0), 2)
		text_top_coord = ymin-5 if ymin-5>5 else ymin+5 # If Not enough space left on top of bounding box write below (left, top) coordinate
		image = cv2.putText(image, '{} : {:.2f}  Depth : {:.2f}'.format(classes[int(box[0])], box[1], (image.shape[1] - point_in_depth[0][0][1])/33.06), (int(xmin), int(text_top_coord)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
	return image
