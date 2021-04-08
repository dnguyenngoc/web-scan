import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    
import pathlib
import tensorflow.compat.v2 as tf
import cv2
import argparse
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from helpers import corner_utils, ocr_helpers


tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')


# CONFIG_GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

    
# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = '/home/pot/Desktop/web-scan/models/discharge_record'
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = '/home/pot/Desktop/web-scan/models/discharge_record/label_map.pbtxt'

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = 0.34

# NUM_CLASSES
NUM_CLASSES = 13

# LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

# LOAD CATEGORY INDEX
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


# LOAD IMAGE TEST
img = ocr_helpers.read_image_from_dir(r'/home/pot/Desktop/web-scan/test_data/discharge_record/1.jpg')
img = np.asarray(img)

# CROP CORNER
edges_image = corner_utils.edges_det(img)
edges_image = cv2.morphologyEx(edges_image, cv2.MORPH_CLOSE, np.ones((5, 11)))
page_contour =  corner_utils.find_page_contours(edges_image)
page_contour =  corner_utils.four_corners_sort(page_contour)
crop_image = corner_utils.persp_transform(img, page_contour)
image = ocr_helpers.resize(crop_image)
image_end = Image.fromarray(np.uint8(image)).convert('RGB')



# DETECTIONS
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_expanded = np.expand_dims(image_rgb, axis=0)

# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
input_tensor = tf.convert_to_tensor(image)
# The model expects a batch of images, so add an axis with `tf.newaxis`.
input_tensor = input_tensor[tf.newaxis, ...]

# input_tensor = np.expand_dims(image_np, 0)
detections = detect_fn(input_tensor)

# All outputs are batches tensors.
# Convert to numpy arrays, and take index [0] to remove the batch dimension.
# We're only interested in the first num_detections.
num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
               for key, value in detections.items()}
detections['num_detections'] = num_detections
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)


# FUCTION FOR SPLIT FIELD
def split_field_discharge_record(detections, num_classes, crop_image):
    im_height, im_width = crop_image.shape[:2]
    boxes = [[im_height, im_width, 0, 0] for i in range(num_classes)]
    detection_classes = detections['detection_classes']
    detection_boxes  = detections['detection_boxes']
    detection_scores  = detections['detection_scores']
#     verify_missing_classes = set(detection_classes)
#     print(verify_missing_classes)
    for i in range(len(detection_classes)):
        class_id = detection_classes[i]
        if detection_scores[i] <= 0.3:
            continue
        (ymin, xmin, ymax, xmax) = (
            detection_boxes[i][0] * im_height, 
            detection_boxes[i][1] * im_width, 
            detection_boxes[i][2] * im_height, 
            detection_boxes[i][3] * im_width
        )
        if ymin < boxes[class_id -1][0]: boxes[class_id -1][0] = ymin
        if xmin < boxes[class_id -1][1]: boxes[class_id -1][1] = xmin
        if ymax > boxes[class_id -1][2]: boxes[class_id -1][2] = ymax
        if xmax > boxes[class_id -1][3]: boxes[class_id -1][3] = xmax
    boxes = np.array(boxes).astype(int)
    return boxes

boxes = split_field_discharge_record(detections, NUM_CLASSES, crop_image)

# (name_boxes, age_boxes, gender_boxes, nation_boxs, country_box, id_boxes, add_boxes, \
#      come_time_boxes, out_time_boxes, diagnostic_boxes, solution_boxes, note_boxes, job_boxes) = \
#         split_field_discharge_record(detections, 13, crop_image)
i = 1
for box in boxes: 
    cv2.imwrite('/home/pot/Desktop/web-scan/notebook/field_{}.jpg'.format(str(i)), np.asarray(image_end.crop((box[1], box[0], box[3], box[2]))))
    i+=1

    
# image_with_detections = image.copy()

# cv2.imwrite('/home/pot/Desktop/web_service/notebook/crop_image.jpg', image_with_detections)

# # SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
# viz_utils.visualize_boxes_and_labels_on_image_array(
#       image_with_detections,
#       detections['detection_boxes'],
#       detections['detection_classes'],
#       detections['detection_scores'],
#       category_index,
#       use_normalized_coordinates=True,
#       max_boxes_to_draw=200,
#       min_score_thresh=MIN_CONF_THRESH,
#       agnostic_mode=False)

# cv2.imwrite('/home/pot/Desktop/web_service/notebook/end.jpg', image_with_detections)

