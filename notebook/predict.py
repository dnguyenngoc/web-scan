import os
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
import imutils
from helpers import corner_utils as utlis
from helpers import ocr_helpers as ocr_utils


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    
import pathlib
import tensorflow as tf
from PIL import Image
import warnings

img = ocr_utils.read_image_from_dir(r'/home/pot/Desktop/web_service/test_image/1.jpg')

img = np.asarray(img)
edges_image = utlis.edges_det(img)
edges_image = cv2.morphologyEx(edges_image, cv2.MORPH_CLOSE, np.ones((5, 11)))
page_contour =  utlis.find_page_contours(edges_image)
page_contour =  utlis.four_corners_sort(page_contour)
crop_image = utlis.persp_transform(img, page_contour)
crop_image = ocr_utils.resize(crop_image)

tf.get_logger().setLevel('ERROR')  

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = '/home/pot/Desktop/artificial_intelligence/workspace/discharge_record/exported_models/ssd_mobilenet_v2_320x320'

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = '/home/pot/Desktop/artificial_intelligence/workspace/discharge_record/models/label_map.pbtxt'

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = 0.34

# LOAD THE MODEL
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# LOAD LABEL MAP DATA FOR PLOTTING

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
    Args:
      path: the file path to the image
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

image = crop_image

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


import pickle


with open('/home/pot/Desktop/web_service/notebook/data_detect.pickle', 'wb') as handle:
    pickle.dump(detections, handle, protocol=4)



# print(df.columns)
# df.to_pickle('/home/pot/Desktop/web_service/notebook/data_detect.pickle', compression='infer', protocol=5, storage_options=None)



image_with_detections = image.copy()
cv2.imwrite('/home/pot/Desktop/web_service/notebook/crop_image.jpg', image_with_detections)

# SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
viz_utils.visualize_boxes_and_labels_on_image_array(
      image_with_detections,
      detections['detection_boxes'],
      detections['detection_classes'],
      detections['detection_scores'],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=MIN_CONF_THRESH,
      agnostic_mode=False)

cv2.imwrite('/home/pot/Desktop/web_service/notebook/end.jpg', image_with_detections)

