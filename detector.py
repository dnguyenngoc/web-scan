"""@Author by Duy Nguyen Ngoc - email: duynguyenngoc@hotmail.com/duynn_1@digi-texx.vn"""


import numpy as np
import pathlib
import cv2
import argparse
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import tensorflow.compat.v2 as tf
from helpers.image_utils import non_max_suppression_fast
from helpers import visualization_utils as viz_utils
from helpers import corner_utils, ocr_helpers
from helpers import load_label_map



class DetectorTF2(object):
    """Load model with tensorflow version from >= 2.0.0 
        
        @Param: path_to_model=path to your folder contains tensorflow models
        @Param: path_to_lables=path to file label_map.pbtxt
        @Param: score_threshold
        @Param: num_classes=number class of models
        
        @Response: {load_model} load tensorflow to backgroud system
        @Response: {predict} dict of [num_detections, detection_classes, score_detections]
        
    """ 
    
    def __init__(
        self, 
        path_to_model,
        path_to_labels, 
        nms_threshold=0.15, 
        score_threshold=0.3,
        num_classes = 13
    ):
        self.path_to_model = path_to_model
        self.path_to_labels = path_to_labels
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.num_classes = num_classes
        self.category_index = load_label_map.create_category_index_from_labelmap(
            path_to_labels, use_display_name=True)
        self.path_to_saved_model = self.path_to_model + "/saved_model"
        self.detect_fn = self.load_model()
    
    def load_model(self):
        detect_fn = tf.saved_model.load(self.path_to_saved_model)
        return detect_fn
    
    def predict(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image_rgb, axis=0)

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # input_tensor = np.expand_dims(image_np, 0)
        detections = self.detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                       for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        
        return detections

        

class Detector(object):
    """Load model with tensorflow LITE version from >= 2.0.0 
        
        @Param: path_to_model=path to your folder contains tensorflow models
        @Param: path_to_lables=path to file label_map.pbtxt
        @Param: score_threshold
        @Param: num_classes=number class of models
        
        @Response: {load_model} load tensorflow to backgroud system
        @Response: {predict} dict of [num_detections, detection_classes, score_detections]
        @Response: {draw} draw boding box of object to origin image
    """
    
    def __init__(self, path_to_model, path_to_labels, nms_threshold=0.15, score_threshold=0.3):
        self.path_to_model = path_to_model
        self.path_to_labels = path_to_labels
        self.category_index = load_label_map.create_category_index_from_labelmap(
            path_to_labels, use_display_name=True)
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold

        # load model
        self.interpreter = self.load_model()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.detection_scores = None
        self.detection_boxes = None
        self.detection_classes = None

    def load_model(self):
        # Load the TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path=self.path_to_model)
        interpreter.allocate_tensors()

        return interpreter
    
    def predict(self, img):
        original = img
        height = self.input_details[0]['shape'][1]
        width = self.input_details[0]['shape'][2]
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        img = np.expand_dims(img, axis=0)

        # Normalize input data
        input_mean = 127.5
        input_std = 127.5
        input_data = (np.float32(img) - input_mean) / input_std
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        self.interpreter.invoke()

        # Retrieve detection results
        self.detection_boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[
            0]  # Bounding box coordinates of detected objects
        self.detection_classes = self.interpreter.get_tensor(self.output_details[1]['index'])[
            0]  # Class index of detected objects
        self.detection_scores = self.interpreter.get_tensor(self.output_details[2]['index'])[
            0]  # Confidence of detected objects

        mask = np.array(self.detection_scores) > self.score_threshold
        self.detection_boxes = np.array(self.detection_boxes)[mask]
        self.detection_classes = np.array(self.detection_classes)[mask]

        self.detection_classes += 1

        # Convert coordinate to original coordinate
        h, w, _ = original.shape
        self.detection_boxes[:, 0] = self.detection_boxes[:, 0] * h
        self.detection_boxes[:, 1] = self.detection_boxes[:, 1] * w
        self.detection_boxes[:, 2] = self.detection_boxes[:, 2] * h
        self.detection_boxes[:, 3] = self.detection_boxes[:, 3] * w

        # Apply non-max suppression
        self.detection_boxes, self.detection_classes = non_max_suppression_fast(boxes=self.detection_boxes,
                                                                                labels=self.detection_classes,
                                                                                overlapThresh=self.nms_threshold)
        return self.detection_boxes, np.array(self.detection_classes).astype("int"), self.category_index

    
    def draw(self, image):
        self.detection_boxes, self.detection_classes, self.category_index = self.predict(image)
        height, width, _ = image.shape

        for i in range(len(self.detection_classes)):
            label = str(self.category_index[self.detection_classes[i]]['name'])
            real_ymin = int(max(1, self.detection_boxes[i][0]))
            real_xmin = int(max(1, self.detection_boxes[i][1]))
            real_ymax = int(min(height, self.detection_boxes[i][2]))
            real_xmax = int(min(width, self.detection_boxes[i][3]))

            cv2.rectangle(image, (real_xmin, real_ymin), (real_xmax, real_ymax), (0, 255, 0), 2)
            cv2.putText(image, label, (real_xmin, real_ymin), cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255),
                        fontScale=0.5)

        return image
