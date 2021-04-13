"""@Author by Duy Nguyen Ngoc - email: duynguyenngoc@hotmail.com/duynn_1@digi-texx.vn"""


import cv2
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from detector import DetectorTF2
from helpers import corner_utils
from helpers import ocr_helpers
from helpers.image_utils import align_image, sort_text
from helpers import load_label_map


class CompletedModel(object):
    def __init__(self):
        self.text_detection_discharge = DetectorTF2(path_to_model='./models/discharge_record',
                                                    path_to_labels='./models/discharge_record/label_map.pbtxt',
                                                    nms_threshold=0.33, score_threshold=0.33)
    
    
    def get_corner_of_discharge_record(self, img):
        edges_image = corner_utils.edges_det(img)
        edges_image = cv2.morphologyEx(edges_image, cv2.MORPH_CLOSE, np.ones((5, 11)))
        page_contour =  corner_utils.find_page_contours(edges_image)
        page_contour =  corner_utils.four_corners_sort(page_contour)
        crop_image = corner_utils.persp_transform(img, page_contour)
        image = ocr_helpers.resize(crop_image)
        image = np.asarray(image)
        return image

    
    def detect_corner(self, image):
        detection_boxes, detection_classes, category_index = self.corner_detection_model.predict(image)

        coordinate_dict = dict()
        height, width, _ = image.shape

        for i in range(len(detection_classes)):
            label = str(category_index[detection_classes[i]]['name'])
            real_ymin = int(max(1, detection_boxes[i][0]))
            real_xmin = int(max(1, detection_boxes[i][1]))
            real_ymax = int(min(height, detection_boxes[i][2]))
            real_xmax = int(min(width, detection_boxes[i][3]))
            coordinate_dict[label] = (real_xmin, real_ymin, real_xmax, real_ymax)

        # align image
        cropped_img = align_image(image, coordinate_dict)
        return cropped_img

    
    def split_field_discharge_record(self, detections, list_class_init, num_classes, crop_image):
        im_height, im_width = crop_image.shape[:2]
        boxes = [[im_height, im_width, 0, 0] for i in range(num_classes)]
        detection_classes = detections['detection_classes']
        detection_boxes  = detections['detection_boxes']
        detection_scores  = detections['detection_scores']
        list_classes = set(detection_classes)
        list_ignore = list_class_init - list_classes
        for i in range(len(detection_classes)):
            class_id = detection_classes[i]
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
#         if boxes[6][2] > boxes[5][2]: boxes[6][0] = boxes[5][2]
#         if boxes[9][2] > boxes[8][2]: 
#             boxes[9][0] = boxes[8][2]
#             boxes[9][2] = boxes[9][0] + 50*im_height/720
#         if boxes[10][2] > boxes[9][2]: 
#             boxes[10][0] = boxes[9][2]
#             boxes[10][2] = boxes[10][0] + 50*im_height/720
        return boxes, list_ignore

    
    def detect_text_discharge_record(self, image, num_classes, list_class_init):
        detections = self.text_detection_discharge.predict(image)
        boxes, list_ignore = self.split_field_discharge_record(detections, list_class_init, num_classes, image)
        return boxes, list_ignore
        
    
    def text_recognition_giay_ra_vien(self, boxes, list_ignore, image, category_index):
        def crop_and_recog(boxes):
            end = image[boxes[0]:boxes[2], boxes[1]:boxes[3]]
            return end
        list_ans = []
        list_class = []
        for i in range(len(boxes)):
            box = boxes[i]
            class_id = i+1
            if class_id in list_ignore:
                list_ignore.remove(class_id)
                continue
            list_ans.append(crop_and_recog(box))
            list_class.append(class_id)
        return list_ans, list_class, category_index

    
    def predict_giay_ra_vien(self, image):
        path_to_labels = self.text_detection_discharge.path_to_labels
        category_index = load_label_map.create_category_index_from_labelmap(path_to_labels, use_display_name=True)
        list_class_init = set(list(category_index.keys()))
        num_classes = len(category_index)
        crop_image = self.get_corner_of_discharge_record(image)
        boxes, list_ignore = self.detect_text_discharge_record(image, num_classes, list_class_init)
        list_ans, list_class, category_index = self.text_recognition_giay_ra_vien(boxes, list_ignore, crop_image, category_index)
        return list_ans, list_class, category_index
