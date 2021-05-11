"""@Author by Duy Nguyen Ngoc - email: duynguyenngoc@hotmail.com/duynn_1@digi-texx.vn"""


import cv2
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from detector import Detector
from helpers import corner_utils
from helpers import ocr_helpers
from helpers.image_utils import align_image, sort_text
from helpers import load_label_map


class CompletedModel(object):
    def __init__(self):
          self.text_detection_model = Detector(path_to_model='./models/identity_card/model.tflite',
                                               path_to_labels='./models/identity_card/label_map.pbtxt',
                                               nms_threshold=0.3, score_threshold=0.3)
    

    def get_corner_of_discharge_record(self, img):
        edges_image = corner_utils.edges_det(img)
        edges_image = cv2.morphologyEx(edges_image, cv2.MORPH_CLOSE, np.ones((5, 11)))
        page_contour =  corner_utils.find_page_contours(edges_image)
        page_contour =  corner_utils.four_corners_sort(page_contour)
        crop_image = corner_utils.persp_transform(img, page_contour)
        image = ocr_helpers.resize(crop_image)
        image = np.asarray(image)
        return image
    
    
    def detect_text_cmnd(self, image):
        id_boxes = None
        name_boxes = None
        birth_boxes = None
        home_boxes = None
        add_boxes = None
        detection_boxes, detection_classes, category_index = self.text_detection_model.predict(image)
        id_boxes, name_boxes, birth_boxes, home_boxes, add_boxes = sort_text(detection_boxes, detection_classes)
        return id_boxes, name_boxes, birth_boxes, home_boxes, add_boxes, category_index

    
    def text_recognition_cmnd(self, image, id_boxes, name_boxes, birth_boxes, home_boxes, add_boxes):
        list_class = [1,2,3,4,5,6]
        field_dict = dict()
        def crop_and_recog(boxes):
            crop = []
            if len(boxes) == 1:
                ymin, xmin, ymax, xmax = boxes[0]
                crop.append(image[ymin:ymax, xmin:xmax])
            else:
                for box in boxes:
                    ymin, xmin, ymax, xmax = box
                    crop.append(image[ymin:ymax, xmin:xmax])
            return crop
        list_ans = list(crop_and_recog(id_boxes))
        list_ans.extend(crop_and_recog(name_boxes))
        list_ans.extend(crop_and_recog(birth_boxes))
        list_ans.extend(crop_and_recog(add_boxes))
        list_ans.extend(crop_and_recog(home_boxes))
        return list_ans, list_class
        
    
    def predict_cmnd(self, image):
        cropped_image =  self.get_corner_of_discharge_record(image)
        id_boxes, name_boxes, birth_boxes, home_boxes, add_boxes, category_index = self.detect_text_cmnd(cropped_image)
        list_ans, list_class = self.text_recognition_cmnd(cropped_image, id_boxes, name_boxes, birth_boxes, home_boxes, add_boxes)
        return list_ans, list_class, category_index
    
