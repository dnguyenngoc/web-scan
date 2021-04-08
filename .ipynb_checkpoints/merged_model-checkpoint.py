from detector import Detector

from detector import DetectorDischargeRecord

from recognition import TextRecognition
from helpers.image_utils import align_image, sort_text
from helpers import load_label_map
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from helpers import corner_utils
from helpers import ocr_helpers
from PIL import Image


class CompletedModel(object):
    def __init__(self):
        self.corner_detection_model = Detector(path_to_model='./models/identity_corner/model.tflite',
                                               path_to_labels='./models/identity_corner/label_map.pbtxt',
                                               nms_threshold=0.2, score_threshold=0.3)
        self.text_detection_model = Detector(path_to_model='./models/identity_card/model.tflite',
                                             path_to_labels='./models/identity_card/label_map.pbtxt',
                                             nms_threshold=0.2, score_threshold=0.2)
#         self.text_detection_discharge = Detector(path_to_model='./models/discharge_record/model.tflite',
#                                              path_to_labels='./models/discharge_record/label_map.pbtxt',
#                                              nms_threshold=0.1, score_threshold=0.1)
        self.text_detection_discharge = DetectorDischargeRecord()
        self.text_recognition_model = TextRecognition(path_to_checkpoint='./models/text_recogintion/transformerocr.pth')
    
    
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

    
    def detect_text_cmnd(self, image):
        id_boxes = None
        name_boxes = None
        birth_boxes = None
        add_boxes = None
        home_boxes = None
        
        # detect text boxes
        detection_boxes, detection_classes, category_index = self.text_detection_model.predict(image)

        # sort text boxes according to coordinate
        id_boxes, name_boxes, birth_boxes, home_boxes, add_boxes = sort_text(detection_boxes, detection_classes)
        
        return id_boxes, name_boxes, birth_boxes, home_boxes, add_boxes

    
    def text_recognition_cmnd(self, image, id_boxes, name_boxes, birth_boxes, home_boxes, add_boxes):
        field_dict = dict()

        # crop boxes according to coordinate
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

        start1 = time.time()
        result = self.text_recognition_model.predict_on_batch(np.array(list_ans))
        end1 = time.time()
        print("predicted time: ", end1 - start1)
        
        field_dict['id'] = result[0]
        field_dict['name'] = ' '.join(result[1:len(name_boxes) + 1])
        field_dict['birth'] = result[len(name_boxes) + 1]
        field_dict['home'] = ' '.join(result[len(name_boxes) + 2: -len(home_boxes)])
        field_dict['add'] = ' '.join(result[-len(home_boxes):])
        return field_dict
    
    
    def predict_cmnd(self, image):
        cropped_image =  self.detect_corner(image)
        id_boxes, name_boxes, birth_boxes, home_boxes, add_boxes = self.detect_text_cmnd(cropped_image)
        result = self.text_recognition_cmnd(cropped_image, id_boxes, name_boxes, birth_boxes, home_boxes, add_boxes)
        return result
    
    
    def split_field_discharge_record_2(self, detection_boxes, detection_classes, num_classes, crop_image):
        im_height, im_width = crop_image.shape[:2]
        boxes = [[im_height, im_width, 0, 0] for i in range(num_classes)]
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
        if boxes[6][2] > boxes[5][2]: boxes[6][0] = boxes[5][2]
        if boxes[9][2] > boxes[8][2]: 
            boxes[9][0] = boxes[8][2]
            boxes[9][2] = boxes[9][0] + 50*im_height/720
        if boxes[10][2] > boxes[9][2]: 
            boxes[10][0] = boxes[9][2]
            boxes[10][2] = boxes[10][0] + 50*im_height/720
        return boxes

    # FUCTION FOR SPLIT FIELD
    def split_field_discharge_record(self, detections, num_classes, crop_image):
        im_height, im_width = crop_image.shape[:2]
        boxes = [[im_height, im_width, 0, 0] for i in range(num_classes)]
        detection_classes = detections['detection_classes']
        detection_boxes  = detections['detection_boxes']
        detection_scores  = detections['detection_scores']
        list_classes = set(detection_classes)
        list_class_init = set([1,2,3,4,5,6,7,8,9,10,11,12,13])
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
        if boxes[6][2] > boxes[5][2]: boxes[6][0] = boxes[5][2]
        if boxes[9][2] > boxes[8][2]: 
            boxes[9][0] = boxes[8][2]
            boxes[9][2] = boxes[9][0] + 50*im_height/720
        if boxes[10][2] > boxes[9][2]: 
            boxes[10][0] = boxes[9][2]
            boxes[10][2] = boxes[10][0] + 50*im_height/720
        return boxes, list_ignore

    
    def detect_text_discharge_record(self, image):
        detections = self.text_detection_discharge.predict(image)
        boxes, list_ignore = self.split_field_discharge_record(detections, 13, image)
        return boxes, list_ignore
        
    
    
    def text_recognition_giay_ra_vien(self, boxes, list_ignore, image, category_index):
        def crop_and_recog(boxes):
            end = image[boxes[0]:boxes[2], boxes[1]:boxes[3]]
#             ocr_helpers.implt(end)
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
        result = self.text_recognition_model.predict_on_batch(np.array(list_ans))
        field_dict = dict()
        for i in range(len(result)):
            field_dict[category_index[list_class[i]]['name']] = result[i]
        return field_dict

    
    def predict_giay_ra_vien(self, image):
        path_to_lables = '/home/pot/Desktop/web-scan/models/discharge_record/ssd_mobilenet_v2_320x320_07_04_2021/label_map.pbtxt'
        crop_image = self.get_corner_of_discharge_record(image)
        boxes, list_ignore = self.detect_text_discharge_record(image)
        category_index = load_label_map.create_category_index_from_labelmap(path_to_lables, use_display_name=True)
        result = self.text_recognition_giay_ra_vien(boxes, list_ignore, crop_image, category_index)
        return result
