from detector import Detector
from recognition import TextRecognition
from utils.image_utils import align_image, sort_text
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


class CompletedModel(object):
    def __init__(self):
        self.corner_detection_model = Detector(path_to_model='./models/corner/model.tflite',
                                               path_to_labels='./models/corner/label_map.pbtxt',
                                               nms_threshold=0.2, score_threshold=0.3)
        self.text_detection_model = Detector(path_to_model='./models/identity_card/model.tflite',
                                             path_to_labels='./models/identity_card/label_map.pbtxt',
                                             nms_threshold=0.2, score_threshold=0.2)
        self.text_detection_discharge = Detector(path_to_model='./models/discharge_record/model.tflite',
                                             path_to_labels='./config_text_detection_giay_ra_vien/label_map.pbtxt',
                                             nms_threshold=0.2, score_threshold=0.2)
        
        self.text_recognition_model = TextRecognition(path_to_checkpoint='./config_text_recognition/transformerocr.pth')

    
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
        cropped_image = image #self.detect_corner(image)
        id_boxes, name_boxes, birth_boxes, home_boxes, add_boxes = self.detect_text_cmnd(cropped_image)
        result = self.text_recognition_cmnd(cropped_image, id_boxes, name_boxes, birth_boxes, home_boxes, add_boxes)
        return result
    
    
    def detect_text_giay_ra_vien(self, image):
        name_boxes, birth_boxes, sex_boxes, nation_boxes, country_boxes, id_boxes, add_boxes, come_time_boxes, \
        out_time_boxes, diagnostic_boxes, solution_boxes, note_boxes = \
        (None, None, None, None, None, None, None, None, None, None, None, None)
        
        h,w = image.shape[0:2] 
        name_boxes = [[165, 0, 190, 500]] # temp
        birth_boxes = [[165, 600, 205, 1200]] # temp
        sex_boxes = [[165, 600, 205, 1200]] # temp
        nation_boxes = [[210, 0, 250, 300]] # temp
        country_boxes = [[165, 273, 189,500]] # temp
        id_boxes = [[165, 273, 190, 500]] # temp
        add_boxes = [[165, 273, 190, 500]] # temp
        come_time_boxes = [[165, 273, 190, 500]] # temp
        out_time_boxes = [[165, 273, 190, 500]] # temp
        diagnostic_boxes = [[165, 273, 190, 500]] # temp
        solution_boxes = [[165, 273, 190, 500]] # temp
        note_boxes = [[165, 273, 190, 500]] # temp
        return name_boxes, birth_boxes, sex_boxes, nation_boxes, country_boxes, id_boxes, add_boxes, come_time_boxes, \
        out_time_boxes, diagnostic_boxes, solution_boxes, note_boxes
    
    
    def text_recognition_giay_ra_vien(self, image, name_boxes, birth_boxes, sex_boxes, nation_boxes, country_boxes, \
        id_boxes, add_boxes, come_time_boxes, out_time_boxes, diagnostic_boxes, solution_boxes, note_boxes):
        
        def crop_and_recog(boxes):
            crop = []
            if len(boxes) == 1:
                ymin, xmin, ymax, xmax = boxes[0]
                crop.append(image[ymin:ymax, xmin:xmax])
            else:
                for box in boxes:
                    ymin, xmin, ymax, xmax = box
                    crop.append(image[ymin:ymax, xmin:xmax])
            plt.imshow(image[ymin:ymax, xmin:xmax])
            plt.show()
            return crop
        
        list_ans = list(crop_and_recog(name_boxes))
        list_ans.extend(crop_and_recog(birth_boxes))
#         list_ans.extend(crop_and_recog(sex_boxes))
        list_ans.extend(crop_and_recog(nation_boxes))
        list_ans.extend(crop_and_recog(country_boxes))
        list_ans.extend(crop_and_recog(id_boxes))
        list_ans.extend(crop_and_recog(add_boxes))
        list_ans.extend(crop_and_recog(come_time_boxes))
        list_ans.extend(crop_and_recog(out_time_boxes))
        list_ans.extend(crop_and_recog(diagnostic_boxes))
        list_ans.extend(crop_and_recog(solution_boxes))
        list_ans.extend(crop_and_recog(note_boxes))
        
        result = self.text_recognition_model.predict_on_batch(np.array(list_ans))

        field_dict = dict()
        field_dict['name'] = result[0].split(': ')[1]
        b_and_s = result[1].split('-')
        field_dict['birth'] = b_and_s[0].split(' ')[1]
        field_dict['sex'] = b_and_s[1].split(': ')[1]
#         field_dict['nation'] = result[3]
#         field_dict['country_boxes'] = result[4]
#         field_dict['id'] = result[5]
#         field_dict['add'] = result[6]
#         field_dict['come_time'] = result[7]
#         field_dict['out_time'] = result[8]
#         field_dict['diagnostic'] = result[9]
#         field_dict['solution'] = result[10]
#         field_dict['note'] = result[11]
        return field_dict

    
    def predict_giay_ra_vien(self, image):
        name_boxes, birth_boxes, sex_boxes, nation_boxes, country_boxes, id_boxes, add_boxes, come_time_boxes, \
        out_time_boxes, diagnostic_boxes, solution_boxes, note_boxes = self.detect_text_giay_ra_vien(image)
        result = self.text_recognition_giay_ra_vien(image, name_boxes, birth_boxes, sex_boxes, nation_boxes, \
                                           country_boxes, id_boxes, add_boxes, come_time_boxes, out_time_boxes, \
                                           diagnostic_boxes, solution_boxes, note_boxes)
        return result
