import os
os.chdir('./')
import time
import datetime
from ftplib import FTP
from pprint import pprint
import ftplib
from settings import config
import io
import urllib.request as urllib2
from identity_card_model import CompletedModel
import numpy as np
from helpers import corner_utils, ocr_helpers
from PIL import Image
import cv2


ftp = FTP()
ftp.connect(config.FTP_URL)
ftp.login(config.FTP_USERNAME, config.FTP_PASSWORD)


print('load model ...')
model = CompletedModel()
print('model is loaded.')


def list(path):
    return ftp.nlst(path)


def upload_np_image(np_image, path, file_type):
    image = Image.fromarray(np_image.astype('uint8'))
    temp = io.BytesIO()
    image.save(temp, format=file_type)
    temp.seek(0)
    ftp.storbinary('STOR ./{path}'.format(path=path), temp)


def read(path):
    fh = urllib2.urlopen('ftp://{user}:{pw}@{host}/{path}'.format(user = config.FTP_USERNAME, pw=config.FTP_PASSWORD, host=config.FTP_URL, path = path))
    return fh.read()


def move(source, destination):
    ftp.rename(source, destination)


def crop_image(img):
    img = np.asarray(img)
    edges_image = corner_utils.edges_det(img)
    edges_image = cv2.morphologyEx(edges_image, cv2.MORPH_CLOSE, np.ones((5, 11)))
    page_contour =  corner_utils.find_page_contours(edges_image)
    page_contour =  corner_utils.four_corners_sort(page_contour)
    crop_image = corner_utils.persp_transform(img, page_contour)
    image = ocr_helpers.resize(crop_image)
    image_end = Image.fromarray(np.uint8(image)).convert('RGB')
    image_with_detections = image.copy()
    return image


def crop_and_recog(boxes, image):
    crop = []
    if len(boxes) == 1:
        ymin, xmin, ymax, xmax = boxes[0]
        crop.append(image[ymin:ymax, xmin:xmax])
    else:
        for box in boxes:
            ymin, xmin, ymax, xmax = box
            crop.append(image[ymin:ymax, xmin:xmax])
    return crop


def handle_detection(name_boxes, img_crop):
    y_min, x_min, y_max, x_max = (name_boxes[0][0], name_boxes[0][1], name_boxes[0][2], name_boxes[0][3])
    for item in name_boxes:
        ymin = item[0]
        xmin = item[1]
        ymax = item[2]
        xmax = item[3]
        if ymin < y_min: y_min = ymin
        if xmin < x_min: x_min = xmin
        if ymax > y_max: y_max = ymax
        if xmax > x_max: x_max = xmax
    return img_crop[y_min:y_max, x_min-10:x_max+10]


identity_card_import_dir = 'identity_card/import/'
identity_card_export_dir = 'identity_card/export/'

while True:
    print('start session ...')
    imports = list(identity_card_import_dir)
    for item in imports:
        image_path = identity_card_import_dir + item
        image = read(image_path)
        image = np.array(Image.open(io.BytesIO(image))) 
        img_crop = crop_image(image)
        im_height,im_width, _ = img_crop.shape
        id_boxes, name_boxes, birth_boxes, home_boxes, add_boxes, category_index = model.detect_text_cmnd(img_crop)
        img_id_boxes = handle_detection(id_boxes, img_crop)
        upload_np_image(img_id_boxes, identity_card_export_dir +'id_'+ item, 'JPEG')
        img_name_boxes = handle_detection(name_boxes, img_crop)
        upload_np_image(img_name_boxes, identity_card_export_dir +'name_'+ item, 'JPEG')
        img_birth_boxes = handle_detection(birth_boxes, img_crop)
        upload_np_image(img_birth_boxes, identity_card_export_dir +'birth_'+ item, 'JPEG')
        img_add_boxes = handle_detection(add_boxes, img_crop)
        upload_np_image(img_add_boxes, identity_card_export_dir +'add_'+ item, 'JPEG')
        img_home_boxes = handle_detection(home_boxes, img_crop)
        upload_np_image(img_home_boxes, identity_card_export_dir +'home_'+ item, 'JPEG')
        move(image_path, identity_card_export_dir + item)
    print('sleep 100 second ...') 
    time.sleep(100)
