import os
os.chdir('./')
import time
import datetime
from helpers.ftp_utils import FTP
from pprint import pprint
from settings import config
import io
from identity_card_model import CompletedModel
import numpy as np
from helpers import image_utils
from PIL import Image
import cv2
import datetime
import requests
import schedule


print('load model ...')
model = CompletedModel()
print('model is loaded.')


def make_string_now_time():
        now = datetime.datetime.utcnow()
        if now.month <= 10:
            month = '0' + str(now.month)
        else:
            month = str(now.month)
        if now.day <= 10:
            day = '0' + str(now.day)
        else:
            day = str(now.day)
        return str(now.year) + '-' + month + '-' + day
    
def job():
    print('start')
    res = requests.get('http://{host}:{port}/api/v1/ml-split/identity-card/import'.format(host=config.BE_HOST, port = config.BE_PORT))
    if res.status_code != 200:
        return None
    imports= res.json()
    for item in imports:
        if item.url == None:
            continue
        try:
            image = image_utils.read_bytes_image_from_url(item.url)
            image = np.array(Image.open(io.BytesIO(image)))
            img_crop = image_utils.crop_image(image)
            im_height, im_width, _ = img_crop.shape
            name_doc = item.name.split('.')[0]
            id_boxes, name_boxes, birth_boxes, home_boxes, add_boxes, category_index = model.detect_text_cmnd(img_crop)
            img_id_boxes = image_utils.handle_detection(id_boxes, img_crop)
            img_name_boxes = image_utils.handle_detection(name_boxes, img_crop)
            img_birth_boxes = image_utils.handle_detection(birth_boxes, img_crop)
            img_add_boxes = image_utils.handle_detection(add_boxes, img_crop)
            img_home_boxes = image_utils.handle_detection(home_boxes, img_crop)
        except Exception as e:
            print(e)
            continue

        list_fields = ['address', 'id', 'home_town', 'name', 'birthday', 'crop_image']

        for k in len_list_fields:
            field_name = list_fields[k]
            
            if k == list_fields:
                fields = {
                    'name': item.name.split('.')[0] + '_' + field_name + '.png',
                    'document_id': item.id,
                }
                files =  {'image': (item.name.split('.')[0] + '_' + field_name + '.png', image, 'image/png')}
                r = requests.post(
                    'http://{host}:{port}/api/v1/ftp/image/document-crop'.format(host=config.BE_HOST, port = config.BE_PORT),
                    files=files, 
                    data=fields, 
                    verify=False
                )
                break
                
            
            fields = {
                'name': item.name.split('.')[0] + '_' + field_name + '.png',
                'document_id': item.id,
                'field_name': field_name
            }
            files =  {'image': (item.name.split('.')[0] + '_' + field_name + '.png', image, 'image/png')}
            r = requests.post(
                'http://{host}:{port}/api/v1/ftp/image/split'.format(host=config.BE_HOST, port = config.BE_PORT),
                files=files, 
                data=fields, 
                verify=False
            )
        
schedule.every(20).seconds.do(job)
while True:
    schedule.run_pending()
    time.sleep(10)