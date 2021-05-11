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



def upload_normal(name, document_id,field_name, image):
    fields = {'name': name,'document_id': document_id,}
    files =  {'image': (name, bytes(image), 'image/png')}
    r = requests.post(
        'http://{host}:{port}/api/v1/ftp/image/document-crop'.format(host=config.BE_HOST, port = config.BE_PORT),
        files=files,
        data=fields,
        verify=False
    )
    return r


def upload_crop(name, document_id, field_name, image):
    fields = {'name': name,'document_id': document_id,}
    files =  {'image': (name, bytes(image), 'image/png')}
    r = requests.post(
        'http://{host}:{port}/api/v1/ftp/image/document-crop'.format(host=config.BE_HOST, port = config.BE_PORT),
        files=files,
        data=fields,
        verify=False
    )
    return r
    
    
def job():
    print('start')
    res = requests.get('http://{host}:{port}/api/v1/ml-split/identity-card/100'.format(host=config.BE_HOST, port = config.BE_PORT))
    print('load all identity card import status 100', res.status_code)
    if res.status_code != 200:
        print(res.json())
        return None
    imports= res.json()
    for item in imports:
        if item['url'] == None:
            continue
        try:
            image = image_utils.read_bytes_image_from_url(item['url'])
            image = np.array(Image.open(io.BytesIO(image)))
            img_crop = image_utils.crop_image(image)
            im_height, im_width, _ = img_crop.shape
            name_doc = item['name'].split('.')[0]
            id_boxes, name_boxes, birth_boxes, home_boxes, add_boxes, category_index = model.detect_text_cmnd(img_crop)
            img_id_boxes = image_utils.handle_detection(id_boxes, img_crop)
            img_name_boxes = image_utils.handle_detection(name_boxes, img_crop)
            img_birth_boxes = image_utils.handle_detection(birth_boxes, img_crop)
            img_add_boxes = image_utils.handle_detection(add_boxes, img_crop)
            img_home_boxes = image_utils.handle_detection(home_boxes, img_crop)
        except Exception as e:
            print('[error] when extract field with: ',e)
            continue

        list_fields = ['id',  'name', 'birthday', 'home_town', 'address']
        list_image_fields = [id_boxes, name_boxes, birth_boxes, home_boxes, add_boxes]
        for i in range(len(list_fields)):
            field_name = list_fields[i]
            image_now = list_image_fields[i]
            name = item['name'].split('.')[0] + '_' + field_name + '.png'
            document_id = item['id']
            print('[run] upload with field: ', field_name)
            r = upload_normal(name, document_id, field_name, image_now)
            print(r.status_code)
        pritn('[run] upload crop_image > done')
        r = upload_crop(name, document_id, field_name, img_crop)
        print(r.status_code)
        
        
schedule.every(1).seconds.do(job)
while True:
    schedule.run_pending()
    time.sleep(10)