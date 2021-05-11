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

ftp = FTP(config.FTP_URL, config.FTP_USERNAME, config.FTP_PASSWORD)

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
    requests = requests.get('http://{host}:{port}/api/v1/ml-split/identity-card/import'.format(host=config.BE_HOST, port = config.BE_PORT))
    if requests.status_code != 200:
        return None
    imports= requests.json()
    for item in imports:
        if item.url == None:
            continue
        
        # create dir of date if not existed
        date_export_dir =  config.IDENTITY_CARD_EXPORT_DIR  + make_string_now_time() + '/'
        ftp.chdir(date_export_dir)
        
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
        exept Exception as e:
            print(e)
            continue
            
            # crop field nas
            ftp.upload_np_image(img_id_boxes, date_export_dir + name_doc + '_id.png', 'PNG')
            ftp.upload_np_image(img_name_boxes, date_export_dir + name_doc + '_name.png', 'PNG')
            ftp.upload_np_image(img_birth_boxes, date_export_dir + name_doc +'_birthday.png', 'PNG')
            ftp.upload_np_image(img_add_boxes, date_export_dir + name_doc + '_address.png', 'PNG')
            ftp.upload_np_image(img_home_boxes, date_export_dir + name_doc + '_home_town.png', 'PNG')
            ftp.upload_np_image(img_crop, date_export_dir + name_doc + '_crop_image.png', 'PNG')
        
schedule.every(100).seconds.do(job)
while True:
    schedule.run_pending()
    time.sleep(10)