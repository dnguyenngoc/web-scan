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
from database.db import db_session
from database.repository_crud import document_crud, document_split_crud
from database.entities import document_entity, document_split_entity
import datetime


ftp = FTP(config.FTP_URL, config.FTP_USERNAME, config.FTP_PASSWORD)


print('load model ...')
model = CompletedModel()
print('model is loaded.')


while True:
    print('start session ...')
    
    # Load list_file
    imports = ftp.list(config.IDENTITY_CARD_IMPORT_DIR)

    for item in imports:
        image_path = config.IDENTITY_CARD_IMPORT_DIR + item
        
        ############################################################
        # DB
        ############################################################
        document = document_crud.create(
            db_session, 
            document_entity.DocumentCreate(
                name = item, 
                type_id = 1, 
                url = 'http://' + config.FTP_URL + '/' + image_path,
                status_id = 1,
                create_date = datetime.datetime.utcnow()
            )
        )
        ############################################################
        # DB
        ############################################################
        
        image = ftp.read(image_path)
        image = np.array(Image.open(io.BytesIO(image))) 
        img_crop = image_utils.crop_image(image)
        im_height,im_width, _ = img_crop.shape
        id_boxes, name_boxes, birth_boxes, home_boxes, add_boxes, category_index \
                                                                  = model.detect_text_cmnd(img_crop)
        
        now = datetime.datetime.utcnow()
        if now.month <= 10:
            month = '0' + str(now.month)
        else:
            month = str(now.month)
            
        if now.day <= 10:
            day = '0' + str(now.day)
        else:
            day = str(now.day)
 
        date_export_dir =  config.IDENTITY_CARD_EXPORT_DIR  + str(now.year) + '-' + month + '-' + day + '/'
        image_export_dir = date_export_dir + str(document.id) + '/'
        ftp.chdir(date_export_dir)
        ftp.chdir(image_export_dir)
        
        # crop field
        img_id_boxes = image_utils.handle_detection(id_boxes, img_crop)
        ftp.upload_np_image(img_id_boxes, image_export_dir + 'id.jpg', 'JPEG')
        
        img_name_boxes = image_utils.handle_detection(name_boxes, img_crop)
        ftp.upload_np_image(img_name_boxes, image_export_dir + 'name.jpg', 'JPEG')
        
        img_birth_boxes = image_utils.handle_detection(birth_boxes, img_crop)
        ftp.upload_np_image(img_birth_boxes, image_export_dir + 'birth.jpg', 'JPEG')
        
        img_add_boxes = image_utils.handle_detection(add_boxes, img_crop)
        ftp.upload_np_image(img_add_boxes, image_export_dir + 'add.jpg', 'JPEG')
        
        img_home_boxes = image_utils.handle_detection(home_boxes, img_crop)
        ftp.upload_np_image(img_home_boxes, image_export_dir + 'home.jpg', 'JPEG')
        
        # Move image import to export
        ftp.upload_np_image(img_crop, image_export_dir + 'crop.jpg', 'JPEG')
        
        ############################################################
        # DB
        ############################################################
        document_split_crud.create(
            db_session, 
            document_split_entity.DocumentSplitCreate(
                name = 'identity',
                type_id = 1,
                url = 'http://' + config.FTP_URL + '/' + image_export_dir + 'id.jpg',
                document_id = document.id, 
                create_date = datetime.datetime.utcnow()
             )
         )
        document_split_crud.create(
            db_session, 
            document_split_entity.DocumentSplitCreate(
                name = 'address',
                type_id = 1,
                url = 'http://' + config.FTP_URL + '/' + image_export_dir + 'add.jpg',
                document_id = document.id, 
                create_date = datetime.datetime.utcnow()
             )
         )
        document_split_crud.create(
            db_session, 
            document_split_entity.DocumentSplitCreate(
                name = 'full_name',
                type_id = 1,
                url = 'http://' + config.FTP_URL + '/' + image_export_dir + 'name.jpg',
                document_id = document.id, 
                create_date = datetime.datetime.utcnow()
             )
         )
        document_split_crud.create(
            db_session, 
            document_split_entity.DocumentSplitCreate(
                name = 'birth_day',
                type_id = 1,
                url = 'http://' + config.FTP_URL + '/' + image_export_dir + 'birth.jpg',
                document_id = document.id, 
                create_date = datetime.datetime.utcnow()
             )
         )
        document_split_crud.create(
            db_session, 
            document_split_entity.DocumentSplitCreate(
                name = 'home_town',
                type_id = 1,
                url = 'http://' + config.FTP_URL + '/' + image_export_dir + 'home.jpg',
                document_id = document.id, 
                create_date = datetime.datetime.utcnow()
             )
         )
        document_split_crud.create(
            db_session, 
            document_split_entity.DocumentSplitCreate(
                name = 'crop_image',
                type_id = 1,
                url = 'http://' + config.FTP_URL + '/' + image_export_dir + 'crop.jpg',
                document_id = document.id, 
                create_date = datetime.datetime.utcnow()
             )
         )
        
        ###########################################################
        # DB
        ###########################################################
        ftp.move(image_path, image_export_dir + 'origin.jpg')
        
        
        ###########################################################
        # DB
        ###########################################################
        document_crud.update(
            db_session,
            document.id,  
            {
                'url': 'http://' + config.FTP_URL + '/' + image_export_dir + 'origin.jpg',
                'update_date': datetime.datetime.utcnow(),
                'status_id': 2,
            }
        )
        ############################################################
        # DB
        ############################################################
    print('sleep 100 second ...') 
    time.sleep(15)

    
