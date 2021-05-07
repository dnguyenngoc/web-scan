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
from database.repository_logic import document_logic
import datetime


print('load model ...')
model = CompletedModel()
print('model is loaded.')


while True:
    ftp = FTP(config.FTP_URL, config.FTP_USERNAME, config.FTP_PASSWORD)


    ############################################################################################
    # LOAD IMAGE HANDLE
    ############################################################################################
    imports = document_logic.get_all_type_and_status(db_session, 1, 1) # import list good


    for item in imports:
        if item.url == None:
            continue
        image = image_utils.read_bytes_image_from_url(item.url)
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
        ftp.chdir(date_export_dir)

        # name of document
        name_doc = item.name.split('.')[0]

        # crop field
        ftp.upload_np_image(img_crop, date_export_dir + name_doc + '_crop.png', 'PNG')
        img_id_boxes = image_utils.handle_detection(id_boxes, img_crop)
        ftp.upload_np_image(img_id_boxes, date_export_dir + name_doc + '_id.png', 'PNG')
        img_name_boxes = image_utils.handle_detection(name_boxes, img_crop)
        ftp.upload_np_image(img_name_boxes, date_export_dir + name_doc + '_name.png', 'PNG')
        img_birth_boxes = image_utils.handle_detection(birth_boxes, img_crop)
        ftp.upload_np_image(img_birth_boxes, date_export_dir + name_doc +'_birth.png', 'PNG')
        img_add_boxes = image_utils.handle_detection(add_boxes, img_crop)
        ftp.upload_np_image(img_add_boxes, date_export_dir + name_doc + '_add.png', 'PNG')
        img_home_boxes = image_utils.handle_detection(home_boxes, img_crop)
        ftp.upload_np_image(img_home_boxes, date_export_dir + name_doc + '_home.png', 'PNG')
        list_extract = ['_crop.png', '_id.png', '_name.png', '_birth.png', '_add.png', '_home.png']


        ############################################################
        # DB
        ############################################################
        for extract_type in list_extract:
            document_split_crud.create(
                db_session,
                document_split_entity.DocumentSplitCreate(
                    name = item.name +  '_id.png',
                    type_id = 1,
                    url = 'http://{be_host}:{be_port}/api/v1/workflow-v1/image/{type_doc}/{status_name}/{name}'.format(
                        be_host = config.BE_HOST,
                        be_port = config.BE_PORT,
                        type_doc = config.IDENTITY_CARD,
                        status_name = config.EXPORT_TYPE_NAME,
                        name = name_doc + extract_type
                    ),
                    document_id = item.id,
                    is_extrated = False,
                    value = None,
                    create_date = datetime.datetime.utcnow()
                )
           )
        document_crud.update(db_session, item.id, {'status_id': 2, 'update_date': datetime.datetime.utcnow()})
    ftp.close()
    time.sleep(120)
