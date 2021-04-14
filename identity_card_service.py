import os
os.chdir('/home/pot/Desktop/web-scan')
import time
from database.db import db_session
from database.repository_crud import document_crud, process_crud
from database.repository_logic import document_logic, process_logic
from database.entities import document_entity, process_entity
import datetime
from identity_card_ai_service import predict_identity_card
from pprint import pprint
from helpers.ftp_utils import FTP
from settings import config


if __name__ == "__main__":
    ftp = FTP(username = config.FTP_USERNAME, password = config.FTP_PASSWORD, url = config.FTP_URL)
    while True:
        try:
            process = process_logic.read_by_status_name_and_type_join_load(db_session, 'upload', config.IDENTITY_CARD)
            for item in process:
                document_id = str(item.document.id)
                process_id = str(item.id)
                url = item.document.url
                list_ans, list_class, category_index = predict_identity_card(open(ftp.load_file(url), 'rb'))
                for i in range(len(list_class)):
                    crop_image = list_ans[i]
                    ftp.create_file('/{server_type}/{document_id}/{field_name}.png'.format(
                        server_type = config.IDENTITY_CARD, document_id = document_id, 
                        field_name = document_id + str(category_index[list_class[i]]['name'])))        
            process_crud.update(
                db_session, 
                document_id, 
                {'status_code': 200, 'status_name': 'extract', 'update_date': datetime.datetime.now()}
            )
            document_crud.update(
                db_session, 
                document_id, 
                {'export_date': datetime.datetime.now(), 'update_date': datetime.datetime.now()}
            )   
        except Excetion as e:
            print(e)
        time.sleep(10)
        