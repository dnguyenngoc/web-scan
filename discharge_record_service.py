import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse
from ai_service import predict_discharge_record
from helpers import image_utils
import json
from fastapi.responses import StreamingResponse
from PIL import Image
from io import BytesIO
import numpy as np
import os
import glob


app_desc = """<h2>Try this app by uploading any image with `predict/image`</h2>"""
app = FastAPI(title="Tensorflow FastAPI", description=app_desc)


# @app.get("/docs", include_in_schema=False)
# async def index():
#     return RedirectResponse(url="/files")


@app.post("/upload-files/")
async def create_upload_files(files: List[UploadFile] = File(...)):
    return {"filenames": [file.filename for file in files]}


@app.get("/")
async def main():
    content = """
        <body>
        <form action="/files/" enctype="multipart/form-data" method="post">
        <input name="files" type="file" multiple>
        <input type="submit">
        </form>
        <form action="/uploadfiles/" enctype="multipart/form-data" method="post">
        <input name="files" type="file" multiple>
        <input type="submit">
        </form>
        </body>
    """
    return HTMLResponse(content=content)
    

@app.post("/predict")
async def predict_api(file: UploadFile = File(...), type_predict: str= 'detection'):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = image_utils.read_image_file(await file.read())
    list_ans, list_class, category_index = predict_discharge_record(image)
    path = './test_data/tmp/discharge_record/'
    files = glob.glob(path + '*')
    for f in files:
        os.remove(f)
    end = []
    for j in range(len(list_class)):
        class_id = list_class[j]
        image = list_ans[j]
        image = Image.fromarray(image)
        image.save(path + '1_' + str(category_index[class_id]['name']) + '.png')
        obj = {
            'session_id': 1,
            'data': {
                'class_name': str(category_index[class_id]['name']), 
                'class_id': class_id,
                'image_path': path+'1_'+str(category_index[class_id]['name'])+'.png', 
                'image_url': 'http://10.1.33.76:8080/images/1_' +  str(category_index[class_id]['name']) + '.png'
            } 
        }
        end.append(obj)
    if type_predict == 'detection':
        return end
    else:
        for image in end:
            image_path = image['image_url']
        return 'improve here'
    
    
# @app.get('/result/{session_id}')
# async def result( session_id: str, field_name: str):
#     path = './test_data/tmp/discharge_record/' + session_id + '_' + field_name + '.png'
#     return StreamingResponse(open(path, 'rb'), media_type="image/png")
    
    
@app.get('/images/{name}')
async def images(name: str):
    path = './test_data/tmp/discharge_record/' + name
    return StreamingResponse(open(path, 'rb'), media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0',port=8080,debug=True)
