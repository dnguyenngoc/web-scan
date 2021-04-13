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


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")

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
        image = list_ans[j]
        image = Image.fromarray(image)
        image.save(path + '1_' + list_class[j] + '.png')
        obj = {
            'id': 1,
            'data': {
                'image_url': path+list_class[j]+'.png', 'class_name': list_class[j], 'class_id': j
            } 
        }
        end.append(obj)
    if type_predict == 'detection':
        return end
    else:
        for image in end:
            image_path = image['image_url']
        return 'improve here'
    
@app.get('/result/{id}')
async def result(field_name: str = 'name'):
    path = './test_data/tmp/discharge_record/' + id + '_' + field_name + '.png'
    return StreamingResponse(open(path, 'rb'), media_type="image/png")
    
    
if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0',port=8080,debug=True)
