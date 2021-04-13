import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse
from ai_service import predict_identity_card, predict_discharge_record
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


@app.get("/home")
def home():
    return "hello world"


@app.post("/predict/identity-card")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = image_utils.read_image_file(await file.read())
    prediction = predict_identity_card(image)
    return prediction


@app.post("/predict/discharge_record")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = image_utils.read_image_file(await file.read())
    prediction, list_class = predict_discharge_record(image)
    files = glob.glob('/home/pot/Desktop/web-scan/test_data/ai_test/*')
    for f in files:
        os.remove(f)
    path = '/home/pot/Desktop/web-scan/test_data/ai_test/'
    end = []
    for j in range(len(prediction)):
        image = prediction[j]
        image = Image.fromarray(image)
        image.save(path+list_class[j]+'.png')
        obj = {'image_url': path+list_class[j]+'.png', 'class_name': list_class[j], 'class_id': j}
        end.append(obj)
    return end


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0',port=8080,debug=True)
