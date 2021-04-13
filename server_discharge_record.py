import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse
from ai_service import predict_discharge_record
from helpers import image_utils
import json


app_desc = """<h2>Try this app by uploading any image with `predict/image`</h2>"""

app = FastAPI(title="Tensorflow FastAPI", description=app_desc)


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")


fake_session = {
    'session_id': 'thisisafakesessionid'
    'status': 'running', 
    'session_name': 'crop_field',
    'description': 'crop_field by mobilenet ssd 320x320 -.~!'
}

@app.post("/predict/discharge_record")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = image_utils.read_image_file(await file.read())
    prediction = predict_discharge_record(image)
    fake_session['session_name'] == 'recognition_field'
    
    return prediction

@app.get("/status/session/{id}")
def status_session(id: str):
    return fake_session


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0',port=8080,debug=True)
