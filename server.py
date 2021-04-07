import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse
from serve_model import *
import json

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
    image = read_image_file(await file.read())
    prediction = predict(image)
    return prediction

@app.post("/predict/discharge_record")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_image_file(await file.read())
    prediction = predict(image)
    return prediction


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0',port=8080,debug=True)
