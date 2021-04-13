from io import BytesIO
import numpy as np
from merged_model_discharge_record import CompletedModel
from PIL import Image

model = load_model()


def load_model():
    print("Model loading.....")
    model = CompletedModel()
    print("Model loaded")
    return model


def predict_identity_card(image: Image.Image):
    global model
    if model is None:
        model = load_model()
    img = np.asarray(image)
    result = model.predict_cmnd(img)
    return result


def predict_discharge_record(image: Image.Image):
    global model
    if model is None:
        model = load_model()
    img = np.asarray(image)
    result = model.predict_giay_ra_vien(img)
    return result

