from io import BytesIO
import numpy as np
from PIL import Image
from merged_model import CompletedModel


model = None


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


def read_image_file(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image


def read_image_from_dir(path) -> Image.Image:
    image = Image.open(path)
    return image