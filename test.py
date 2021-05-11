from helpers import image_utils
import requests
from settings import config


import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from helpers import corner_utils, ocr_helpers
import requests


def read_bytes_image_from_url(url):
    response = requests.get(url)
    image_bytes = BytesIO(response.content)
    return image_bytes.read()

res = requests.get('http://{host}:{port}/api/v1/ml-split/identity-card/100'.format(host=config.BE_HOST, port = config.BE_PORT))

print(res)