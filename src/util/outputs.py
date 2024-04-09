import json
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path


# default directory for outputting values
OUT_DIR = '/home/USER/vcl/out/experiments/'
MODEL_DIR = '/home/USER/vcl/out/models/'
IMAGE_DIR = '/home/USER/vcl/images/'


def write_as_json(filename, data):
    """
    Dumps the given data into the specified file using JSON formatting. The file
    is created if it does not exist.

    Args:
        filename: path to file to dump JSON into
        data: numeric data to dump
    """
    if not os.path.exists(os.path.dirname(OUT_DIR + filename)):
        print('creating ...')
        os.makedirs(os.path.dirname(OUT_DIR + filename))

    with open(OUT_DIR + filename, "w") as f:
        json.dump(data, f)


def save_model(model, filename):
    path =  Path(MODEL_DIR + filename)
    os.makedirs(path.parent, exist_ok=True)
    torch.save(model,str(path))


def load_model(filename):
    if not os.path.exists(os.path.dirname(MODEL_DIR)):
        raise FileNotFoundError()
    return torch.load(MODEL_DIR + filename)


def save_generated_image(data: np.ndarray, filename: str):
    if not os.path.exists(os.path.dirname(IMAGE_DIR)):
        print('creating ...')
        os.makedirs(os.path.dirname(IMAGE_DIR))

    data = data * 255
    image = Image.fromarray(data)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(IMAGE_DIR + filename)
    np.save(IMAGE_DIR + filename + str('.npy'), data)
