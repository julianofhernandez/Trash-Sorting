from tensorflow import keras
import numpy as np

def ssd_preds(images):
    if not isinstance(images, list):
        images = [images]

    return [np.zeros(10) for image in images]
