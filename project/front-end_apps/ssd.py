import requests
import json
from io import BytesIO
import cv2


def ssd_preds(images, process_online, single_classification):
    if not isinstance(images, list):
        images = [images]
    if process_online and len(images) == 1:
        byte_arr = BytesIO()
        byte_arr.write(cv2.imencode('.jpg', images[0])[1])
        byte_arr.seek(0)
        res = requests.post(
            'http://127.0.0.1:5001/read/inference/default', files={'image': byte_arr})
        if res.status_code == 200:
            res = json.loads(res.content)
        else:
            return None

    elif process_online:
        byte_arrs = {}
        for index, img in enumerate(images):

            byte_arr = BytesIO()
            byte_arr.write(cv2.imencode('.jpg', img)[1])
            byte_arr.seek(0)
            byte_arrs[f"image_{index}"] = byte_arr

        res = requests.post('http://127.0.0.1:5001/read/batch-inference/default',
                            data={'num_image': len(byte_arrs)}, files=byte_arrs)
        if res.status_code == 200:
            res = json.loads(res.content)
        else:
            return None
    else:
        # Load model and predict
        raise NotImplementedError('TODO: Local Models are not supported')
    return res
