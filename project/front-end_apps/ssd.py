import numpy as np
import requests
import json
from io import BytesIO
from typing import List, Union, Dict, Any
import cv2


def ssd_preds(images: Union[List[np.ndarray], np.ndarray], process_online: bool, single_classification: bool) -> Union[Dict[str, Any], None]:
    """
    Processes images to get object detection predictions using a remote SSD model.

    Parameters:
        images (List[np.ndarray] or np.ndarray): A single image or a list of images in OpenCV format (numpy arrays).
        process_online (bool): If True, sends the images to a remote server for processing.
                               If False, a local model should be used (currently not implemented).
        single_classification (bool): If True, only a single classification is returned per image.
                                      If False, multiple classifications can be returned per image.

    Returns:
        A dictionary containing the predictions for each image, or None
    """

    # Ensure the input is a list of images
    if not isinstance(images, list):
        images = [images]

    # Process a single image online
    if process_online and len(images) == 1:
        byte_arr = BytesIO()
        byte_arr.write(cv2.imencode('.jpg', images[0])[1])
        byte_arr.seek(0)

        # Send the image to the remote server for processing
        res = requests.post(
            'http://127.0.0.1:5001/read/inference/default', files={'image': byte_arr})
        
        # If the request is successful, parse the JSON response
        if res.status_code == 200:
            res = json.loads(res.content)
        else:
            return None

    # Process multiple images online
    elif process_online:
        byte_arrs = {}
        for index, img in enumerate(images):

            byte_arr = BytesIO()
            byte_arr.write(cv2.imencode('.jpg', img)[1])
            byte_arr.seek(0)
            byte_arrs[f"image_{index}"] = byte_arr

        # Send the images to the remote server for processing
        res = requests.post('http://127.0.0.1:5001/read/batch-inference/default',
                            data={'num_image': len(byte_arrs)}, files=byte_arrs)
        if res.status_code == 200:
            res = json.loads(res.content)
        else:
            return None
        
    # If process_online is False, raise an error (local model processing is not implemented)
    else:
        # Load model and predict
        raise NotImplementedError('TODO: Local Models are not supported')
    return res
