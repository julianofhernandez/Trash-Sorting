import cv2
import numpy as np
import requests
import json
from io import BytesIO
from typing import List, Union, Dict, Any
from PIL import Image

from model_offline import preds as offpreds


HOST = "127.0.0.1"
PORT = 8000


def send_request(
    url: str, files: Dict[str, BytesIO], data: Dict[str, Union[int, str]] = None
) -> Union[Dict[str, Any], None]:
    """
    Sends an HTTP POST request to the specified URL with optional data and files attached.
    If the request is successful (status code is 200), it returns the response content as a dictionary.
    If the request is not successful, it returns None.

    Args:
        url: The URL to send the request to.
        files: A dictionary of file-like objects to send as files in the request.
        data: Optional dictionary of data to send as form data in the request.

    Returns:
        A dictionary containing the response content or None if the request was unsuccessful.
    """
    try:
        res = requests.post(url, data=data, files=files)
        res.raise_for_status()
        return json.loads(res.content)
    except requests.exceptions.RequestException:
        return None


def preds(
    images: Union[List[np.ndarray], np.ndarray],
    process_online: bool,
    model_offline: str,
) -> Union[Dict[str, Any], None]:
    """
    Processes images to get object detection predictions using a remote preds model.

    Args:
        images: A single image or a list of images in OpenCV format (numpy arrays).
        process_online: If True, sends the images to a remote server for processing.
                        If False, a local model should be used (currently not implemented).
        model_offline: The name of the local model to use if process_online is False.

    Returns:
        A dictionary containing the predictions for each image, or None
    """

    # Ensure the input is a list of images
    if not isinstance(images, list):
        images = [images]

    # Process images online
    if process_online:
        # Convert images to byte arrays and store in a dictionary
        byte_arrs = {}
        for index, img in enumerate(images):
            byte_arr = BytesIO()
            byte_arr.write(cv2.imencode(".jpg", img)[1])
            byte_arr.seek(0)
            byte_arrs[f"image_{index}"] = byte_arr

        # Set the URL based on the number of images
        url = f"http://{HOST}:{PORT}/read/batch-inference/default"
        data = {"num_image": len(byte_arrs)}

        # Send the request and return the response content as a dictionary
        res = send_request(url, files=byte_arrs, data=data)

    # Local model processing is not implemented
    else:
        images = [
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in images
        ]
        res = offpreds(images, model_offline)[0]

    return res
