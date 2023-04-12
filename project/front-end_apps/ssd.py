'''
import numpy as np #Importing numpy as np: A library for working with arrays and matrices of numerical data.
import requests #Importing requests: A library for sending HTTP requests and receiving responses in Python.
import json #Importing json: A library for working with JSON data in Python.
from io import BytesIO #Importing BytesIO from io: A class for working with in-memory binary data streams.
from typing import List, Union, Dict, Any #Importing types used for type hinting in Python.


def preds(images: Union[List[np.ndarray], np.ndarray], process_online: bool, single_classification: bool) -> Union[Dict[str, Any], None]:
    """
    Processes images to get object detection predictions using a remote preds model.

    Parameters:
        images (List[np.ndarray] or np.ndarray): A single image or a list of images in OpenCV format (numpy arrays).
        process_online (bool): If True, sends the images to a remote server for processing.
                               If False, a local model should be used (currently not implemented).
        single_classification (bool): If True, only a single classification is returned per image.
                                      If False, multiple classifications can be returned per image.

    Returns:
        A dictionary containing the predictions for each image, or None
    """
    #This code checks if the images input is a list, and if not, it converts it to a list with a single element.
    # Ensure the input is a list of images
    if not isinstance(images, list):
        images = [images]

    #If process_online is True and there is only one image in the input, the code encodes the image in the JPEG format, 
    #creates a byte buffer from it, and sends it to a remote server for processing using a HTTP POST request. If the 
    #request is successful (i.e. the status code is 200), the response content (which is in JSON format) is loaded into 
    #a Python dictionary (res), which is then returned. If the request is not successful, the function returns None.
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

    #If process_online is True and there are multiple images in the input, the code encodes each image in the JPEG 
    #format, creates a byte buffer from each image, and sends all the images to the remote server for batch processing 
    #using a HTTP
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
'''
import numpy as np #Importing numpy as np: A library for working with arrays and matrices of numerical data.
import requests #Importing requests: A library for sending HTTP requests and receiving responses in Python.
import json #Importing json: A library for working with JSON data in Python.
from io import BytesIO #Importing BytesIO from io: A class for working with in-memory binary data streams.
from typing import List, Union, Dict, Any #Importing types used for type hinting in Python.

#This function sends a HTTP POST request to a given URL, with optional data and files attached. If the request is 
#successful (i.e. the status code is 200), it loads the response content (which is in JSON format) into a Python 
#dictionary and returns it. If the request is not successful, it returns None.
def send_request(url: str, files: Dict[str, BytesIO], data: Dict[str, Union[int, str]] = None) -> Union[Dict[str, Any], None]:
    try:
        res = requests.post(url, data=data, files=files)
        res.raise_for_status()  # Raises an HTTPError if the status code is >= 400
        return json.loads(res.content)
    except requests.exceptions.RequestException:
        return None

#This function processes a single image or a list of images to get object detection predictions using a remote preds model. 
#If process_online is True and there is only one image in the input, the function encodes the image in JPEG format, 
#creates a byte buffer from it, and sends it to a remote server for processing using a HTTP POST request. If the 
#request is successful (i.e. the status code is 200), the response content (which is in JSON format) is loaded into a 
#Python dictionary (res), which is then returned. If the request is not successful, the function returns None. The 
#single_classification argument controls whether multiple classifications can be returned per image or not.
def preds(images: Union[List[np.ndarray], np.ndarray], process_online: bool, single_classification: bool) -> Union[Dict[str, Any], None]:
    """
    Processes images to get object detection predictions using a remote preds model.

    Args:
        images: A single image or a list of images in OpenCV format (numpy arrays).
        process_online: If True, sends the images to a remote server for processing.
                        If False, a local model should be used (currently not implemented).
        single_classification: If True, only a single classification is returned per image.
                               If False, multiple classifications can be returned per image.

    Returns:
        A dictionary containing the predictions for each image, or None
    """

    #This code checks if the images input is a list, and if not, it converts it to a list with a single element.
    # Ensure the input is a list of images
    if not isinstance(images, list):
        images = [images]

    '''If process_online is True and there is only one image in the input, the code encodes the image in the JPEG format, 
    creates a byte buffer from it, and sends it to a remote server for processing using a HTTP POST request. If the 
    request is successful (i.e. the status code is 200), the response content (which is in JSON format) is loaded into 
    a Python dictionary (res), which is then returned. If the request is not successful, the function returns None.'''
    # Process images online
    if process_online:
        byte_arrs = {}
        for index, img in enumerate(images):
            byte_arr = BytesIO()
            byte_arr.write(cv2.imencode('.jpg', img)[1])
            byte_arr.seek(0)
            byte_arrs[f"image_{index}"] = byte_arr

        url = 'http://127.0.0.1:5001/read/batch-inference/default' if len(images) > 1 else 'http://127.0.0.1:5001/read/inference/default'
        data = {'num_image': len(byte_arrs)} if len(images) > 1 else None
        res = send_request(url, files=byte_arrs, data=data)

    # Local model processing is not implemented
    else:
        raise NotImplementedError('TODO: Local Models are not supported')

    return res