def display(data):
	print("[Server]: " + data)
	
display("Attempting to initialize the server...")

from flask import Flask, request, send_file, jsonify, send_from_directory
from io import BytesIO
import os
import numpy as np
from ssd import ssd_preds


HOST = 'localhost'
PORT = 5001 # https://stackoverflow.com/a/72797062

LABELS = {
    0: 'water bottles',
    1: 'platic bags'
}

TRASH_BIN_LABELS = {
    'water bottles': 'recylable',
    'plastic bags': 'recyclable'

}

MODELS_DIR = 'models/'
MODELS = {}
def existing_models():
    global MODELS_DIR, MODELS
    for model in os.listdir(MODELS_DIR):
        MODELS[model.rsplit('.', 1)[0]] = os.path.join(MODELS_DIR, model)
existing_models()
print(MODELS)

METADATAS_DIR = 'metadatas/'

DEV_KEY = 'secretkey'

try:
	app = Flask(__name__)
except Exception as e:
	display("Failed to launch server, terminating process...")
	print(e)
	exit()
display("Successfully launched server")


@app.route("/create/model/<model_name>", methods = ['POST'])
def handle_model_create(model_name):
    """
    This function allows a model and metadata to be uploaded to the server to later be used or downloaded.
    """
    global DEV_KEY, MODELS_DIR, MODELS, METADATAS_DIR
    if 'key' in request.form and request.form['key'] == DEV_KEY:
        error_msg = None
        error_code = 0
        successful = True
        if model_name in MODELS:
            error_msg = 'Model already exists'
            error_code = 5
            successful = False
        elif 'model' not in request.files:
            error_msg = 'Missing model in files part of request'
            error_code = 3
            successful = False
        elif 'metadata' not in request.form:
            error_msg = 'Missing metadata in form part of request'
            error_code = 4
            successful = False
        else:
            try:
                model = request.files['model']
                path = os.path.join(MODELS_DIR, model_name + '.h5')
                model.save(path)
                with open(os.path.join(METADATAS_DIR, model_name + '.json'), 'w') as file:
                    file.write(request.form['metadata'])
                MODELS[model_name] = path
            except Exception as e:
                error_msg = str(e)
                error_code = 2
                successful = False
        return jsonify({
            'successful': successful,
            'error_msg': error_msg,
            'error_code': error_code
        }), 200
    return jsonify({
        'successful': False,
        'error_msg': 'Invalid Key',
        'error_code': 1
    }), 200


@app.route("/read/inference/<model_name>", methods = ['POST'])
def handle_read_inference(model_name):
    """
        This function will take a model name and an image to make a single prediction.
    """
    global LABELS, TRASH_BIN_LABELS
    error_msg = None
    error_code = 0
    predictions = []
    if 'image' not in request.files:
        error_msg = 'Missing image in files part of request'
        error_code = 3
    else:
        try:
            byte_arr = BytesIO()
            request.files['image'].save(byte_arr)
            img = np.frombuffer(byte_arr.getvalue(), np.uint8)

            predictions_single = ssd_preds(img)  # Predict the correct label fo the image, returns bouding box, probs

            for i in range(len(predictions_single)):
                class_prob = predictions_single[i]['probs']    # Get the sample of predictions, ex. [0.7, 0.6, 0.1]
                ndx = np.argmax(class_prob )                   # This will return the highest confidence value
                label = LABELS[ndx]                            # Get the label
                trash_bin_label = TRASH_BIN_LABELS[ndx]        # Get the trash_bin_label ex. recycle...
                bounding_box = predictions_single[i]['bbox']   # Get the coordinate of the prediction in the image
                predictions.append({
                    'obj_label': label,
                    'trash_bin_label': trash_bin_label,
                    'bounding_box': bounding_box,
                    'class_prob': class_prob
                })
        except Exception as e:
            error_msg = str(e)
            error_code = 2
    return jsonify({
        'model_name': model_name,
        'predictions': predictions,
        'error_msg': error_msg,
        'error_code': error_code,
    }), 200


@app.route("/read/batch-inference/<model_name>", methods = ['POST'])
def handle_batch_inference(model_name):
    """
        This function will take a model name and multiple images to make predictions.
    """
    global LABELS, TRASH_BIN_LABELS
    error_msg = None
    error_code = 0
    batch_predictions = []
    if 'image_0' not in request.files:
        error_msg = 'Missing image_0 in files part of request'
        error_code = 3
    elif 'num_image' not in request.form:
        error_msg = 'Missing num_image in form part of request'
        error_code = 4
    else:
        try:
            num_images = int(request.form['num_image'])

            imgs = []
            for i in range(num_images):
                byte_arr = BytesIO()
                request.files[f'image_{i}'].save(byte_arr)
                img = np.frombuffer(byte_arr.getvalue(), np.uint8)
                imgs.append(img)

            all_predictions = ssd_preds(imgs) # Predict the correct label fo the image, returns bouding box, probs

            for predictions_single in all_predictions:
                predictions = []
                for i in range(len(predictions_single)):
                    class_prob = predictions_single[i]['probs']    # Get the sample of predictions, ex. [0.7, 0.6, 0.1]
                    ndx = np.argmax(class_prob)                    # This will return the highest confidence value
                    label = LABELS[ndx]                            # Get the label
                    trash_bin_label = TRASH_BIN_LABELS[ndx]        # Get the trash_bin_label ex. recycle...
                    bounding_box = predictions_single[i]['bbox']   # Get the coordinate of the prediction in the image
                    predictions.append({
                        'obj_label': label,
                        'trash_bin_label': trash_bin_label,
                        'bounding_box': bounding_box,
                        'class_prob': class_prob
                    })
                batch_predictions.append(predictions)
        except Exception as e:
            error_msg = str(e)
            error_code = 2

    return jsonify({
        'model_name': model_name,
        'batch_predictions': batch_predictions,
        'error_msg': error_msg,
        'error_code': error_code,
        'model_name': model_name
    }), 200  
	

@app.route("/read/model/list", methods = ['GET'])
def handle_model_list():
    """
    Get list of models that are usable with their metadata.
    """
    global METADATAS_DIR
    model_list = []

    error_msg = None
    error_code = 0
    try:
        for metadata_file in os.listdir(METADATAS_DIR):
            metadata = open(os.path.join(METADATAS_DIR, metadata_file), 'r').read()
            model_list.append({
                'model_name': metadata_file.rsplit(".", 1)[0],
                'metadata': metadata
            })
    except Exception as e:
        error_msg = str(e)
        error_code = 2

    return jsonify({
        'model_list': model_list,
        'error_msg': error_msg,
        'error_code': error_code
    }), 200
	

@app.route("/read/model/<model_name>", methods = ['GET'])
def handle_download_model(model_name):
    """
        Send the model to the user from the server as a file.
    """
    global MODELS, MODELS_DIR
    if model_name in MODELS:
        # NOT RESTFUL but fine
        return send_from_directory(MODELS_DIR, model_name + '.h5'), 200
    return jsonify({
        'error_msg': 'Model Not Found',
        'error_code': 3
    }), 200


@app.route("/read/metadata/<model_name>", methods = ['GET'])
def handle_metadata(model_name):
    """
        Send the model metadata to the user.
    """
    global MODELS, MODELS_DIR
    if model_name in MODELS:
        error_msg = None
        error_code = 0
        metadata = open(os.path.join(METADATAS_DIR, model_name + '.json'), 'r').read()
        result = {
            'metadata': metadata,
            'error_msg': error_msg,
            'error_code': error_code
        }
        return jsonify(result), 200
    return jsonify({
        'metadata': '',
        'error_msg': 'Model Metadata Not Found',
        'error_code': 3
    }), 200


@app.route("/update/model/<model_name>", methods = ['PUT'])
def handle_update_model(model_name):
    """
    Overwrite the model or metadata of the specified model.
    """
    global DEV_KEY, MODELS_DIR, METADATAS_DIR
    if 'key' in request.form and request.form['key'] == DEV_KEY:
        error_msg = None
        error_code = 0
        successful = True
        try:
            if 'model' in request.files:
                model = request.files['model']
                path = os.path.join(MODELS_DIR, model_name + '.h5')
                model.save(path)
            if 'metadata' in request.form:
                # overwritting instead of updating
                with open(os.path.join(METADATAS_DIR, model_name + '.json'), 'w') as file:
                    file.write(request.form['metadata'])
        except Exception as e:
            error_msg = str(e)
            error_code = 2
            successful = False
        result = {
            'successful': successful,
            'error_msg': error_msg,
            'error_code': error_code
        }
        return jsonify(result), 200
    result = {
        'successful': False,
        'error_msg': 'Invalid Key',
        'error_code': 1
    }
    return jsonify(result), 200


@app.route("/delete/model/<model_name>", methods = ['DELETE'])
def delete_model(model_name):
    """
    Delete the model and metadata of the specified model.
    """
    global DEV_KEY, METADATAS_DIR, MODELS_DIR
    if 'key' in request.form and request.form['key'] == DEV_KEY:
        error_msg = None
        error_code = 0
        successful = True
        try:
            del MODELS[model_name]
            #delete metadata
            os.remove(os.path.join(METADATAS_DIR, model_name + '.json'))
            #delete model
            os.remove(os.path.join(MODELS_DIR, model_name + '.h5'))
        except Exception as e:
            error_msg = str(e)
            error_code = 2
            successful = False
        result = {
            'successful': successful,
            'error_msg': error_msg,
            'error_code': error_code
        }
        return jsonify(result), 200
    result = {
        'successful': False,
        'error_msg': 'Invalid Key',
        'error_code': 1
    }
    return jsonify(result), 200


if __name__ == "__main__":
	app.run(debug=True, threaded=True, host=HOST, port=PORT)
