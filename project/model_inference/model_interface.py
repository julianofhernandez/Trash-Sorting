def display(data):
	print("[Server]: " + data)
	
display("Attempting to initialize the server...")

from time import sleep
from flask import Flask, request, send_file, jsonify, send_from_directory
from tensorflow import keras
import numpy as np
import os
import json
from ssd import ssd_preds

# TODOs at some point
# handle missing key better
# handle incorrect or missing model name
# handke image transfer better
# handle model downlaod with restful api
# Keep track of models and metadata on server

HOST = 'localhost'
PORT = 5000

LABELS = {
    0: 'water bottles',
    1: 'platic bags'
}

TRASH_BIN_LABELS = {
    'water bottles': 'recylable',
    'plastic bags': 'recyclable'

}

MODELS_DIR = 'models/'
METADATAS_DIR = 'metadatas/'

DEV_KEY = "secretkey"

try:
	app = Flask(__name__)
except Exception as e:
	display("Failed to launch server, terminating process...")
	print(e)
	exit()
display("Successfully launched server")



#Create/Post
@app.route("/create/model/<model_name>", methods = ['POST'])
def handle_model_create(model_name):
    """
    This function creates a model
    """
    key = request.form['key']
    if key == DEV_KEY:
        error_msg = None
        error_code = 0
        successful = True
        try:
            model = request.form["model"]
            model.save(os.path.join(MODELS_DIR, model_name))

            # keep track on server

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

#Read/Get
@app.route("/read/inference/<model_name>", methods = ['POST'])
def handle_read_inference(model_name):
    """
        This function will Make prediction on single image.
    """

    # assuming image is sent as bytes
    img = np.fromstring(request.form["image"], np.uint8)         #Grab the image
    
    error_msg = None
    error_code = 0
    try:
        predictions_single = ssd_preds(img)     #Predict the correct label fo the image, returns bouding box, probs

        #tf.keras returns a list of lists so only select [0]
        predictions = []
        for i in range(len(predictions_single)):
            class_prob = predictions_single[i]['probs']    #Get the sample of predictions, ex. [0.7, 0.6, 0.1]
            ndx = np.argmax(class_prob )  #This will return the highest confidence value
            label = LABELS[ndx]                     #Get the label
            trash_bin_label = TRASH_BIN_LABELS[ndx]        #get the trash_bin_label ex. recycle...
            bounding_box = predictions_single[i]['bbox']   #Get the coordinate of the prediction in the image
            predictions.append({
                'obj_label': label,
                'trash_bin_label': trash_bin_label,
                'bounding_box': bounding_box,
                'class_prob': class_prob
            })
    except Exception as e:
        error_msg = str(e)
        error_code = 2

    #jsonify returns a Response object with the application/json mimetype set, convert dict -> json
    return jsonify({
        'model_name': model_name,
        'predictions': predictions,
        'error_msg': error_msg,
        'error_code': error_code,
        'model_name': model_name
    }), 200             
	

@app.route("/read/batch-inference/<model_name>", methods = ['POST'])
def handle_batch_inference(model_name):
    """
        This function will predict multiple images
    """
    model_name = ssd_preds.name
	
    num_images = request.form['num_image']

    imgs = []
    for i in range(num_images):
        imgs.append(np.fromstring(request.form[f'image_{i}']))
    
    error_msg = None
    error_code = 0
    try:
        predictions_single = ssd_preds(imgs)     #Predict the correct label fo the image, returns bouding box, probs

        #tf.keras returns a list of lists so only select [0]
        batch_predictions = []
        for predictions_single in predictions:
            predictions = []
            for i in range(len(predictions_single)):
                class_prob = predictions_single[i]['probs']    #Get the sample of predictions, ex. [0.7, 0.6, 0.1]
                ndx = np.argmax(class_prob )  #This will return the highest confidence value
                label = LABELS[ndx]                     #Get the label
                trash_bin_label = TRASH_BIN_LABELS[ndx]        #get the trash_bin_label ex. recycle...
                bounding_box = predictions_single[i]['bbox']   #Get the coordinate of the prediction in the image
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

    #jsonify returns a Response object with the application/json mimetype set, convert dict -> json
    return jsonify({
        'model_name': model_name,
        'batch_predictions': predictions,
        'error_msg': error_msg,
        'error_code': error_code,
        'model_name': model_name
    }), 200  
	

@app.route("/read/model/list", methods = ['GET'])
def handle_model_list():
    """
    Get list of models that are usable
    """
    model_list = []

    error_msg = None
    error_code = 0
    try:
        for metadata_file in os.listdir(METADATAS_DIR):
            metadata = json.loads(open(os.path.join(METADATAS_DIR, metadata_file), 'r'))
            model_list.append({
                'model_name': metadata_file.rsplit(".", 1),
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
        https://stackoverflow.com/questions/24577349/flask-download-a-file
        send the model to the user from the server 
        MDOEL_DIR, specify the file name 
        just return the model 
    """
    if model_name:
        # Restify at some point
        return send_from_directory(directory=MODELS_DIR, filename=model_name + '.h5'), 200
    return jsonify({
        'error_msg': 'Model Not Found',
        'error_code': 3
    }), 200

#Update/Put
@app.route("/update/model/<model_name>", methods = ['PUT'])
def handle_update_model(model_name):
    key = request.form['key']
    if key == DEV_KEY:
        error_msg = None
        error_code = 0
        successful = True
        try:
            nmodel = request.form['new_model']
            nmetadata = request.form['new_metadata']

            #load old metadata
            old_metadata = json.loads(open(os.path.join(METADATAS_DIR, model_name + '.json'), 'r'))
            old_metadata.update(nmetadata)
            #save new metadata
            with open(os.path.join(METADATAS_DIR, model_name + '.json'), 'w') as file:
                file.write(jsonify(old_metadata))

            #save model
            nmodel.save(os.path.join(MODELS_DIR, model_name))
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

@app.route("/update/metadata/<model_name>", methods = ['PUT'])
def update_metadata(model_name):
    key = request.form['key']
    if key == DEV_KEY:
        error_msg = None
        error_code = 0
        successful = True
        try:
            nmetadata = request.form['new_metadata']

            #load old metadata
            old_metadata = json.loads(open(os.path.join(METADATAS_DIR, model_name + '.json'), 'r'))
            old_metadata.update(nmetadata)
            #save new metadata
            with open(os.path.join(METADATAS_DIR, model_name + '.json'), 'w') as file:
                file.write(jsonify(old_metadata))
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

#Delete/Delete
@app.route("/delete/<model_name>", methods = ['DELETE'])
def delete_model(model_name):
    key = request.form['key']
    if key == DEV_KEY:
        error_msg = None
        error_code = 0
        successful = True
        try:
            #delete metadata
            os.remove(os.path.join(METADATAS_DIR, model_name))
            #delete model
            os.remove(os.path.join(MODELS_DIR, model_name))
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

@app.route("/delete/metadata/<model_name>", methods = ['DELETE'])
def delete_metadata(model_name):
    key = request.form['key']
    if key == DEV_KEY:
        error_msg = None
        error_code = 0
        successful = True
        try:
            #delete metadata
            os.remove(os.path.join(METADATAS_DIR, model_name))
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


@app.route("/delete/metadata-tag/<model_name>", methods = ['DELETE'])
def delete_metadata_tag(model_name):
    key = request.form['key']
    if key == DEV_KEY:
        error_msg = None
        error_code = 0
        successful = True
        try:
            tag = request.form['tag']
            #load old metadata
            old_metadata = json.loads(open(os.path.join(METADATAS_DIR, model_name + '.json'), 'r'))
            del old_metadata[tag]

            #save new metadata
            with open(os.path.join(METADATAS_DIR, model_name + '.json'), 'w') as file:
                file.write(jsonify(old_metadata))
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