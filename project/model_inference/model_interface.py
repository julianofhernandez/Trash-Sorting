def display(data):
	print("[Server]: " + data)
	
display("Attempting to initialize the server...")

import logging
from time import sleep
from flask import Flask, request, send_file, jsonify, send_from_directory
from tensorflow import keras
import numpy as np
import os
import json

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

MODELS_DIR = '.'
METADATAS_DIR = '.'

try:
	app = Flask(__name__)
except Exception as e:
	display("Failed to launch server, terminating process...")
	print(e)
	exit()
display("Successfully launched server")

dev_key = "secretkey"

#Create/Post
@app.route("/create/<entry_name>", methods = ['POST'])
def handle_create(model_name):
    """
       This function creates a model
    """
	key_create = request.form['key']
	if key_create == dev_key:
        
		return {
            'successful': 'TRUE/FALSE',
            'error_msg': 'None or str',
            'error_code': 'None or int'
            }, 200

	return {'msg': 'invalid key'}, 403

#Read/Get
@app.route("/read/inf/<entry_name>", methods = ['POST'])
def handle_read_inference(model_name):
    """
        This function will Make prediction on single image.
    """
    model_name = ssd_preds.name

    img = np.fromstring(request.form["image"], np.uint8)         #Grab the image
    
    error_msg = None
    error_code = 0
    try:
        predictions_single = ssd_preds(img)     #Predict the correct label fo the image, returns bouding box, class, probs

        #tf.keras returns a list of lists so only select [0]
        predictions = []
        for i in range(len(predictions_single)):
            ndx = np.argmax(predictions_single[i])  #This will return the highest confidence value
            label = LABELS[ndx]                     #Get the label
            trash_bin_label = TRASH_BIN_LABELS[ndx]        #get the trash_bin_label ex. recycle...
            boudning_box = predictions_single[i]            #Get the coordinate of the prediction in the image
            class_prob = predictions_single                 #Get the sample of predictions, ex. [0.7, 0.6, 0.1]
            predictions.append({
                'obj_label': label,
                'trash_bin_label': trash_bin_label,
                'bounding_box': boudning_box,
                'class_prob': class_prob
            })
    except Exception as e:
        error_msg = str(e)
        error_code = 1
               

    #assuming predictions_single will return bouding_box, class, bin_class, prob, and error msg
    result = {
        'model_name': model_name,
        'image': img,
        'predictions': predictions,
        'error_msg': error_msg,
        'error_code': error_code,
        'model_name': model_name
    }

	return jsonify(result), 200             #jsonify returns a Response object with the application/json mimetype set, convert dict -> json
	

@app.route("/read/batch_inf/<entry_name>", methods = ['POST'])
def handle_batch_inference(model_name):
	#TODO: Implement output
    """
        This function will predict multiple images
    """
    model_name = ssd_preds.name
	
    num_images = request.form['num_image']

    img = []
    for i in range(num_images):
        img.append(np.fromstring(request.form[f'image_{i}']))

    
    error_msg = None
    error_code = 0
    try:
        predictions = ssd_preds(img)     #Predict the correct label fo the image, returns bouding box, class, probs

        #tf.keras returns a list of lists so only select [0]
        batch_predictions = []
        for predictions_single in predictions:
            predictions = []
            for i in range(0, len(predictions_single)):
                ndx = np.argmax(predictions_single[i])  #This will return the most highest confidence value
                label = LABELS[ndx]                     #Get the label of the highest confidence value
                trash_bin_label = TRASH_BIN_LABELS[ndx]        #----- how to get the trash bin label?
                boudning_box = predictions_single[i]    #----- how to get the bounding box? 
                class_prob = predictions_single   
                predictions.append({
                    'obj_label': label,
                    'trash_bin_label': trash_bin_label,
                    'bounding_box': boudning_box,
                    'class_prob': class_prob
                })
            batch_predictions.append(predictions)
    except Exception as e:
        error_msg = str(e)
        error_code = 1
               

    #assuming predictions_single will return bouding_box, class, bin_class, prob, and error msg
    result = {
        'model_name': model_name,
        'images': img,
        'batch_predictions': batch_predictions,
        'error_msg': error_msg,
        'error_code': error_code,
        
        
    }
	return jsonify(result), 200
	

@app.route("/read/list/<entry_name>", methods = ['GET'])
def handle_usable_models(model_list):
    nModel = request.form['new_model']
    """
    Get list of models that are usable
    """
    model_list = []
    for i in range(ssd_preds):
        model_list.append({
            'model_name': ssd_preds[i].name,
            'metadata': {
                'cpu/gpu': 'gpu',
                'optimized': False,
                'version': '0.1'
            }
        })

    result = {'model_list': model_list}
	return jsonify(result), 200
	

@app.route("/read/download/<entry_name>", methods = ['GET'])
def handle_download_models(model_name):
    """

        https://stackoverflow.com/questions/24577349/flask-download-a-file
        send the model to the user from the server 
        MDOEL_DIR, specify the file name 
        just return the model 

        @app.route('/uploads/<path:filename>', methods=['GET', 'POST'])
        def download(filename):
        uploads = os.path.join(current_app.root_path, app.config['UPLOAD_FOLDER'])
        return send_from_directory(directory=uploads, filename=filename)


        return send_from_directory(directory=MODELS_DIR, filename=model_name + '.h5') 
    """
	if model_name:
		return send_from_directory(directory=MODELS_DIR, filename=model_name + '.h5'), 200
	return {'msg': 'invalid model'}, 403

#Update/Put
@app.route("/update/<model_name>", methods = ['PUT'])
def update_model(model_name):
    key = request.form['key']
    if key == dev_key:
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

        result = {
            'successful': True,
            'error_message': None,
            'error_code': None
        }

        return jsonify(result), 200
    result = {
        'successful': False,
        'error_message': 'Invalid Key',
        'error_code': 1
    }

    return jsonify(result), 200

@app.route("/update/metadata/<model_name>", methods = ['PUT'])
def update_metadata(model_name):
    key = request.form['key']
    if key == dev_key:
        nmetadata = request.form['new_metadata']

        #load old metadata
        old_metadata = json.loads(open(os.path.join(METADATAS_DIR, model_name + '.json'), 'r'))
        old_metadata.update(nmetadata)
        #save new metadata
        with open(os.path.join(METADATAS_DIR, model_name + '.json'), 'w') as file:
            file.write(jsonify(old_metadata))

        result = {
            'successful': True,
            'error_message': None,
            'error_code': None
        }

        return jsonify(result), 200
    result = {
        'successful': False,
        'error_message': 'Invalid Key',
        'error_code': 1
    }

    return jsonify(result), 200

#Delete/Delete
@app.route("/delete/<model_name>", methods = ['DELETE'])
def delete_model(model_name):
    key = request.form['key']
    if key == dev_key:
        #delete metadata
        os.remove(os.path.join(METADATAS_DIR, model_name))
        #delete model
        os.remove(os.path.join(MODELS_DIR, model_name))


        result = {
            'successful': True,
            'error_message': None,
            'error_code': None
        }

        return jsonify(result), 200
    result = {
        'successful': False,
        'error_message': 'Invalid Key',
        'error_code': 1
    }

    return jsonify(result), 200

@app.route("/delete/metadata/<model_name>", methods = ['DELETE'])
def delete_metadata(model_name):
    key = request.form['key']
    if key == dev_key:
        
        #delete metadata
        os.remove(os.path.join(METADATAS_DIR, model_name))

        result = {
            'successful': True,
            'error_message': None,
            'error_code': None
        }

        return jsonify(result), 200
    result = {
        'successful': False,
        'error_message': 'Invalid Key',
        'error_code': 1
    }

    return jsonify(result), 200


@app.route("/delete/metadata-tag/<model_name>", methods = ['DELETE'])
def delete_metadata_tag(model_name):
    key = request.form['key']
    if key == dev_key:
        tag = request.form['tag']
        #load old metadata
        old_metadata = json.loads(open(os.path.join(METADATAS_DIR, model_name + '.json'), 'r'))
        del old_metadata[tag]

        #save new metadata
        with open(os.path.join(METADATAS_DIR, model_name + '.json'), 'w') as file:
            file.write(jsonify(old_metadata))

        result = {
            'successful': True,
            'error_message': None,
            'error_code': None
        }

        return jsonify(result), 200
    result = {
        'successful': False,
        'error_message': 'Invalid Key',
        'error_code': 1
    }

    return jsonify(result), 200
    
if __name__ == "__main__":
	app.run(debug=True, threaded=True, host=HOST, port=PORT) 