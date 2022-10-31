def display(data):
	print("[Server]: " + data)
	
display("Attempting to initialize the server...")

from time import sleep
from flask import Flask, request, send_file
import logging
#from document_scanner import scan

HOST = 'localhost'
PORT = 5000

try:
	app = Flask(__name__)
except Exception as e:
	display("Failed to launch server, terminating process...")
	print(e)
	exit()
display("Successfully launched server")

dev_key = "secretkey"

#Create/Post

#Read/Get

#Update/Put
@app.route("/update/<model_name>", methods = ['PUT'])
def update_model(model_name):
    key = request.form['key']
    if key == dev_key:
        # TODO: update model...
        return {"message": "model successfully updated"}, 200
    return {"error" : "invalid key"}, 403


@app.route("/update/metadata/<model_name>", methods = ['PUT'])
def update_metadata(model_name):
    key = request.form['key']
    if key == dev_key:
        # TODO: update model metadata...
        return {"message": "model metadata successfully updated"}, 200
    return {"error" : "invalid key"}, 403


@app.route("/update/metadata-tag/<model_name>", methods = ['PUT'])
def update_metadata_tag(model_name):
    key = request.form['key']
    if key == dev_key:
        # TODO: update model metadata tag...
        return {"message": "model metadata tag successfully updated"}, 200
    return {"error" : "invalid key"}, 403


#Delete/Delete
@app.route("/delete/<model_name>", methods = ['DELETE'])
def delete_model(model_name):
    key = request.form['key']
    if key == dev_key:
        # delete model...
        return {"message": "model successfully deleted"}, 200
    return {"error" : "invalid key"}, 403


@app.route("/delete/metadata-tag/<model_name>", methods = ['DELETE'])
def delete_metadata_tag(model_name):
    key = request.form['key']
    if key == dev_key:
        # delete model metadata tag...
        return {"message": "model metadata tag successfully deleted"}, 200
    return {"error" : "invalid key"}, 403


if __name__ == "__main__":
	app.run(debug=True, threaded=True, host=HOST, port=PORT) 
