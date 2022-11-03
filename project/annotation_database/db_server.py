def display(data):
	print("[Server]: " + data)
	
display("Attempting to initialize the server...")

from time import sleep
from flask import Flask, request, send_file
import logging
from document_scanner import scan

HOST = '192.168.1.199'
PORT = 5000

try:
	app = Flask(__name__)
except Exception as e:
	display("Failed to launch server, terminating process...")
	print(e)
	exit()
display("Successfully launched server")

#Creates a new entry
@app.route("/create/entry", methods = ['POST'])
def handle_entry():
	f = request.files['file']
	#to complete later...
	return None

#Creates many entries
@app.route("/create/entries", methods = ['POST'])
def handle_entries():
	f = request.files['file']
	#to complete later...
	return None

#Query for count of queries
@app.route("/read/count/<filter>", methods = ['GET'])
def handle_count_query(filter):
	return None

#Query for a single entry
@app.route("/read/entry/<filter>", methods = ['GET'])
def handle_get_entry(filter):
	return None

#Query for all entries given a single search
@app.route("/read/search/<filter>", methods = ['GET'])
def handle_search_entries(filter):
	return None

#Query for entry with no or least annotations
@app.route("/read/annotation/min", methods = ['GET'])
def handle_get_entry_min_annotation():
	return None
	
#Query for entry with most annotations
@app.route("/read/annotation/max", methods = ['GET'])
def handle_get_entry_max_annotations():
	return None
	
#Fully replace annotations given image unique identifier
@app.route("/update/annotation/<id>", methods=['PUT'])
def handle_annotation(id):
	return None

#Mix annotations given when given unique image identifer
@app.route("/update/mix-annotation/<id>", methods=['PUT'])
def handle_mix_annotation(id):
	return None

#Edit metadata of image given unique image identifer
@app.route("/update/metadata/<id>", methods=['PUT'])
def handle_metadata(id):
	return None

#Edit metadata tag

@app.route("/update/metadata-tag/<id>", methods=['PUT'])
def handle_metadata_tag(id):
	return None

#Remove metadata
@app.route("/delete/metadata/<id>", methods=['DELETE'])
def handle_metadata(id):
	return None

#Remove metadata tag
@app.route("/delete/metadata-tag/<id>", methods=['DELETE'])
def handle_metadata_tag(id):
	return None

#Remove annotations
@app.route("/delete/annotations/<id>", methods=['DELETE'])
def handle_annotations(id):
	return None

#Remove image and all info on it
@app.route("/delete/image/<id>", methods=['DELETE'])
def handle_image(id):
	return None

if __name__ == "__main__":
	app.run(threaded=True, host=HOST, port=PORT)