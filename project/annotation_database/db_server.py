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
@app.route("/query/count/<filter>", methods = ['GET'])
def handle_count_query(filter):
	return None

#Query for a single entry
@app.route("/query/entry/<filter>", methods = ['GET'])
def handle_get_entry(filter):
	return None

#Query for all entries given a single search
@app.route("/query/search/<filter>", methods = ['GET'])
def handle_search_entries(filter):
	return None

#Query for entry with no or least annotations
@app.route("/query/annotationless/<filter>", methods = ['GET'])
def handle_get_entry_no_annotation(filter):
	return None
	
#Query for entry with most annotations
@app.route("/query/maxannotation/<filter>", methods = ['GET'])
def handle_get_entry_max_annotations(filter):
	return None
	
#Fully replace annotations given image unique identifier
@app.route("/update/annotationreplace/<filter>", methods=['PUT'])
def annotation_replacement(filter):
	return None

#Mix annotations given when given unique image identifer
@app.route("/update/annotategivenimage/<filter>", methods=['PUT'])
def annotate_given_image(filter):
	return None

#Edit metadata of image given unique image identifer
@app.route("/update/editmetadatagivenimage/<filter>", methods=['PUT'])
def edit_metadata_given_image(filter):
	return None

#Edit metadata tag

@app.route("/update/editmetadatatag/<filter>", methods=['PUT'])
def edit_metadata_tag(filter):
	return None

#Remove metadata
@app.route("/delete/removemetadata/<filter>", methods=['DELETE'])
def remove_meta_data(filter):
	return None

#Remove metadata tag
@app.route("/delete/removemetadatatag/<filter>", methods=['DELETE'])
def remove_meta_data_tag(filter):
	return None

#Remove annotations
@app.route("/delete/removeannotations/<filter>", methods=['DELETE'])
def remove_annotations(filter):
	return None

#Remove image and all info on it
@app.route("/delete/removeimageandinfo/<filter>", methods=['DELETE'])
def remove_image_and_info(filter):
	return None



if __name__ == "__main__":
	app.run(threaded=True, host=HOST, port=PORT) 