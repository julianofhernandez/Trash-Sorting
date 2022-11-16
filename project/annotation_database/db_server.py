"""
The Rest API for connecting to the annotation database to upload, read,
edit, and delete images and their annotations.

TODO: finish incomplete skeleton methods
TODO: figure out how to test things
TODO: update error codes to be consistent
TODO: make sure all the stuff with uploading and downloading images works
Note: for format of the files when uploading to the database, include extension
if we allow for images of different extensions

Last updated 11/16
"""

import sqlite3
#from time import sleep
import os
from flask import Flask, jsonify, request, send_file, send_from_directory
import logging
        
HOST = 'localhost'
PORT = 5000

DEV_KEY = "secret_key"

IMAGE_DIR = "images/"

TABLE_NAME = "image_data"

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif'}

# database to connect to for all sqlite connections
# for prod replace with database 
IMAGE_DATA_DB = "imageDB.db"

def display(data):
        print("[Server]: " + data)

display("Attempting to initialize the server...")

try:
	app = Flask(__name__)
	app.config['UPLOAD_FOLDER'] = IMAGE_DIR

        # limit max size for image size
	app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024 
except Exception as e:
	display("Failed to launch server, terminating process...")
	print(e)
	exit()

display("Successfully launched server")


def create_server():
        """
        Sets up the Sqlite database and creates the tables
        """
        # for prod replace :memory: with db directory
        conn = sqlite3.connect(IMAGE_DATA_DB)

        cursor = conn.cursor()

        cursor.execute("""CREATE TABLE IF NOT EXISTS image_data (
                                        name TEXT,
                                        annotations TEXT,
                                        num_annotation INTEGER,
                                        metadata TEXT
                                )""")
        conn.commit()
        conn.close()

# comment this line out to not create the tables
create_server()

def invalid_request(error_msg = 'Invalid Key', error_code = 1, code = 401):
        """
        Returns Format for invalid response. By default returns a response
        for an invalid developer key
        """
        return jsonify({
                'successful': False,
                'error_msg': 'Invalid Key',
                'error_code': 1
        }), code

def allowed_file(filename):
        """
        Returns true if the file is in a valid format
        """
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route("/create/entry", methods = ['POST'])
def handle_entry():
        """
        Creates a new entry in the database. Uploads an image
        to the directory and other data to database
        """
        
        key = request.form['key']
        if key != DEV_KEY:
                return invalid_request()

        # check if required data is there
        if 'file' not in request.files or 'name' not in request.form:
                return invalid_request(error_msg = 'no file selected or missing required data',
                                            error_code = 2, code = 200)
        
        image = request.files['image']

        # check if file is valid
        if image.filename == '' or not allowed_file(image):
                return invalid_request(error_msg = 'invalid file selected',
                                            error_code = 2, code = 200)

        # save image to directory
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], image.filename))

        # collect data from form
        name = request.form['name']
        annotations = request.form['annotations']
        num_annotation = request.form['num_annotation']
        metadata = request.form['metadata']

        # upload data to database
        query = """INSERT INTO image_data
                        (name, annotations, num_annotation, metadata)
                        VALUES
                        (?, ?, ?, ?)"""

        data = [name, annotations, metadata, num_annotation]

        conn = sqlite3.connect(IMAGE_DATA_DB)
        cursor = conn.cursor()

        cursor.execute(query, data)

        conn.commit()
        conn.close()
                
        return jsonify({
                'successful': True,
                'error_msg': None,
                'error_code': None
        }), 200




@app.route("/create/entries", methods = ['POST'])
def handle_entries():
        """
        Creates many entries. Uploads multiple images to the database
        """
        
        key = request.form['key']
        if key != DEV_KEY:
                return invalid_request()

        # TODO: figure this out

        return jsonify({
                'successful': True,
                'error_msg': None,
                'error_code': None
        }), 200



#=====================================================================

@app.route("/read/count/<filter>", methods = ['GET'])
def handle_count_query(filter):
        """
        Query for count of queries.

        TODO: This query is possibly not what is supposed to be intended?
	TODO: also download the image from dir
        """

        conn = sqlite3.connect(IMAGE_DATA_DB)

        cursor = conn.cursor()

        cursor.execute("SELECT * FROM image_data")
        
        rows = cursor.fetchall()

        conn.close()

        data = {}
        i = 0
        for row in rows:
                image_data = 'image' + str(i)
                data.update({image_data:
                             {'name': row[0],
                              'annotations': row[1],
                              'num_annotation': row[2],
                              'metadata': row[3]
                }})
                i += 1
        
        return jsonify({
                'data':data,
                'successful': True,
                'error_msg': None,
                'error_code': None
        }), 200



@app.route("/read/entry/<filter>", methods = ['GET'])
def handle_get_entry(filter):
        """
        Query for a single entry
        """
        
        conn = sqlite3.connect(IMAGE_DATA_DB)

        cursor = conn.cursor()

        cursor.execute("SELECT * FROM image_data WHERE name = :name", {'name': filter})
        
        image_data = cursor.fetchone()

        conn.close()

        if not image_data:
                return jsonify({
                        'successful': False,
                        'error_msg': "No results from query",
                        'error_code': 10
                }), 200
        
        data = {'name': image_data[0], 'annotations': image_data[1],
                'num_annotation': image_data[2], 'metadata': image_data[3]
                }

        # we assume that if the data is in the database, the corresponding image
        # will be there so we do not do any check for if the image is there
        # as_attachment=True
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename = data['name']),
        jsonify({
                'data': data,
                'successful': True,
                'error_msg': None,
                'error_code': None
        }), 200



@app.route("/read/search/<filter>", methods = ['GET'])
def handle_search_entries(filter):
        """
        Query for all entries given a single search
        """
        conn = sqlite3.connect(IMAGE_DATA_DB)

        cursor = conn.cursor()

        cursor.execute("SELECT * FROM image_data WHERE :filter", {'filter': filter})
        
        rows = cursor.fetchall()

        conn.close()

        data = {}
        i = 0
        for row in rows:
                image_data = 'image' + str(i)
                data.update({image_data:
                             {'name': row[0],
                              'annotations': row[1],
                              'num_annotation': row[2],
                              'metadata': row[3]
                }})
                i += 1
        
        return jsonify({
                'data':data,
                'successful': True,
                'error_msg': None,
                'error_code': None
        }), 200



@app.route("/read/annotation/min", methods = ['GET'])
def handle_get_entry_min_annotation():
        """
        Query for entry with no or least annotations
        """
        
        conn = sqlite3.connect(IMAGE_DATA_DB)

        cursor = conn.cursor()

        cursor.execute("SELECT * FROM image_data ORDER BY num_annotation ASC LIMIT 1")
        
        first = cursor.fetchone()
        
        conn.close()

        data = {'name': first[0], 'annotations': first[1],
                'num_annotation': first[2], 'metadata': first[3]
                }
        
        return jsonify({
                'data':data,
                'successful': True,
                'error_msg': None,
                'error_code': None
        }), 200

	


@app.route("/read/annotation/max", methods = ['GET'])
def handle_get_entry_max_annotations():
        """
        Query for entry with most annotations
        """
        
        conn = sqlite3.connect(IMAGE_DATA_DB)

        cursor = conn.cursor()

        cursor.execute("SELECT * FROM image_data ORDER BY num_annotation DESC LIMIT 1")
        
        first = cursor.fetchone()
        
        conn.close()

        data = {'name': first[0], 'annotations': first[1],
                'num_annotation': first[2], 'metadata': first[3]
                }
        
        return jsonify({
                'data':data,
                'successful': True,
                'error_msg': None,
                'error_code': None
        }), 200




#=====================================================================
# TODO: add queries to update methods, pretty much copy and paste with
# small changes to the queries


@app.route("/update/annotation/<id>", methods=['PUT'])
def handle_annotation(id):
        """
        Fully replace annotations given image unique identifier
        """ 
        key = request.form['key']
        if key != DEV_KEY:
                return invalid_request()
        
        error_msg = None
        error_code = 0
        successful = True

                
        return jsonify({
                'successful': successful,
                'error_msg': error_msg,
                'error_code': error_code
        }), 200




@app.route("/update/mix-annotation/<id>", methods=['PUT'])
def handle_mix_annotation(id):
        """
        Mix annotations given when given unique image identifer
        """
        key = request.form['key']
        if key != DEV_KEY:
                return invalid_request()
        
        error_msg = None
        error_code = 0
        successful = True

                
        return jsonify({
                'successful': successful,
                'error_msg': error_msg,
                'error_code': error_code
        }), 200




@app.route("/update/metadata/<id>", methods=['PUT'])
def handle_metadata(id):
        """
        Edit metadata of image given unique image identifer
        """
        key = request.form['key']
        if key != DEV_KEY:
                return invalid_request()
        
        error_msg = None
        error_code = 0
        successful = True

                
        return jsonify({
                'successful': successful,
                'error_msg': error_msg,
                'error_code': error_code
        }), 200




@app.route("/update/metadata-tag/<id>", methods=['PUT'])
def handle_metadata_tag(id):
        """
        Edit metadata tag
        """
        key = request.form['key']
        if key != DEV_KEY:
                return invalid_request()
        
        error_msg = None
        error_code = 0
        successful = True

                
        return jsonify({
                'successful': successful,
                'error_msg': error_msg,
                'error_code': error_code
        }), 200


#=====================================================================


@app.route("/delete/metadata/<id>", methods=['DELETE'])
def delete_metadata(id):
        """
        Remove metadata

        TODO: FINISH
        Possibly might not need this method?
        """
        
        key = request.form['key']
        if key != DEV_KEY:
                return invalid_request()
        
        conn = sqlite3.connect(IMAGE_DATA_DB)

        cursor = conn.cursor()

        cursor.execute("DELETE FROM image_data WHERE name = :name", {'name': id})
        
        conn.commit()
        conn.close()

                
        return jsonify({
                'successful': True,
                'error_msg': None,
                'error_code': None
        }), 200



@app.route("/delete/metadata-tag/<id>", methods=['DELETE'])
def delete_metadata_tag(id):
        """
        Remove metadata tag

        TODO: FINISH
        """
        
        key = request.form['key']
        if key != DEV_KEY:
                return invalid_request()
        
        error_msg = None
        error_code = 0
        successful = True

                
        return jsonify({
                'successful': successful,
                'error_msg': error_msg,
                'error_code': error_code
        }), 200



@app.route("/delete/annotations/<id>", methods=['DELETE'])
def delete_annotations(id):
        """
        Remove metadata tags

        TODO: FINISH
        """
        
        key = request.form['key']
        if key != DEV_KEY:
                return invalid_request()
        
        error_msg = None
        error_code = 0
        successful = True

                
        return jsonify({
                'successful': successful,
                'error_msg': error_msg,
                'error_code': error_code
        }), 200



@app.route("/delete/image/<id>", methods=['DELETE'])
def delete_image(id):
        """
        Remove image and all correlated info on it
        For troubleshooting refer to 
        https://stackoverflow.com/questions/26647248/how-to-delete-files-from-the-server-with-flask
        """
        
        key = request.form['key']
        if key != DEV_KEY:
                return invalid_request()

        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], id))
       
                
        return jsonify({
                'successful': True,
                'error_msg': None,
                'error_code': None
        }), 200



if __name__ == "__main__":
	app.run(threaded=True, host=HOST, port=PORT)
