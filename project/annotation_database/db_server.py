"""
The Rest API for connecting to the annotation database to upload, read,
edit, and delete images and their annotations.

Note: for format of the files when uploading to the database, include extension
if we allow for images of different extensions
"""

def display(data):
    print("[Server]: " + data)

display("Attempting to initialize the server...")

import sqlite3
import os
from flask import Flask, jsonify, request, send_file, send_from_directory

    
HOST = 'localhost'
PORT = 5000

DEV_KEY = "secretkey"

IMAGE_DIR = "images/"

TABLE_NAME = "image_data"

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif'}

# database to connect to for all sqlite connections
# for prod replace with database :memory:
IMAGE_DATA_DB = "imageDB.db"

def create_server():
    """
    Sets up the Sqlite database and creates the tables
    """
    conn = sqlite3.connect(IMAGE_DATA_DB)
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS image_data (
                    id INTEGER,
                    annotation TEXT,
                    num_annotations INTEGER,
                    dataset TEXT,
                    metadata TEXT
                )""")
    conn.commit()
    conn.close()

create_server()

def get_max_entries():
    conn = sqlite3.connect(IMAGE_DATA_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(id) FROM image_data")
    num_entries = cursor.fetchone()[0]
    conn.close()
    return 0 if num_entries is None else num_entries

NUM_ENTRIES = get_max_entries() + 1


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


def invalid_request(error_msg = 'Invalid Key', error_code = 1, code = 401):
    """
    Returns Format for invalid response. By default returns a response
    for an invalid developer key
    """
    return jsonify({
        'successful': False,
        'error_msg': error_msg,
        'error_code': error_code
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
    global NUM_ENTRIES
    if 'key' not in request.form or request.form['key'] != DEV_KEY:
        return invalid_request()
    elif 'image' not in request.files:
        return invalid_request(error_msg = 'Missing image in file part of request',
                               error_code = 3, code = 200)
    elif 'annotation' not in request.form:
        return invalid_request(error_msg = 'Missing annotation in form request',
                               error_code = 4, code = 200)
    elif 'num_annotations' not in request.form:
        return invalid_request(error_msg = 'Missing num_annotations in form request',
                               error_code = 5, code = 200)
    elif 'dataset' not in request.form:
        return invalid_request(error_msg = 'Missing dataset in form request',
                               error_code = 6, code = 200)
    elif 'metadata' not in request.form:
        return invalid_request(error_msg = 'Missing metadata in form request',
                               error_code = 7, code = 200)
    #elif not request.files['image'].filename or not allowed_file(request.files['image'].filename):
    #    return invalid_request(error_msg = 'Invalid image file.',
    #                           error_code = 8, code = 200)
    else:
        error_msg = None
        error_code = 0
        data = 0
        try:
            image = request.files['image']

            # collect data from form
            annotation = request.form['annotation']
            num_annotations = request.form['num_annotations']
            dataset = request.form['dataset']
            metadata = request.form['metadata']

            # upload data to database
            query = """INSERT INTO image_data
                    (id, annotation, num_annotations, dataset, metadata)
                    VALUES
                    (?, ?, ?, ?, ?)"""

            entry = [NUM_ENTRIES, annotation, num_annotations, dataset, metadata]
            NUM_ENTRIES += 1

            # save image to directory
            ext = image.filename.rsplit('.', 1)
            data = str(entry[0])
            if len(ext) == 2:
                image.save(os.path.join(IMAGE_DIR, data + '.' + ext[1]))
            else:
                image.save(os.path.join(IMAGE_DIR, data))

            conn = sqlite3.connect(IMAGE_DATA_DB)
            cursor = conn.cursor()
            cursor.execute(query, entry)
            conn.commit()
            conn.close()
        except Exception as e:
            error_msg = str(e)
            error_code = 2
            data = 0 
        return jsonify({
            'data': data,
            'error_msg': error_msg,
            'error_code': error_code,
        }), 200


@app.route('/read/entry/image/<id>')
def handle_get_entry_image(id):
    error_msg = None
    error_code = 0
    try:
        file = [filename for filename in os.listdir(IMAGE_DIR) if filename.startswith(id)][0]
        return send_from_directory(IMAGE_DIR, file), 200
    except IndexError as e:
        error_msg = str(e)
        error_code = 3
    except Exception as e:
        error_msg = str(e)
        error_code = 2
    return jsonify({
        'error_msg': error_msg,
        'error_code': error_code
    }), 200


@app.route('/read/entry/<id>')
@app.route('/read/entry/data/<id>')
def handle_get_entry_metadata(id):
    """
    Query for a single entry's data
    """
    error_msg = None
    error_code = 0
    data = None
    try:
        conn = sqlite3.connect(IMAGE_DATA_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM image_data WHERE id = :id", {'id': id})
        entry = cursor.fetchone()
        conn.close()
        if entry:
            data = {
                'id': entry[0], 'annotation': entry[1],
                'num_annotations': entry[2], 'dataset': entry[3],
                'metadata': entry[4]
            }
        else:
            error_msg = 'No results from query'
            error_code = 3
    except Exception as e:
        error_msg = str(e)
        error_code = 2
    return jsonify({
        'data': data,
        'error_msg': error_msg,
        'error_code': error_code
    }), 200


@app.route("/read/search/<filter>", methods = ['GET'])
def handle_search_entries(filter):
    """
    Query for all entries given a single search
    """
    error_msg = None
    error_code = 0
    data = None
    try:
        conn = sqlite3.connect(IMAGE_DATA_DB)
        cursor = conn.cursor()
        cursor.execute(f'SELECT * FROM image_data WHERE {filter}')
        rows = cursor.fetchall()
        conn.close()

        data = []
        for row in rows:
            data.append(
                {'id': row[0],
                'annotation': row[1],
                'num_annotations': row[2],
                'dataset': row[3],
                'metadata': row[4]}
            )
    except Exception as e:
        error_msg = str(e)
        error_code = 2

    return jsonify({
        'data': data,
        'error_msg': error_msg,
        'error_code': error_code
    }), 200


@app.route("/read/annotation/min", methods = ['GET'])
def handle_get_entry_min_annotation():
    """
    Query for entry with no or least annotations
    """
    data = None
    error_msg = None
    error_code = 0
    try:
        conn = sqlite3.connect(IMAGE_DATA_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM image_data ORDER BY num_annotations ASC LIMIT 1")
        entry = cursor.fetchone()
        conn.close()

        if entry:
            data = {
                'id': entry[0], 'annotation': entry[1],
                'num_annotations': entry[2], 'dataset': entry[3],
                'metadata': entry[4]
            }
        else:
            error_msg = 'No results from query'
            error_code = 3
    except Exception as e:
        error_msg = str(e)
        error_code = 2
    
    return jsonify({
        'data': data,
        'error_msg': error_msg,
        'error_code': error_code
    }), 200


@app.route("/read/annotation/max", methods = ['GET'])
def handle_get_entry_max_annotations():
    """
    Query for entry with most annotations
    """
    data = None
    error_msg = None
    error_code = 0
    try:
        conn = sqlite3.connect(IMAGE_DATA_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM image_data ORDER BY num_annotations DESC LIMIT 1")
        entry = cursor.fetchone()
        conn.close()

        if entry:
            data = {
                'id': entry[0], 'annotation': entry[1],
                'num_annotations': entry[2], 'dataset': entry[3],
                'metadata': entry[4]
            }
        else:
            error_msg = 'No results from query'
            error_code = 3
    except Exception as e:
        error_msg = str(e)
        error_code = 2
    
    return jsonify({
        'data': data,
        'error_msg': error_msg,
        'error_code': error_code
    }), 200


@app.route("/update/approve/<id>", methods=['PUT'])
def handle_annotation_approved(id):
    """
    Increment Annotation approval
    """ 
    if 'key' not in request.form or request.form['key'] != DEV_KEY:
        return invalid_request()

    error_msg = None
    error_code = 0
    successful = True
    try:
        conn = sqlite3.connect(IMAGE_DATA_DB)
        cursor = conn.cursor()
        statement = "UPDATE image_data SET num_annotations = num_annotations + 1 WHERE id = :id"
        cursor.execute(statement, {'id': id})
        conn.commit()
        conn.close()
    except Exception as e:
        error_msg = str(e)
        error_code = 2
        successful = False 

    return jsonify({
        'successful': successful,
        'error_msg': error_msg,
        'error_code': error_code
    }), 200


@app.route("/update/disapprove/<id>", methods=['PUT'])
def handle_annotation_disapproved(id):
    """
    Decrement Annotation approval
    """ 
    if 'key' not in request.form or request.form['key'] != DEV_KEY:
        return invalid_request()

    error_msg = None
    error_code = 0
    successful = True
    try:
        conn = sqlite3.connect(IMAGE_DATA_DB)
        cursor = conn.cursor()
        statement = "UPDATE image_data SET num_annotations = MAX(num_annotations - 1, 0) WHERE id = :id"
        cursor.execute(statement, {'id': id})
        conn.commit()
        conn.close()
    except Exception as e:
        error_msg = str(e)
        error_code = 2
        successful = False 

    return jsonify({
        'successful': successful,
        'error_msg': error_msg,
        'error_code': error_code
    }), 200


#Leave blank - note on Rest api google doc
@app.route("/update/mix-annotation/<id>", methods=['PUT'])
def handle_mix_annotation(id):
    """
    Mix annotations given when given unique image identifer
    """
    if 'key' not in request.form or request.form['key'] != DEV_KEY:
        return invalid_request()
    
    error_msg = None
    error_code = 0
    successful = True

    return jsonify({
        'successful': successful,
        'error_msg': error_msg,
        'error_code': error_code
    }), 200


@app.route("/update/entry/<id>", methods=['PUT'])
def handle_entry_update(id):
    """
    Edit all data of an entry besides ID and Image content
    """
    if 'key' not in request.form or request.form['key'] != DEV_KEY:
        return invalid_request()

    error_msg = None
    error_code = 0
    successful = True
    try:
        conn = sqlite3.connect(IMAGE_DATA_DB)
        cursor = conn.cursor()
        for col in ['metadata', 'annotation', 'dataset', 'num_annotations']:
            if col in request.form:
                statement = f"UPDATE image_data SET {col} = :{col} WHERE id = :id"
                cursor.execute(statement, {'id': id, col: request.form[col]})
        conn.commit()
        conn.close()
    except Exception as e:
        error_msg = str(e)
        error_code = 2
        successful = False 

    return jsonify({
        'successful': successful,
        'error_msg': error_msg,
        'error_code': error_code
    }), 200


@app.route("/delete/entry/<id>", methods=['DELETE'])
def delete_image(id):
    """
    Remove image and all correlated info on it
    For troubleshooting refer to 
    https://stackoverflow.com/questions/26647248/how-to-delete-files-from-the-server-with-flask
    """
    if 'key' not in request.form or request.form['key'] != DEV_KEY:
        return invalid_request()

    error_msg = None
    error_code = 0
    successful = True
    try:
        conn = sqlite3.connect(IMAGE_DATA_DB)
        cursor = conn.cursor()
        statement = "DELETE FROM image_data WHERE id = ?"
        cursor.execute(statement, [id])
        conn.commit()
        conn.close()

        filename = [filename for filename in os.listdir(IMAGE_DIR) if filename.startswith(id)][0]
        os.remove(os.path.join(IMAGE_DIR, filename))
    except Exception as e:
        error_msg = str(e)
        error_code = 2
        successful = False 

    return jsonify({
        'successful': successful,
        'error_msg': error_msg,
        'error_code': error_code
    }), 200



if __name__ == "__main__":
    app.run(debug=False, threaded=False, host=HOST, port=PORT)
