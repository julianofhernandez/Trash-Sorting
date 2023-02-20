"""
The Blueprint for the Rest API for connecting to the annotation database to upload, read,
edit, and delete images and their annotations.

Note: for format of the files when uploading to the database, include extension
if we allow for images of different extensions
"""

import sqlite3
import os
from flask import jsonify, request, send_from_directory, Blueprint


"""
We need to register the methods under a blueprint instead of just app.
That way we can use the factory pattern to create an app and use the app
resource for testing

https://stackoverflow.com/questions/39714995/404-response-when-running-flaskclient-test-method
https://flask.palletsprojects.com/en/2.2.x/blueprints/
"""
db_server = Blueprint('db_server', __name__)

DEV_KEY = None
IMAGE_DIR = 'images/'
IMAGE_DATA_DB = 'imageDB.db'
NUM_ENTRIES = None


def create_db():
    """
    Sets up the Sqlite database and creates the tables
    """
    global IMAGE_DATA_DB
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


def get_max_entries():
    global IMAGE_DATA_DB
    conn = sqlite3.connect(IMAGE_DATA_DB)
    cursor = conn.cursor()
    cursor.execute('SELECT MAX(id) FROM image_data')
    num_entries = cursor.fetchone()[0]
    conn.close()
    return 0 if num_entries is None else num_entries


def create_files():
    global IMAGE_DIR
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)


def setup():
    global NUM_ENTRIES, DEV_KEY
    create_db()
    create_files()
    NUM_ENTRIES = get_max_entries() + 1
    if os.path.exists('KEY.ps') and os.path.isfile('KEY.ps'):
        DEV_KEY = open('KEY.ps', 'r').read()
    else:
        DEV_KEY = 'secretkey'


def invalid_request(error_msg='Invalid Key', error_code=1, code=401):
    """
    Returns Format for invalid response. By default returns a response
    for an invalid developer key
    """
    return jsonify({
        'successful': False,
        'error_msg': error_msg,
        'error_code': error_code
    }), code


@db_server.route('/create/entry', methods=['POST'])
def handle_entry():
    """
    Creates a new entry in the database. Uploads an image
    to the directory and other data to database
    """
    global NUM_ENTRIES, IMAGE_DIR, IMAGE_DATA_DB
    if 'key' not in request.form or request.form['key'] != DEV_KEY:
        return invalid_request()
    elif 'image' not in request.files:
        return invalid_request(error_msg='Missing image in file part of request',
                               error_code=3, code=200)
    elif 'annotation' not in request.form:
        return invalid_request(error_msg='Missing annotation in form request',
                               error_code=4, code=200)
    elif 'num_annotations' not in request.form:
        return invalid_request(error_msg='Missing num_annotations in form request',
                               error_code=5, code=200)
    elif 'dataset' not in request.form:
        return invalid_request(error_msg='Missing dataset in form request',
                               error_code=6, code=200)
    elif 'metadata' not in request.form:
        return invalid_request(error_msg='Missing metadata in form request',
                               error_code=7, code=200)
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

            entry = [NUM_ENTRIES, annotation,
                     num_annotations, dataset, metadata]
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


@db_server.route('/read/entry/image/<id>')
def handle_get_entry_image(id):
    global IMAGE_DIR
    error_msg = None
    error_code = 0
    try:
        file = [filename for filename in os.listdir(
            IMAGE_DIR) if filename.startswith(id)][0]
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


@db_server.route('/read/entry/data/<id>')
def handle_get_entry_metadata(id):
    """
    Query for a single entry's data
    """
    global IMAGE_DATA_DB
    error_msg = None
    error_code = 0
    data = None
    try:
        conn = sqlite3.connect(IMAGE_DATA_DB)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM image_data WHERE id = :id', {'id': id})
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


@db_server.route('/read/search/<filter>', methods=['GET'])
def handle_search_entries(filter):
    """
    Query for all entries given a single search
    """
    global IMAGE_DATA_DB
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


@db_server.route('/read/annotation/min', methods=['GET'])
def handle_get_entry_min_annotation():
    """
    Query for entry with no or least annotations
    """
    global IMAGE_DATA_DB
    data = None
    error_msg = None
    error_code = 0
    try:
        conn = sqlite3.connect(IMAGE_DATA_DB)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM image_data ORDER BY num_annotations ASC LIMIT 1')
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


@db_server.route('/read/annotation/max', methods=['GET'])
def handle_get_entry_max_annotations():
    """
    Query for entry with most annotations
    """
    global IMAGE_DATA_DB
    data = None
    error_msg = None
    error_code = 0
    try:
        conn = sqlite3.connect(IMAGE_DATA_DB)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM image_data ORDER BY num_annotations DESC LIMIT 1')
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


@db_server.route('/update/approve/<id>', methods=['PUT'])
def handle_annotation_approved(id):
    """
    Increment Annotation approval
    """
    global IMAGE_DATA_DB, DEV_KEY
    if 'key' not in request.form or request.form['key'] != DEV_KEY:
        return invalid_request()

    error_msg = None
    error_code = 0
    successful = True
    try:
        conn = sqlite3.connect(IMAGE_DATA_DB)
        cursor = conn.cursor()
        statement = 'UPDATE image_data SET num_annotations = num_annotations + 1 WHERE id = :id'
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


@db_server.route('/update/disaprove/<id>', methods=['PUT'])
def handle_annotation_disapproved(id):
    """
    Decrement Annotation approval
    """
    global IMAGE_DATA_DB, DEV_KEY
    if 'key' not in request.form or request.form['key'] != DEV_KEY:
        return invalid_request()

    error_msg = None
    error_code = 0
    successful = True
    try:
        conn = sqlite3.connect(IMAGE_DATA_DB)
        cursor = conn.cursor()
        statement = 'UPDATE image_data SET num_annotations = MAX(num_annotations - 1, 0) WHERE id = :id'
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


# Leave blank - note on Rest api google doc
@db_server.route('/update/mix-annotation/<id>', methods=['PUT'])
def handle_mix_annotation(id):
    """
    Mix annotations given when given unique image identifer
    """
    global DEV_KEY
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


@db_server.route('/update/entry/<id>', methods=['PUT'])
def handle_entry_update(id):
    """
    Edit all data of an entry besides ID and Image content
    """
    global IMAGE_DATA_DB, DEV_KEY
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
                statement = f'UPDATE image_data SET {col} = :{col} WHERE id = :id'
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


@db_server.route('/delete/entry/<id>', methods=['DELETE'])
def delete_image(id):
    """
    Remove image and all correlated info on it
    For troubleshooting refer to 
    https://stackoverflow.com/questions/26647248/how-to-delete-files-from-the-server-with-flask
    """
    global IMAGE_DATA_DB, DEV_KEY, IMAGE_DIR
    if 'key' not in request.form or request.form['key'] != DEV_KEY:
        return invalid_request()

    error_msg = None
    error_code = 0
    successful = True
    try:
        conn = sqlite3.connect(IMAGE_DATA_DB)
        cursor = conn.cursor()
        statement = 'DELETE FROM image_data WHERE id = ?'
        cursor.execute(statement, [id])
        conn.commit()
        conn.close()

        filename = [filename for filename in os.listdir(
            IMAGE_DIR) if filename.startswith(id)][0]
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
