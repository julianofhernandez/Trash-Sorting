"""
The Rest API for connecting to the annotation database to upload, read,
edit, and delete images and their annotations.
"""

def display(data):
    print("[Server]: " + data)

display("Attempting to initialize the server...")

import sqlite3
import os
from flask import Flask, jsonify, request, send_file, send_from_directory, Blueprint

    
HOST = 'localhost'

PORT = 5000

IMAGE_DIR = "images/"


# https://flask.palletsprojects.com/en/2.2.x/patterns/appfactories/
def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = IMAGE_DIR

    # limit max size for image size
    app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024

    #app.config.from_pyfile(config_filename)

    #from yourapplication.model import db
    #db.init_app(app)

    from db_server import db_server, create_db

    create_db()
    
    app.register_blueprint(db_server, url_prefix="")
    
    return app

try:    
    app = create_app()
except Exception as e:
    display("Failed to launch server, terminating process...")
    print(e)
    exit()

display("Successfully launched server")



if __name__ == "__main__":
    app.run(debug=False, threaded=False, host=HOST, port=PORT)
