"""
The Rest API for connecting to the annotation database to upload, read,
edit, and delete images and their annotations.
"""

from flask import Flask
import argparse


def display(data):
    print('[Server]: ' + data)


display('Attempting to initialize the server...')


HOST = 'localhost'
PORT = 5000


# https://flask.palletsprojects.com/en/2.2.x/patterns/appfactories/
def create_app():
    from db_server import db_server, setup, IMAGE_DIR
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = IMAGE_DIR
    app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024
    # app.config.from_pyfile(config_filename)

    setup()
    app.register_blueprint(db_server, url_prefix='')
    return app


try:
    app = create_app()
except Exception as e:
    display('Failed to launch server, terminating process...')
    print(e)
    exit()

display('Successfully launched server')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Annotation Database Server',
        description='Server that can send and recieve images and annotations.'
    )
    app.run(debug=False, threaded=False, host=HOST, port=PORT)
