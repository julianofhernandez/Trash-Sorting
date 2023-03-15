"""
The Rest API for model inference stuff.
"""

from flask import Flask
from flask_cors import CORS
import argparse

def display(data):
    print('[Server]: ' + data)


display('Attempting to initialize the server...')


HOST = 'localhost'
PORT = 5001  # https://stackoverflow.com/a/72797062

def create_app():
    from model_inference import model_inference, setup, MODELS_DIR
    app = Flask(__name__)

    setup()
    app.register_blueprint(model_inference, url_prefix='')
    return app


try:
    app = create_app()
    CORS(app)   #Create CORS header setup to allow request from the domain
except Exception as e:
    display('Failed to launch server, terminating process...')
    print(e)
    exit()
    
display('Successfully launched server')


if __name__ == '__main__':
    #from werkzeug.middleware.profiler import ProfilerMiddleware
    # app.wsgi_app = ProfilerMiddleware(app.wsgi_app, restrictions=[
    #                                  5], profile_dir='./')
    parser = argparse.ArgumentParser(
        prog='Model Inference Server',
        description='Server that handles predictions on images'
    )
    app.run(debug=False, threaded=False, host=HOST, port=PORT)
