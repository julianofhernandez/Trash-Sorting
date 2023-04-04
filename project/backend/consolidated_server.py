from flask import Flask
from annotation_database import db_server as server
from model_inference import model_inference as inference
from website import web_server as web


HOST = 'localhost'
PORT = 8000


def create_app():
    app = Flask(__name__, static_folder='./website/build')
    app.config['UPLOAD_FOLDER'] = server.IMAGE_DIR
    app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024

    server.setup()
    inference.setup()

    app.register_blueprint(server.db_server, url_prefix='')
    app.register_blueprint(inference.model_inference, url_prefix='')
    app.register_blueprint(web.web_server, url_prefix='')

    return app


app = create_app()

if __name__ == '__main__':
    app.run(debug=False, threaded=False, host=HOST, port=PORT)
