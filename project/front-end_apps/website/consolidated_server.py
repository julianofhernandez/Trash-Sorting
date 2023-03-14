from flask import Flask, render_template, Blueprint

#sys.path.append()
#sys.path.append()
import sys
sys.path.append('../annotation_database')
sys.path.append('../model_inference')
import db_server as server
import model_inference as inference

HOST = 'localhost'
PORT = 8000

html_app = Blueprint('html_app', __name__)

def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = server.IMAGE_DIR
    app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024

    server.setup()
    inference.setup()

    app.register_blueprint(html_app, url_prefix = '')
    app.register_blueprint(server.db_server, url_prefix = '')
    app.register_blueprint(inference.model_inference, url_prefix = '')

    return app

@html_app.route('/')
def index():
    return render_template('website/webpage.html')


app = create_app()

if __name__ == '__main__':
    app.run(debug=False, host=HOST, port=PORT)
