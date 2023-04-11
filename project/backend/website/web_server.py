from flask import Blueprint, send_from_directory


web_server = Blueprint('web_server', __name__, static_folder='build')


@web_server.route('/')
def serve():
    return send_from_directory(web_server.static_folder, 'index.html')


@web_server.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(web_server.static_folder + '/static', path)
