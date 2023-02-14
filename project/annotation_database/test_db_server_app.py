from flask import Flask
import pytest
import requests
import numpy as np
import json
from io import BytesIO
import sqlite3
from db_server_app import create_app

"""
We use fixtures to create the app and client for testing.
The functions under @pytest.fixture() do not need to be called
and the resources are already loaded into memory by pytest.
The resources are loaded under the same name as the function.

To use the fixtures in a function, put the name of the resource
in the parameters of the function of the test

As long as the tests are ran in the correct order and the db is newly created
all test should pass. For future see how to set up temporary directories for test
and confirm the correct files were added/deleted. plus set up the db in a temporary
state for each test to confirm it works

https://flask.palletsprojects.com/en/2.2.x/testing/
"""

key = 'secretkey'
invalid_key = ''

@pytest.fixture()
def app():
    app = create_app()
    app.config.update({
        "TESTING": True,
    })

    # other setup can go here

    yield app

    # clean up / reset resources here


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def runner(app):
    return app.test_cli_runner()

def test_handle_entry(client):

	# test invalid key
        data = {'key': invalid_key, 'annotation': '', 'num_annotations': 3, 'dataset': 'custom', 'metadata': ''}
        res = client.post("http://localhost:5000/create/entry", data = data)
        assert res.status_code == 401
        assert res.get_json() == {'error_code': 1, 'error_msg': 'Invalid Key', 'successful': False}

        # test image not in files
        data = {'key': key, 'annotation': '', 'num_annotations': 3, 'dataset': 'custom', 'metadata': ''}
        res = client.post("http://localhost:5000/create/entry", data = data)
        assert res.status_code == 200
        assert res.get_json() == {'error_code': 3, 'error_msg': 'Missing image in file part of request', 'successful': False}

        # test annotation not in form
        data = {'key': key, 'dataset': 'custom', 'metadata': '', 'image': open('cat.jpg','rb')}
        res = client.post("http://localhost:5000/create/entry", data = data)
        assert res.status_code == 200
        assert res.get_json() == {'error_code': 4, 'error_msg': 'Missing annotation in form request', 'successful': False}


        # test num_annotation not in form
        data = {'key': key, 'annotation': '', 'dataset': 'custom', 'metadata': '', 'image': open('cat.jpg','rb')}
        res = client.post("http://localhost:5000/create/entry", data = data)
        assert res.status_code == 200
        assert res.get_json() == {'error_code': 5, 'error_msg': 'Missing num_annotations in form request', 'successful': False}


	# test dataset not in form
        data = {'key': key, 'annotation': '', 'num_annotations': 3, 'metadata': '', 'image': open('cat.jpg','rb')}
        res = client.post("http://localhost:5000/create/entry", data = data)
        assert res.status_code == 200
        assert res.get_json() == {'error_code': 6, 'error_msg': 'Missing dataset in form request', 'successful': False}

	# test metadata not in form
        data = {'key': key, 'annotation': '', 'num_annotations': 3, 'dataset': 'custom', 'image': open('cat.jpg','rb')}
        res = client.post("http://localhost:5000/create/entry", data = data)
        assert res.status_code == 200
        assert res.get_json() == {'error_code': 7, 'error_msg': 'Missing metadata in form request', 'successful': False}

	# test entry success
        data = {'key': key, 'annotation': '', 'num_annotations': 3, 'dataset': 'custom', 'metadata': '', 'image': open('cat.jpg','rb')}
        res = client.post("http://localhost:5000/create/entry", data = data)
        assert res.status_code == 200
        #assert res.get_json() == {'data': '1', 'error_code': 0, 'error_msg': None, 'successful': True}


def test_handle_get_entry_image(client):

	# test get entry success
        res = client.get('http://localhost:5000/read/entry/image/1')
        assert res.status_code == 200
        assert res.get_json() == None

        #Question for future: possibly want to check if file was downloaded?

        # test out of bounds index
        res = client.get('http://localhost:5000/read/entry/image/99999')
        assert res.status_code == 200
        assert res.get_json() == {'error_code': 3, 'error_msg': 'list index out of range'}

def test_handle_get_entry_metadata(client):
        valid_res = {"data":{"annotation":"","dataset":"custom","id":1,"metadata":"","num_annotations":3},"error_code":0,"error_msg":None}

	# test empty results from query
        res = client.get('http://localhost:5000/read/entry/data/1')
        assert res.status_code == 200
        assert res.get_json() == valid_res

	# test get entry metadata success
        res = client.get('http://localhost:5000/read/entry/data/999999')
        assert res.status_code == 200
        assert res.get_json() == {'data': None, 'error_code': 3, 'error_msg': 'No results from query'}

def test_handle_search_entries(client):
        valid_res1 = {'data': [{'annotation': '', 'dataset': 'custom', 'id': 1, 'metadata': '', 'num_annotations': 3}], 'error_code': 0, 'error_msg': None}
        valid_res2 = {'data': [], 'error_code': 0, 'error_msg': None}

	# test empty results
        res = client.get('http://localhost:5000/read/search/id=1')
        assert res.status_code == 200
        assert res.get_json() == valid_res1

	# test success
        res = client.get('http://localhost:5000/read/search/id=999999')
        assert res.status_code == 200
        assert res.get_json() == valid_res2

def test_get_min_max_annotation(client):
        #test for min annotations
        res = client.get('http://localhost:5000/read/annotation/min')
        assert res.status_code == 200
        assert res.get_json() == {'data': {'annotation': '', 'dataset': 'custom', 'id': 1, 'metadata': '', 'num_annotations': 3}, 'error_code': 0, 'error_msg': None}

        # test empty when db empty?
        #assert res.get_json() == {'data': None, 'error_code': 3, 'error_msg': 'No results from query'}

def test_handle_get_entry_max_annotations(client):
	#test for max annotations
        res = client.get('http://localhost:5000/read/annotation/max')
        assert res.status_code == 200
        assert res.get_json() == {'data': {'annotation': '', 'dataset': 'custom', 'id': 1, 'metadata': '', 'num_annotations': 3}, 'error_code': 0, 'error_msg': None}

        # test empty when db empty?
        #assert res.get_json() == {'data': None, 'error_code': 3, 'error_msg': 'No results from query'}

def test_handle_annotation_approved(client):
	# test invalid key
        res = client.put('http://localhost:5000/update/approve/1', data={'key': invalid_key})
        assert res.status_code == 401
        assert res.get_json() == {'error_code': 1, 'error_msg': 'Invalid Key', 'successful': False}

	# test test success
        res = client.put('http://localhost:5000/update/approve/1', data={'key': key})
        assert res.status_code == 200
        assert res.get_json() == {'error_code': 0, 'error_msg': None, 'successful': True}

def test_handle_annotation_disapproved(client):
        # test invalid key
        res = client.put('http://localhost:5000/update/disaprove/1', data={'key': invalid_key})
        assert res.status_code == 401
        assert res.get_json() == {'error_code': 1, 'error_msg': 'Invalid Key', 'successful': False}
	# test test success
        res = client.put('http://localhost:5000/update/disaprove/1', data={'key': key})
        assert res.status_code == 200
        assert res.get_json() == {'error_code': 0, 'error_msg': None, 'successful': True}

def test_handle_entry_update(client):
	# test invalid key
        res = client.put('http://localhost:5000/update/entry/1', data={'key': invalid_key})
        assert res.status_code == 401
        assert res.get_json() == {'error_code': 1, 'error_msg': 'Invalid Key', 'successful': False}

	# test success
        res = client.put('http://localhost:5000/update/entry/1', data={'key': key})
        assert res.status_code == 200
        assert res.get_json() == {'error_code': 0, 'error_msg': None, 'successful': True}

def test_delete_image(client):
        # tests invalid key
        res = client.delete('http://localhost:5000/delete/entry/1', data={'key': invalid_key})
        assert res.status_code == 401
        assert res.get_json() == {'error_code': 1, 'error_msg': 'Invalid Key', 'successful': False}
        
	# test success
        res = client.delete('http://localhost:5000/delete/entry/1', data={'key': key})
        assert res.status_code == 200
        assert res.get_json() == {'error_code': 0, 'error_msg': None, 'successful': True}

        # test out of bounds error
        res = client.delete('http://localhost:5000/delete/entry/99999', data={'key': key})
        assert res.status_code == 200
        assert res.get_json() == {'error_code': 2, 'error_msg': 'list index out of range', 'successful': False}        

