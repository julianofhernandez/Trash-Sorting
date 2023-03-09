import json
import requests
from io import BytesIO
import numpy as np
import cv2


class TestCreate:
    def test_invalid_key(self):
        res = requests.post('http://localhost:5001/create/model/test',
                            data={'key': 'wrongkey',
                                  'metadata': json.dumps({'a': 5})},
                            files={'model': open('README.md', 'rb')})
        print(json.loads(res.content))
        print(json.dumps(res.content.decode()))
        assert res.status_code == 200
        assert json.loads(res.content) == {'error_code': 1,
                                           'error_msg': 'Invalid Key',
                                           'successful': False}

    def test_already_exists(self):
        res = requests.post('http://localhost:5001/create/model/test',
                            data={'key': 'secretkey',
                                  'metadata': json.dumps({'a': 5})},
                            files={'model': open('README.md', 'rb')})
        assert res.status_code == 200
        assert json.loads(res.content) == {'error_code': 5,
                                           'error_msg': 'Model already exists',
                                           'successful': False}

    def test_missing_model(self):
        res = requests.post('http://localhost:5001/create/model/test2',
                            data={'key': 'secretkey',
                                  'metadata': json.dumps({'a': 5})},
                            files={})
        assert res.status_code == 200
        assert json.loads(res.content) == {'error_code': 3,
                                           'error_msg': 'Missing model in files part of request',
                                           'successful': False}

    def test_missing_metadata(self):
        res = requests.post('http://localhost:5001/create/model/test2',
                            data={'key': 'secretkey'},
                            files={'model': open('README.md', 'rb')})
        assert res.status_code == 200
        assert json.loads(res.content) == {'error_code': 4,
                                           'error_msg': 'Missing metadata in form part of request',
                                           'successful': False}

    def test_new_success(self):
        res = requests.post('http://localhost:5001/create/model/test2',
                            data={'key': 'secretkey',
                                  'metadata': json.dumps({'b': 6})},
                            files={'model': open('README.md', 'rb')})
        assert res.status_code == 200
        assert json.loads(res.content) == {'error_code': 0,
                                           'error_msg': None,
                                           'successful': True}


class TestReadInference:
    def test_no_image(self):
        res = requests.post(
            'http://localhost:5001/read/inference/test', files={})
        assert res.status_code == 200
        assert json.loads(res.content) == {'error_code': 3,
                                           'error_msg': 'Missing image in files part of request',
                                           'model_name': 'test',
                                           'predictions': None}

    def test_success(self):
        byte_arr = BytesIO()
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        byte_arr.write(cv2.imencode('.jpg', image)[1])
        byte_arr.seek(0)
        res = requests.post(
            'http://localhost:5001/read/inference/test', files={'image': byte_arr})
        res_content = json.loads(res.content)

        assert res.status_code == 200
        assert 'error_code' in res_content and res_content['error_code'] == 0
        assert 'error_msg' in res_content and res_content['error_msg'] is None
        assert 'model_name' in res_content and res_content['model_name'] == 'test'
        assert 'predictions' in res_content
        assert set(['bounding_box', 'class_probs', 'obj_label', 'trash_bin_label']
                   ).issubset(set(res_content['predictions']))


class TestReadBatchInf:
    def test_missing_images(self):
        res = requests.post(
            'http://localhost:5001/read/batch-inference/test', data={'num_image': 2}, files={})
        res, res.content
        assert res.status_code == 200
        assert json.loads(res.content) == {'batch_predictions': None,
                                           'error_code': 3,
                                           'error_msg': 'Missing image_0 in files part of request',
                                           'model_name': 'test'}

    def test_missing_num_image(self):
        byte_arr, byte_arr2 = '', ''
        res = requests.post('http://localhost:5001/read/batch-inference/test',
                            data={}, files={'image_0': byte_arr, 'image_1': byte_arr2})
        assert res.status_code == 200
        assert json.loads(res.content) == {'batch_predictions': None,
                                           'error_code': 4,
                                           'error_msg': 'Missing num_image in form part of request',
                                           'model_name': 'test'}

    def test_success(self):
        byte_arr = BytesIO()
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        byte_arr.write(cv2.imencode('.jpg', image)[1])
        byte_arr.seek(0)
        res = requests.post(
            'http://localhost:5001/read/inference/test', files={'image': byte_arr})
        res_content = json.loads(res.content)

        assert res.status_code == 200
        assert 'error_code' in res_content and res_content['error_code'] == 0
        assert 'error_msg' in res_content and res_content['error_msg'] is None
        assert 'model_name' in res_content and res_content['model_name'] == 'test'
        assert 'predictions' in res_content
        assert set(['bounding_box', 'class_probs', 'obj_label', 'trash_bin_label']
                   ).issubset(set(res_content['predictions']))


class TestReadModelList:
    def test_list(self):
        res = requests.get('http://localhost:5001/read/model/list')
        res, res.content, json.loads(res.content)['model_list']
        assert res.status_code == 200
        assert json.loads(res.content) == {'error_code': 0,
                                           'error_msg': None,
                                           'model_list':
                                           [{'metadata': json.dumps({'a': 5}), 'model_name': 'test'},
                                            {'metadata': json.dumps({'b': 6}), 'model_name': 'test2'}]
                                           }


class TestReadModel:
    def test_unknown(self):
        res = requests.get('http://localhost:5001/read/model/watermelon')
        assert res.status_code == 200
        assert json.loads(res.content) == {'error_code': 3,
                                           'error_msg': 'Model Not Found'}

    def test_known(self):
        res = requests.get('http://localhost:5001/read/model/test')
        assert res.status_code == 200
        assert str(res.content) == "b'test'"


class TestReadMetadata:
    def test_unknown(self):
        res = requests.get('http://localhost:5001/read/metadata/watermelon')
        assert res.status_code == 200
        assert json.loads(res.content) == {'error_code': 3,
                                           'error_msg': 'Model Metadata Not Found',
                                           'metadata': ''}

    def test_known(self):
        res = requests.get('http://localhost:5001/read/metadata/test')
        assert res.status_code == 200
        assert json.loads(res.content) == {'error_code': 0,
                                           'error_msg': None,
                                           'metadata': json.dumps({'a': 5})}


class TestUpdate:
    def test_invalid_key(self):
        res = requests.put('http://localhost:5001/update/model/test',
                           data={'key': 'badkey', 'metadata': json.dumps({'a': 7})})
        assert res.status_code == 200
        assert json.loads(res.content) == {'error_code': 1,
                                           'error_msg': 'Invalid Key',
                                           'successful': False}

    def test_model(self):
        res = requests.put('http://localhost:5001/update/model/test2',
                           data={'key': 'secretkey', 'model': open('README.md', 'rb')})
        assert res.status_code == 200
        assert json.loads(res.content) == {'error_code': 0,
                                           'error_msg': None,
                                           'successful': True}

    def test_metadata(self):
        res = requests.put('http://localhost:5001/update/model/test',
                           data={'key': 'secretkey', 'metadata': json.dumps({'a': 7})})
        assert res.status_code == 200
        assert json.loads(res.content) == {'error_code': 0,
                                           'error_msg': None,
                                           'successful': True}
        res = requests.put('http://localhost:5001/update/model/test',
                           data={'key': 'secretkey', 'metadata': json.dumps({'a': 5})})
        res, res.content


class TestDelete:
    def test_invalid_key(self):
        res = requests.delete(
            'http://localhost:5001/delete/model/test', data={'key': 'badkey'})
        assert res.status_code == 200
        assert json.loads(res.content) == {'error_code': 1,
                                           'error_msg': 'Invalid Key',
                                           'successful': False}

    def test_success(self):
        res = requests.delete(
            'http://localhost:5001/delete/model/test2', data={'key': 'secretkey'})
        assert res.status_code == 200
        assert json.loads(res.content) == {'error_code': 0,
                                           'error_msg': None,
                                           'successful': True}
