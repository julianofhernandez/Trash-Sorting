"""
Tests for the flask image database function api calls.
Tests are divided into classes for reading, getting metadata
getting images, updating the db and deleting from the db
"""

import pytest
from db_server_app import create_annotation_server


key = "secretkey"
invalid_key = ""


# Creates fixtures which simulate a live server
@pytest.fixture()
def app():
    app = create_annotation_server()
    app.config.update(
        {
            "TESTING": True,
        }
    )

    # additional future setup can go here

    yield app

    # clean up / reset any additional resources here


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def runner(app):
    return app.test_cli_runner()


class TestReadEntry:
    def test_invalid_key(self, client):
        data = {
            "key": invalid_key,
            "annotation": "",
            "num_annotations": 3,
            "dataset": "custom",
            "metadata": "",
        }
        res = client.post("http://localhost:5000/create/entry", data=data)
        assert res.status_code == 401
        assert res.get_json() == {
            "error_code": 1,
            "error_msg": "Invalid Key",
            "successful": False,
        }

    def test_image_missing(self, client):
        data = {
            "key": key,
            "annotation": "",
            "num_annotations": 3,
            "dataset": "custom",
            "metadata": "",
        }
        res = client.post("http://localhost:5000/create/entry", data=data)
        assert res.status_code == 200
        assert res.get_json() == {
            "error_code": 3,
            "error_msg": "Missing image in file part of request",
            "successful": False,
        }

    def test_missing_annotation(self, client):
        data = {
            "key": key,
            "dataset": "custom",
            "metadata": "",
            "image": open("cat.jpg", "rb"),
        }
        res = client.post("http://localhost:5000/create/entry", data=data)
        assert res.status_code == 200
        assert res.get_json() == {
            "error_code": 4,
            "error_msg": "Missing annotation in form request",
            "successful": False,
        }

    def test_missing_num_annotation(self, client):
        data = {
            "key": key,
            "annotation": "",
            "dataset": "custom",
            "metadata": "",
            "image": open("cat.jpg", "rb"),
        }
        res = client.post("http://localhost:5000/create/entry", data=data)
        assert res.status_code == 200
        assert res.get_json() == {
            "error_code": 5,
            "error_msg": "Missing num_annotations in form request",
            "successful": False,
        }

    def test_missing_dataset(self, client):
        data = {
            "key": key,
            "annotation": "",
            "num_annotations": 3,
            "metadata": "",
            "image": open("cat.jpg", "rb"),
        }
        res = client.post("http://localhost:5000/create/entry", data=data)
        assert res.status_code == 200
        assert res.get_json() == {
            "error_code": 6,
            "error_msg": "Missing dataset in form request",
            "successful": False,
        }

    def test_missing_metadata(self, client):
        data = {
            "key": key,
            "annotation": "",
            "num_annotations": 3,
            "dataset": "custom",
            "image": open("cat.jpg", "rb"),
        }
        res = client.post("http://localhost:5000/create/entry", data=data)
        assert res.status_code == 200
        assert res.get_json() == {
            "error_code": 7,
            "error_msg": "Missing metadata in form request",
            "successful": False,
        }

    def test_success(self, client):
        data = {
            "key": key,
            "annotation": "",
            "num_annotations": 3,
            "dataset": "custom",
            "metadata": "",
            "image": open("cat.jpg", "rb"),
        }
        res = client.post("http://localhost:5000/create/entry", data=data)
        assert res.status_code == 200
        assert res.get_json() == {"data": "1", "error_code": 0, "error_msg": None}


class TestGetImage:
    def test_image_success(self, client):
        res = client.get("http://localhost:5000/read/entry/image/1")
        assert res.status_code == 200
        assert res.get_json() == None

    def test_image_out_of_bounds(self, client):
        res = client.get("http://localhost:5000/read/entry/image/99999")
        assert res.status_code == 200
        assert res.get_json() == {
            "error_code": 3,
            "error_msg": "list index out of range",
        }


class TestGetMetadata:
    def test_empty_result(self, client):
        valid_res = {
            "data": {
                "annotation": "",
                "dataset": "custom",
                "id": 1,
                "metadata": "",
                "num_annotations": 3,
            },
            "error_code": 0,
            "error_msg": None,
        }

        res = client.get("http://localhost:5000/read/entry/data/1")
        assert res.status_code == 200
        assert res.get_json() == valid_res

    def test_success(self, client):
        res = client.get("http://localhost:5000/read/entry/data/999999")
        assert res.status_code == 200
        assert res.get_json() == {
            "data": None,
            "error_code": 3,
            "error_msg": "No results from query",
        }


class test_handle_search_entries:
    def test_empty_results(self, client):
        valid_res = {
            "data": [
                {
                    "annotation": "",
                    "dataset": "custom",
                    "id": 1,
                    "metadata": "",
                    "num_annotations": 3,
                }
            ],
            "error_code": 0,
            "error_msg": None,
        }
        res = client.get("http://localhost:5000/read/search/id=1")
        assert res.status_code == 200
        assert res.get_json() == valid_res

    def test_success(self, client):
        valid_res = {"data": [], "error_code": 0, "error_msg": None}
        res = client.get("http://localhost:5000/read/search/id=999999")
        assert res.status_code == 200
        assert res.get_json() == valid_res


class TestReadEntries:
    def test_get_min_max_annotation(self, client):
        # test for min annotations
        res = client.get("http://localhost:5000/read/annotation/min")
        assert res.status_code == 200
        assert res.get_json() == {
            "data": {
                "annotation": "",
                "dataset": "custom",
                "id": 1,
                "metadata": "",
                "num_annotations": 3,
            },
            "error_code": 0,
            "error_msg": None,
        }


class TestHandleGetEntryMaxnnotations:
    def test_max_annotation_success(self, client):
        res = client.get("http://localhost:5000/read/annotation/max")
        assert res.status_code == 200
        assert res.get_json() == {
            "data": {
                "annotation": "",
                "dataset": "custom",
                "id": 1,
                "metadata": "",
                "num_annotations": 3,
            },
            "error_code": 0,
            "error_msg": None,
        }


class TestHandleAnnotationApproved:
    def test_approved_invalid_key(self, client):
        res = client.put(
            "http://localhost:5000/update/approve/1", data={"key": invalid_key}
        )
        assert res.status_code == 401
        assert res.get_json() == {
            "error_code": 1,
            "error_msg": "Invalid Key",
            "successful": False,
        }

    def test_approved_success(self, client):
        res = client.put("http://localhost:5000/update/approve/1", data={"key": key})
        assert res.status_code == 200
        assert res.get_json() == {
            "error_code": 0,
            "error_msg": None,
            "successful": True,
        }


class test_handle_annotation_disapproved:
    def test_disapproved_invalid_key(client):
        res = client.put(
            "http://localhost:5000/update/disaprove/1", data={"key": invalid_key}
        )
        assert res.status_code == 401
        assert res.get_json() == {
            "error_code": 1,
            "error_msg": "Invalid Key",
            "successful": False,
        }

    def test_disapproved_success(client):
        res = client.put("http://localhost:5000/update/disaprove/1", data={"key": key})
        assert res.status_code == 200
        assert res.get_json() == {
            "error_code": 0,
            "error_msg": None,
            "successful": True,
        }


class TestHandleEntryUpdate:
    def test_update_invalid_key(self, client):
        res = client.put(
            "http://localhost:5000/update/entry/1", data={"key": invalid_key}
        )
        assert res.status_code == 401
        assert res.get_json() == {
            "error_code": 1,
            "error_msg": "Invalid Key",
            "successful": False,
        }

    def test_update_entry_success(self, client):
        res = client.put("http://localhost:5000/update/entry/1", data={"key": key})
        assert res.status_code == 200
        assert res.get_json() == {
            "error_code": 0,
            "error_msg": None,
            "successful": True,
        }


class TestDeleteImage:
    def test_delete_invalid_key(self, client):
        res = client.delete(
            "http://localhost:5000/delete/entry/1", data={"key": invalid_key}
        )
        assert res.status_code == 401
        assert res.get_json() == {
            "error_code": 1,
            "error_msg": "Invalid Key",
            "successful": False,
        }

    def test_delete_success(self, client):
        res = client.delete("http://localhost:5000/delete/entry/1", data={"key": key})
        assert res.status_code == 200
        assert res.get_json() == {
            "error_code": 0,
            "error_msg": None,
            "successful": True,
        }

    def test_delete_out_of_bounds(self, client):
        res = client.delete(
            "http://localhost:5000/delete/entry/99999", data={"key": key}
        )
        assert res.status_code == 200
        assert res.get_json() == {
            "error_code": 2,
            "error_msg": "list index out of range",
            "successful": False,
        }
