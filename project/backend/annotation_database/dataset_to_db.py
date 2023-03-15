# TODO: Test with Server and further optimization (next sprint?)
# TODO: Adjust to other datasets imagenet, taco, etc.
# Worked on in Sprint 5, research done on how to increase dataset size. Postponed for future sprints.

import requests
import os
from multiprocessing import Pool

HOST = 'localhost'
PORT = 5000

URL = f'{HOST}:{PORT}/create/entry'

KEY = 'SECRETKEY'

def send_to_db_server(image_filepath, annotations, metadata):
    global KEY
    response = requests.post(
        URL, data={'key': KEY, 'annotations': annotations,
                   'num_annotation': 1, 'metadata': metadata},
        files={'image': open(image_filepath, 'rb')}
    )
    print(response)


if __name__ == '__main__':
    # TO LOOK AT: https://stackoverflow.com/questions/29104107/upload-image-using-post-form-data-in-python-requests

    image_dataset_dir = './images'
    annotations_dataset_dir = './annotations'
    origin_dataset = 'TACO'

    def send(file):
        global annotations_dataset_dir, image_dataset_dir, origin_dataset
        annotations = open(os.path.join(annotations_dataset_dir, file), 'r').read()
        send_to_db_server(os.path.join(image_dataset_dir, file),
                        annotations,
                        metadata={'origin': origin_dataset})

    with Pool(8) as p:
        p.map(os.listdir(image_dataset_dir))

    print('Complete')
