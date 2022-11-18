import os
import json
import numpy as np
import matplotlib
import argparse
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import requests
from pycocotools.coco import COCO
from WasteWizard import WasteWizard
from git import Repo

#! Add requirements.txt to our main requirements
local_repo = "TACO_NEW"
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_path', required=False, default='TACO', help='Path to download photos locally')
parser.add_argument('--output', required=False, default='taco_parsed.json', help='Output parsed JSON')
args = parser.parse_args()

def getCocoAnnotations(annotation_file):
    if not os.path.exists(args.dataset_path):
        print('Cloning from https://github.com/pedropro/TACO.git')
        Repo.clone_from('https://github.com/pedropro/TACO.git', args.dataset_path)
    downloadedData = os.path.join(args.dataset_path, 'data')
    cmdStr = 'cmd /c "cd '+args.dataset_path+' & python download.py"'
    os.system(cmdStr)
    annotation_file=os.path.join(args.dataset_path, "data/annotations.json")
    if not os.path.exists(annotation_file):
        return "no TACO annotations file at {annotation_file}"
    coco_annotation = COCO(annotation_file=annotation_file)
    ww = WasteWizard()

    # Category IDs.
    cat_ids = coco_annotation.getCatIds()
    # All categories.
    cats = coco_annotation.loadCats(cat_ids)
    cat_names = [cat["name"] for cat in cats]
    print("Categories translation")

    # Loop through all images
    img_ids = coco_annotation.getImgIds()
    print(f"There are {len(img_ids)} total images")
    newAnnotationsList = [] 
    for img_id in img_ids:
        img_info = coco_annotation.loadImgs([img_id])[0]
        img_file_name = img_info["file_name"]
        img_url = img_info["coco_url"]

        # Get all the annotations for the specified image.
        ann_ids = coco_annotation.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco_annotation.loadAnns(ann_ids)
        # Loop through each object in the images
        for objectAnnotation in anns:
            newCategory = cat_names[objectAnnotation['category_id']]
            newAnnotation = {
                'file_path':os.path.join(downloadedData, img_file_name),
                'Original category': newCategory,
                'New category':ww.searchTerm(newCategory),
                'x': objectAnnotation['bbox'][0],
                'y':objectAnnotation['bbox'][1],
                'width':objectAnnotation['bbox'][2],
                'height':objectAnnotation['bbox'][3],
            }
            # print(str(newAnnotation))
            newAnnotationsList.append(newAnnotation)
    print(str(len(newAnnotationsList))+' annotatins parsed')
    with open(args.output, 'w') as f:
        json.dump(newAnnotationsList, f)
        print(f"File saved to {args.output}")
    return newAnnotationsList


if __name__ == "__main__":
    taco = getCocoAnnotations('TACO')
    print(str(taco))