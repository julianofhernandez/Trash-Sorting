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
import time
from progress.bar import Bar

#! Add requirements.txt to our main requirements
local_repo = "TACO_NEW"
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_path', required=False, default='TACO', help='Path to download photos locally')
parser.add_argument('--category_translation', required=False, default='category_translation.json', help='List of new categories, this needs to be checked before running')
parser.add_argument('--output', required=False, default='taco_parsed.json', help='Output parsed JSON')
args = parser.parse_args()

def downloadRepository():
    if not os.path.exists(args.dataset_path):
        print('Cloning from https://github.com/pedropro/TACO.git')
        Repo.clone_from('https://github.com/pedropro/TACO.git', args.dataset_path)
    # Exec Python download
    cmdStr = 'cmd /c "cd '+args.dataset_path+' & python download.py"'
    os.system(cmdStr)
    # Check that it worked
    if not os.path.exists(os.path.join(args.dataset_path, "data/annotations.json")):
        return "no TACO annotations file at {annotation_file}"

def loadCategoryTranslation(cats):
    categoriesJson = []
    if os.path.exists(args.category_translation):
        with open(args.category_translation, 'r') as f:
            print(f"Using categories translation from file at: {args.category_translation}")
            categoriesJson=json.load(f)
    else:
        ww = WasteWizard()
        for cat in cats:
            combined = {"Old Category":cat["name"],
                        ## Swap these two lines below to 1 use search or 2 waste bin 
                        # "New Category":str(ww.searchTerm(cat["name"]))}
                        "New Category":str(ww.getBestOption(ww.searchId(cat["name"])))}
            print(combined)
            categoriesJson.append(combined)
        with open(args.category_translation, 'w') as f:
            json.dump(categoriesJson, f)
            print(f"File saved to {args.category_translation}")
    oldCategories = [i["Old Category"] for i in categoriesJson]
    newCategories = [i["New Category"] for i in categoriesJson]
    print(f"Condensed {len(set(oldCategories))} categories down to {len(set(newCategories))}")
    print(f"{set(newCategories)}")
    return categoriesJson


def getCocoAnnotations(annotation_file):
    downloadRepository()
    downloadedData = os.path.join(args.dataset_path, 'data')
    annotation_file=os.path.join(args.dataset_path, "data/annotations.json")
    coco_annotation = COCO(annotation_file=annotation_file)
    ww = WasteWizard()

    # Category IDs.
    cat_ids = coco_annotation.getCatIds()
    cats = coco_annotation.loadCats(cat_ids)
    # cat_names = [cat["name"] for cat in cats]
    print("Categories translation")
    
    ## Create category translation list
    categoriesJson = loadCategoryTranslation(cats)

    # Loop through all images
    img_ids = coco_annotation.getImgIds()
    newAnnotationsList = [] 
    with Bar(f"Parsing {len(img_ids)} images: ", max=len(img_ids)) as bar:
        for i in range(len(img_ids)):
            img_id = img_ids[i]
            bar.next()
            img_info = coco_annotation.loadImgs([img_id])[0]
            img_file_name = img_info["file_name"]
            img_url = img_info["coco_url"]

            # Get all the annotations for the specified image.
            ann_ids = coco_annotation.getAnnIds(imgIds=[img_id], iscrowd=None)
            anns = coco_annotation.loadAnns(ann_ids)
            # Loop through each object in the images

            for objectAnnotation in anns:
                oldCategory = categoriesJson[objectAnnotation['category_id']]["Old Category"]
                newCategory = categoriesJson[objectAnnotation['category_id']]["New Category"]
                newAnnotation = {
                    'file_path':os.path.join(os.getcwd(),downloadedData, img_file_name),
                    'Original category': oldCategory,
                    'New category': newCategory,
                    'x': objectAnnotation['bbox'][0],
                    'y':objectAnnotation['bbox'][1],
                    'width':objectAnnotation['bbox'][2],
                    'height':objectAnnotation['bbox'][3],
                }
                newAnnotationsList.append(newAnnotation)
        with open(args.output, 'w') as f:
            json.dump(newAnnotationsList, f)
            print(f"File saved to {args.output}")
        return newAnnotationsList


if __name__ == "__main__":
    taco = getCocoAnnotations('TACO')