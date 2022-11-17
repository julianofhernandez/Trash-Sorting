import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import requests
from pycocotools.coco import COCO
from WasteWizard import WasteWizard


def getCocoAnnotations(annotation_file):
    coco_annotation = COCO(annotation_file=annotation_file)
    ww = WasteWizard()

    # Category IDs.
    cat_ids = coco_annotation.getCatIds()
    # All categories.
    cats = coco_annotation.loadCats(cat_ids)
    cat_names = [cat["name"] for cat in cats]

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
                'file_path':img_file_name,
                'Original category': newCategory,
                'New category':ww.searchTerm(newCategory),
                'x': objectAnnotation['bbox'][0],
                'y':objectAnnotation['bbox'][1],
                'width':objectAnnotation['bbox'][2],
                'height':objectAnnotation['bbox'][3],
            }
            print(str(newAnnotation))
            newAnnotationsList.append(newAnnotation)
    print(str(len(newAnnotationsList))+' annotatins parsed')
    return newAnnotationsList



if __name__ == "__main__":
    taco = getCocoAnnotations('TACO/data/annotations.json')
    print(str(taco))
    json.dumps(taco)

class WasteWizard:
    recollect_url = ''
    score_threshold = 0.1
    
    def __init__(self, url='http://api.recollect.net/api/areas/Sacramento'):
        self.recollect_url = url

    def searchTerm(self,query):
        '''Uses fuzzySearch but only returns the top result
        returns title'''
        results = self.fuzzySearch(query)
        # If there's no results, or the results aren't confident, return None
        if len(results) == 0:
            return None
        if float(results[0]['score']) < self.score_threshold:
            return None
        else:
            return results[0]['title']

    def searchId(self, query):
        '''Uses fuzzySearch but only returns the top result
        returns id'''
        results = self.fuzzySearch(query)
        # If there's no results, or the results aren't confident, return None
        if len(results) == 0:
            return None
        if float(results[0]['score']) < self.score_threshold:
            return None
        else:
            return results[0]['id']

    def fuzzySearch(self, query, scores=False):
        '''Searches the waste wizard and returns a list of matches
        serachResult = [{title: plastic, id: 114556, score: 5.6},
                        {title: plastic bottle, id: 45244, score: 4.5}]'''
        url = self.recollect_url+'/services/waste/pages?suggest='+query+'&type=material&set=default&include_links=true&locale=en-US&accept_list=true&_=1666888380289'
        response = self.recollect_api(url)
        searchResults = []
        for result in response.json():
            searchResults.append({
                'title': result['title'],
                'id': result['id'],
                'score': result['score']
            })
        return searchResults

    def getBestOption(self, id):
        '''Searches by id and returns the returns how to dispose of it'''
        url = 'https://api.recollect.net/api/areas/Sacramento/services/waste/pages/en-US/'+id+'.json'
        response = self.recollect_api(url)
        for section in response.json()['sections']:
            try:
                if section['title'].lower() == 'best option':
                    return section['rows'][0]['value']
            except:
                pass
        return None

    def getSpecialInstructions(self, id):
        '''Doesn't work for every entry, but will return HTML instructions for special items'''
        url = 'https://api.recollect.net/api/areas/Sacramento/services/waste/pages/en-US/'+id+'.json'
        response = self.recollect_api(url)
        for section in response.json()['sections']:
            try:
                if section['title'].lower() == 'special instructions':
                    return section['rows'][0]['value']
            except:
                pass
        return None

    def listBestOptions(self):
        '''Lists all possible trash/recycling/compost/hazard categries'''

    def listAllCategories(self):
        return ["trash", "reyclcing"]

    def recollect_api(self, url):
        proxies = {
            'http': 'http://proxy-chain.intel.com:911'
        }
        response = requests.get(url, proxies=proxies)
        if response.status_code == 200:
            return response
        else:
            print("Failed response from API")