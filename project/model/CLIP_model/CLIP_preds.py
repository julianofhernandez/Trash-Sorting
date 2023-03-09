import torch
import clip
import json
import numpy as np
from PIL import Image

MODELS = {}

device = "cuda" if torch.cuda.is_available() else "cpu"

def clip_preds(images, model_name):
    """This function takes in a list of images and a model name and returns a list of dictionaries containing the classification results for each image.

    Args:
    images (list or array-like): List of images to classify
    model_name (str): Name of the model to use for classification

    Returns:
    results (list of dicts): List of dictionaries containing classification results for each image
    """
    global MODELS
    # Check if input is a list, and if not assume it is one image and wrap it in a list
    if not isinstance(images, list):
        images = [images]

    # If the specified model has not been loaded previously, load it
    if model_name in MODELS:
        model, preprocess, object_text_features, trash_bin_text_features, object_categories, trash_bins_categories = MODELS[model_name]
    else:
        # Load the CLIP model and preprocess function
        model, preprocess = clip.load("ViT-B/32")
        model.eval()
        #model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
        # tokenizer = open_clip.get_tokenizer('ViT-H-14')

        # Define the object and trash bin categories and their descriptions
        object_categories = ["Aluminium foil", "Battery", "Aluminium blister pack", "Carded blister pack", 
                             "Other plastic bottle", "Clear plastic bottle", "Glass bottle", 
                             "Plastic bottle cap", "Metal bottle cap", "Broken glass", "Food Can", 
                             "Aerosol", "Drink can", "Toilet tube", "Other carton", "Egg carton", 
                             "Drink carton", "Corrugated carton", "Meal carton", "Pizza box", "Paper cup", 
                             "Disposable plastic cup", "Foam cup", "Glass cup", "Other plastic cup", 
                             "Food waste", "Glass jar", "Plastic lid", "Metal lid", "Other plastic", 
                             "Magazine paper", "Tissues", "Wrapping paper", "Normal paper", "Paper bag", 
                             "Plastified paper bag", "Plastic film", "Six pack rings", "Garbage bag", 
                             "Other plastic wrapper", "Single-use carrier bag", "Polypropylene bag", 
                             "Crisp packet", "Spread tub", "Tupperware", "Disposable food container", 
                             "Foam food container", "Other plastic container", "Plastic glooves", 
                             "Plastic utensils", "Pop tab", "Rope & strings", "Scrap metal", "Shoe", 
                             "Squeezable tube", "Plastic straw", "Paper straw", "Styrofoam piece", 
                             "Unlabeled litter", "Cigarette"]
        trash_bins_categories = ["Garbage","Recylable","Organic Waste","Household hazardous waste"]
        
        # Encode the object and trash bin descriptions using the CLIP model
        object_descriptions = [f"photo of a {label}" for label in object_categories]
        trash_bins_descriptions = [f"photo of {label}" for label in trash_bins_categories]
        
        object_descriptions_tokens = clip.tokenize(object_descriptions).to(device)
        trash_bins_descriptions_tokens = clip.tokenize(trash_bins_descriptions).to(device)

        object_text_features = model.encode_text(object_descriptions_tokens)
        trash_bin_text_features = model.encode_text(trash_bins_descriptions_tokens)

        # Normalize the features
        object_text_features /= object_text_features.norm(dim=-1, keepdim=True)
        trash_bin_text_features /= trash_bin_text_features.norm(dim=-1, keepdim=True)
        
        MODELS[model_name] = (model, preprocess, object_text_features, trash_bin_text_features, object_categories, trash_bins_categories)

    results = []

    for image in images:
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            object_similarity = (100.0 * image_features @ object_text_features.T).softmax(dim=-1)[0]
            trash_bin_probs = (100.0 * image_features @ trash_bin_text_features.T).softmax(dim=-1)[0]

            #object_class_probs = object_class_probs.cpu()
            object_class_probs, object_indices = object_similarity.topk(10)
            object_indices = object_indices.numpy()
            object_class_probs = object_class_probs.numpy()
            object_class_probs = object_class_probs / object_class_probs.sum()
            object_classes = [object_categories[index] for index in object_indices]
            
            obj_max_val = 0
            for value, index in zip(object_class_probs, object_indices):
                obj_max_val = max(obj_max_val, value.item())
                print(f"{object_categories[index]}: {value.item():.2f}")
                
            get_probs = object_class_probs.tolist() 
            object_class = object_classes[get_probs.index(obj_max_val)]
            
            bin_max_val = 0
            for index, prob in enumerate(trash_bin_probs):
                bin_max_val = max(bin_max_val, prob.item())
                print(f"{trash_bins_categories[index]}: {prob.item():.2f}")
            
            object_trash_class = trash_bins_categories[((trash_bin_probs == bin_max_val).nonzero(as_tuple=False)).item()]
            
            
        result = {
            'object_class': object_class, # CLIP prediction of most likely trash type
            'object_class_probs': object_class_probs.tolist(), # CLIP prediction of what the trash is
            
            'object_classes': object_classes, # labels of possible trash types
            'object_trash_class': object_trash_class, # object type and its bin
            
            'trash_class_probs': trash_bin_probs.tolist(), # CLIP prediction of what bin is the trash going to 
            'trash_classes': trash_bins_categories # labels of possible bins
        }
        
        results.append(result)

    return results


"""
path = ''
clip_preds(Image.open(path), clip)
"""
