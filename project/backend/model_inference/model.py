import os
import json
import numpy as np
import torch
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
import open_clip

MODELS = {}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

"""
The preds() function takes in a list of images and a model name as input and 
returns a list of dictionaries containing the classification results for each image. 
It first checks if the input is a list and, if not, it wraps the input in a list. Then, 
based on the model name, it calls the appropriate function to classify the images. The
supported models are 'efficientnet_v2_l' and 'clip'. If the model name is 'DEFAULT', 
it will be replaced with 'clip'.
"""


def preds(images, model_name):
    """
    This function takes in a list of images and a model name and
    eturns a list of dictionaries containing the classification results for each image.

    Args:
    images (list or array-like): List of images to classify
    model_name (str): Name of the model to use for classification

    Returns:
    results (list of dicts): List of dictionaries containing classification results for each image
    """
    # Check if input is a list, and if not assume it is one image and wrap it in a list
    if not isinstance(images, list):
        images = [images]

    if model_name.upper() == "DEFAULT":
        model_name = "efficientnet_v2_l"
        # model_name = "ViT-g-14"

    # Match model name to the appropriate model specific preds method
    if model_name.startswith("efficientnet_v2_l"):
        results = efficentnet_preds(images, model_name)
    elif model_name.startswith("test"):
        results = old_test_preds(images, model_name)
    elif model_name.startswith("ViT"):
        results = open_clip_preds(images, model_name)
    else:
        results = []

    return results


def old_test_preds(images, model_name):
    """
    The old_test_preds() function is a dummy function that returns the same classification
    results for all input images. It returns a list of dictionaries with keys 'object_class',
    'object_class_probs', 'object_classes', 'object_trash_class', 'object_trash_class_probs',
    'trash_class', 'trash_class_probs', and 'trash_classes'.
    """
    result = {
        # 'bounding_box': [((.5, .5), (.7, .7))],
        # 'class_probs': [1.0, 0, 0, 0, 0],
        # 'obj_label': 'test',
        # 'trash_bin_label': 'test'
        "object_class": "Other plastic bottle",
        "object_class_probs": [
            0.3539431691169739,
            0.3484818935394287,
            0.06859111785888672,
            0.054287757724523544,
            0.039106689393520355,
            0.03449060395359993,
            0.030443403869867325,
            0.026867564767599106,
            0.026444964110851288,
            0.0173428263515234,
        ],
        "object_classes": [
            "Other plastic bottle",
            "Clear plastic bottle",
            "Other plastic container",
            "Disposable plastic cup",
            "Other plastic cup",
            "Other plastic",
            "Polypropylene bag",
            "Plastic glooves",
            "Plastic lid",
            "Single-use carrier bag",
        ],
        "object_trash_class": "Garbage",
        "object_trash_class_probs": [0.25, 0.25, 0.25, 0.25],
        "trash_class": "Recyclable",
        "trash_class_probs": [
            0.02020263671875,
            0.88623046875,
            0.07867431640625,
            0.01500701904296875,
        ],
        "trash_classes": [
            "Garbage",
            "Recyclable",
            "Organic Waste",
            "Household hazardous waste",
        ],
    }
    return [result for _ in images]


# The efficentnet_preds() function takes in a list of images and an efficientnet model name as input and returns a
# list of dictionaries containing the classification results for each image. It checks if the specified model has
# been loaded previously, and if so, it uses that model. If the specified model has not been loaded previously, it
# loads the default model and saves it to MODELS. It iterates through each input image and predicts its classes. It
# then calculates the probability distribution over the four trash classes and returns a dictionary with keys
# 'object_class_probs', 'object_classes', 'trash_class_probs', and 'trash_classes'.


def efficentnet_preds(images, model_name):
    """This function takes in a list of images and a model name and returns a list of dictionaries containing the classification results for each image.

    Args:
    images (list or array-like): List of images to classify
    model_name (str): Name of an efficentnet model to use for classification

    Returns:
    results (list of dicts): List of dictionaries containing classification results for each image
    """
    global MODELS

    # Check if the specified model has been loaded previously, and if so, use that model
    if model_name in MODELS:
        model, transform, labels, conversion_dict = MODELS[model_name]

    # If the specified model has not been loaded previously, load the default model and save it to MODELS
    # assuming the model exists
    else:
        MODELS.clear()

        model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT).to(DEVICE)
        model.eval()
        transform = EfficientNet_V2_L_Weights.DEFAULT.transforms()
        labels = EfficientNet_V2_L_Weights.DEFAULT.meta["categories"]

        # load json file but join path with os to __file__
        conversion_dict = json.load(
            open(
                os.path.join(
                    os.path.dirname(__file__),
                    "models/efficientnet_v2_l/conversion_dict.json",
                )
            )
        )

        # Cache the model and its associated data
        MODELS[model_name] = (model, transform, labels, conversion_dict)

    # Iterate through each input image and predict its classes
    results = []
    for image in images:
        with torch.no_grad():
            outputs = model(transform(image).unsqueeze(0).to(DEVICE))
        outputs = outputs[0].cpu()
        ndxs = torch.topk(outputs, k=10).indices.squeeze(0).numpy()
        outputs = outputs.numpy()

        # Calculate probability distribution over the four trash classes
        trash_probs = np.array([0.0, 0.0, 0.0, 0.0])
        trash_classes = [
            "Garbage",
            "Recyclable",
            "Organic Waste",
            "Household hazardous waste",
        ]
        object_class_probs = outputs[ndxs]
        object_class_probs = object_class_probs / object_class_probs.sum()
        object_classes = [labels[ndx] for ndx in ndxs]

        # Sum probabilities for each trash class from the top 10 predictions
        for ndx in ndxs:
            trash_probs[conversion_dict[str(ndx)]["index"]] += outputs[ndx]
        trash_probs = trash_probs / trash_probs.sum()

        trash_index = conversion_dict[str(ndxs[0])]["index"]

        # Save classification results for the current image to a dictionary and append it to the results list
        result = {
            "object_class": labels[ndxs[0]],
            "object_class_probs": object_class_probs.tolist(),
            "object_classes": object_classes,
            # conversion_dict[str(ndxs[0])]['bin_class']
            "object_trash_class": trash_classes[trash_index],
            "object_trash_class_probs": trash_probs.tolist(),
            "trash_class": trash_classes[np.argmax(trash_probs)],
            "trash_class_probs": trash_probs.tolist(),
            "trash_classes": trash_classes,
        }
        results.append(result)

    return results


def get_weights(CLIP_MODELS):
    m = {
        "ViT-g-14": "laion2B-s12B-b42K",
        "ViT-L-14": "laion2B-s32B-b82K",
        "ViT-B-16": "laion2B-s34B-b88K",
    }
    return m[CLIP_MODELS]


# The code relies on the PyTorch and TorchVision libraries and the clip module for the clip_preds() function.


def open_clip_preds(images, model_name):
    """This function takes in a list of images and a model name and returns a list of dictionaries containing the classification results for each image.

    Args:
    images (list or array-like): List of images to classify
    model_name (str): Name of a clip model to use for classification

    Returns:
    results (list of dicts): List of dictionaries containing classification results for each image
    """
    # If the specified model has not been loaded previously, load it
    if model_name in MODELS:
        (
            model,
            preprocess,
            object_text_features,
            trash_bin_text_features,
            object_categories,
            trash_bins_categories,
            conversion_dict,
        ) = MODELS[model_name]
    else:
        MODELS.clear()
        # Load the CLIP model and preprocess function

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=get_weights(model_name)
        )

        model = model.to(DEVICE)

        tokenizer = open_clip.get_tokenizer(model_name)

        # Define the object and trash bin categories and their descriptions
        object_categories = [
            "Aluminium foil",
            "Battery",
            "Aluminium blister pack",
            "Carded blister pack",
            "Other plastic bottle",
            "Clear plastic bottle",
            "Glass bottle",
            "Plastic bottle cap",
            "Metal bottle cap",
            "Broken glass",
            "Food Can",
            "Aerosol",
            "Drink can",
            "Toilet tube",
            "Other carton",
            "Egg carton",
            "Drink carton",
            "Corrugated carton",
            "Meal carton",
            "Pizza box",
            "Paper cup",
            "Disposable plastic cup",
            "Foam cup",
            "Glass cup",
            "Other plastic cup",
            "Food waste",
            "Glass jar",
            "Plastic lid",
            "Metal lid",
            "Other plastic",
            "Magazine paper",
            "Tissues",
            "Wrapping paper",
            "Normal paper",
            "Paper bag",
            "Plastified paper bag",
            "Plastic film",
            "Six pack rings",
            "Garbage bag",
            "Other plastic wrapper",
            "Single-use carrier bag",
            "Polypropylene bag",
            "Crisp packet",
            "Spread tub",
            "Tupperware",
            "Disposable food container",
            "Foam food container",
            "Other plastic container",
            "Plastic glooves",
            "Plastic utensils",
            "Pop tab",
            "Rope & strings",
            "Scrap metal",
            "Shoe",
            "Squeezable tube",
            "Plastic straw",
            "Paper straw",
            "Styrofoam piece",
            "Unlabeled litter",
            "Cigarette",
        ]
        trash_bins_categories = [
            "Garbage",
            "Recyclable",
            "Organic Waste",
            "Household hazardous waste",
        ]

        # Encode the object and trash bin descriptions using the CLIP model
        object_descriptions = [f"photo of a {label}" for label in object_categories]
        trash_bins_descriptions = [
            f"photo of {label}" for label in trash_bins_categories
        ]
        trash_bins_descriptions[
            0
        ] += " (non-recyclable plastics; not recyclable, not organic, and not hazardous waste)"
        trash_bins_descriptions[1] += " (glass, plastics, cans, paper, or cardboard)"
        trash_bins_descriptions[2] += " (food, plants, or other compostable items)"
        trash_bins_descriptions[
            3
        ] += " (batteries, electronics, or other hazardous items)"

        # Tokenize and send to device
        object_descriptions_tokens = tokenizer(object_descriptions).to(DEVICE)
        trash_bins_descriptions_tokens = tokenizer(trash_bins_descriptions).to(DEVICE)

        # Encode the descriptions
        with torch.no_grad():
            object_text_features = model.encode_text(object_descriptions_tokens)
            trash_bin_text_features = model.encode_text(trash_bins_descriptions_tokens)

        # Normalize the features
        object_text_features /= object_text_features.norm(dim=-1, keepdim=True)
        trash_bin_text_features /= trash_bin_text_features.norm(dim=-1, keepdim=True)

        conversion_dict = json.load(
            open(
                os.path.join(
                    os.path.dirname(__file__),
                    "models/efficientnet_v2_l/conversion_dict.json",
                )
            )
        )

        # Cache the model and its associated data
        MODELS[model_name] = (
            model,
            preprocess,
            object_text_features,
            trash_bin_text_features,
            object_categories,
            trash_bins_categories,
            conversion_dict,
        )

    results = []

    for image in images:
        image = preprocess(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            image_features = model.encode_image(image)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        object_similarity = (
            (100.0 * image_features @ object_text_features.T)
            .softmax(dim=-1)[0]
            .cpu()
            .type(torch.float)
        )
        trash_bin_probs = (
            (100.0 * image_features @ trash_bin_text_features.T)
            .softmax(dim=-1)[0]
            .cpu()
            .type(torch.float)
            .numpy()
        )

        object_class_probs, object_indices = object_similarity.topk(10)
        object_indices = object_indices.numpy()
        object_class_probs = object_class_probs.numpy()
        object_class_probs = object_class_probs / object_class_probs.sum()
        object_classes = [object_categories[index] for index in object_indices]
        object_class = object_categories[object_indices[0]]

        object_trash_probs = np.array([0.0, 0.0, 0.0, 0.0]) + 1
        object_trash_probs = object_trash_probs / object_trash_probs.sum()

        result = {
            # CLIP prediction of most likely object type
            "object_class": object_class,
            "object_class_probs": object_class_probs.tolist(),
            "object_classes": object_classes,
            # From CLIP object prediction, predict what bin it goes to
            "object_trash_class": trash_bins_categories[np.argmax(object_trash_probs)],
            "object_trash_class_probs": object_trash_probs.tolist(),
            # CLIP prediction of what bin is the trash going to
            "trash_class": trash_bins_categories[np.argmax(trash_bin_probs)],
            "trash_class_probs": trash_bin_probs.tolist(),
            "trash_classes": trash_bins_categories,
        }

        results.append(result)

    return results
