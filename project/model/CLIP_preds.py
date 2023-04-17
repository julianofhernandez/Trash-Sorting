import torch  # Imports the PyTorch library.
import clip

MODELS = {}

device = "cuda" if torch.cuda.is_available() else "cpu"


def clip_preds(images, model_name):
    """^Defines a function named clip_preds() that takes in a list of images and a model name as input. The
    function classifies each image and returns a list of dictionaries containing the classification results
    for each image."""

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
        (
            model,
            preprocess,
            object_text_features,
            trash_bin_text_features,
            object_categories,
            trash_bins_categories,
        ) = MODELS[model_name]
    else:
        # Load the CLIP model and preprocess function
        model, preprocess = clip.load("ViT-B/32")
        model.eval()
        # model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
        # tokenizer = open_clip.get_tokenizer('ViT-H-14')

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
        object_descriptions = [  # Define lists of object and trash bin descriptions using f-strings, which allow for embedding variable values into strings.
            f"photo of a {label}" for label in object_categories
        ]
        trash_bins_descriptions = [
            f"photo of {label}" for label in trash_bins_categories
        ]

        """The next two lines tokenize the descriptions using the clip.tokenize() method and assign the resulting 
        tensors to variables object_descriptions_tokens and trash_bins_descriptions_tokens. These tensors are 
        then moved to the device specified earlier in the code."""
        object_descriptions_tokens = clip.tokenize(object_descriptions).to(device)
        trash_bins_descriptions_tokens = clip.tokenize(trash_bins_descriptions).to(
            device
        )

        """The next two lines encode the text features using the model.encode_text() method, which produces 
        feature vectors for each description. These vectors are then assigned to variables object_text_features 
        and trash_bin_text_features."""
        object_text_features = model.encode_text(object_descriptions_tokens)
        trash_bin_text_features = model.encode_text(trash_bins_descriptions_tokens)

        # Normalize the features
        object_text_features /= object_text_features.norm(dim=-1, keepdim=True)
        trash_bin_text_features /= trash_bin_text_features.norm(dim=-1, keepdim=True)

        MODELS[model_name] = (
            model,
            preprocess,
            object_text_features,
            trash_bin_text_features,
            object_categories,
            trash_bins_categories,
        )

    results = []

    for image in images:  # Looping through each image in the input list images.
        image = (
            preprocess(image).unsqueeze(0).to(device)
        )  # Preprocessing the image and converting it to a tensor.
        with torch.no_grad():
            image_features = model.encode_image(
                image
            )  # Encoding the image tensor into a feature vector using the CLIP model.

            image_features /= image_features.norm(
                dim=-1, keepdim=True
            )  # Normalizing the image feature vector.
            object_similarity = (
                100.0 * image_features @ object_text_features.T
            ).softmax(dim=-1)[0]
            trash_bin_probs = (
                100.0 * image_features @ trash_bin_text_features.T
            ).softmax(dim=-1)[0]
            # ^Calculating the cosine similarity between the image feature vector and the pre-encoded feature vectors of each object category and trash bin category using matrix multiplication.

            # object_class_probs = object_class_probs.cpu()
            object_class_probs, object_indices = object_similarity.topk(
                10
            )  # Selecting the top 10 most similar object categories based on the probability scores.
            object_indices = object_indices.numpy()
            object_class_probs = object_class_probs.numpy()
            object_class_probs = object_class_probs / object_class_probs.sum()
            object_classes = [object_categories[index] for index in object_indices]

            obj_max_val = 0
            for value, index in zip(object_class_probs, object_indices):
                obj_max_val = max(
                    obj_max_val, value.item()
                )  # Determining the most likely object class by finding the maximum probability score among the top 10.
                print(f"{object_categories[index]}: {value.item():.2f}")

            get_probs = object_class_probs.tolist()
            object_class = object_classes[
                get_probs.index(obj_max_val)
            ]  # Selecting the most probable trash bin class based on the similarity scores for trash bins.

            bin_max_val = 0
            for index, prob in enumerate(trash_bin_probs):
                bin_max_val = max(bin_max_val, prob.item())
                print(f"{trash_bins_categories[index]}: {prob.item():.2f}")

            object_trash_class = trash_bins_categories[
                ((trash_bin_probs == bin_max_val).nonzero(as_tuple=False)).item()
            ]

        result = {  # Creating a dictionary called result containing the predicted object class, its probability scores, labels of possible trash types, and labels of possible bins.
            "object_class": object_class,  # CLIP prediction of most likely trash type
            # CLIP prediction of what the trash is
            "object_class_probs": object_class_probs.tolist(),
            "object_classes": object_classes,  # labels of possible trash types
            "object_trash_class": object_trash_class,  # object type and its bin
            # CLIP prediction of what bin is the trash going to
            "trash_class_probs": trash_bin_probs.tolist(),
            "trash_classes": trash_bins_categories,  # labels of possible bins
        }

        results.append(result)  # Appending the dictionary to a list called results.

    return results  # Returning the list of dictionaries containing predictions for all input images.


"""
path = ''
clip_preds(Image.open(path), clip)
"""
