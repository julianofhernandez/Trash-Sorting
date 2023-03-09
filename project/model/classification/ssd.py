import torch
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
import json
import numpy as np

MODELS = {}


def ssd_preds(images, model_name):
    global MODELS
    if not isinstance(images, list):
        images = [images]

    if model_name in MODELS:
        model, transform, labels, conversion_dict = MODELS[model_name]
    else:
        model = efficientnet_v2_l(
            weights=EfficientNet_V2_L_Weights.DEFAULT).to('cuda')
        model.eval()
        transform = EfficientNet_V2_L_Weights.DEFAULT.transforms()
        labels = EfficientNet_V2_L_Weights.DEFAULT.meta['categories']
        conversion_dict = json.load(open('conversion_dict.json'))
        MODELS[model_name] = (model, transform, labels, conversion_dict)

    results = []
    for image in images:
        with torch.no_grad():
            outputs = model(transform(image).unsqueeze(0).to('cuda'))
        outputs = outputs[0].cpu()
        ndxs = torch.topk(outputs, k=10).indices.squeeze(0).numpy()
        outputs = outputs.numpy()

        trash_probs = np.array([0., 0., 0., 0.])
        trash_classes = ['trash', 'recycling', 'compost', 'ewaste']
        object_class_probs = outputs[ndxs]
        object_class_probs = object_class_probs / object_class_probs.sum()
        object_classes = [labels[ndx] for ndx in ndxs]

        for ndx in ndxs:
            # statically could do better than addition
            trash_probs[conversion_dict[str(ndx)]['index']] += outputs[ndx]
        trash_probs = trash_probs / trash_probs.sum()

        result = {
            'trash_class_probs': trash_probs,
            'trash_classes': trash_classes,

            'object_class_probs': object_class_probs,
            'object_classes': object_classes,

            'object_class': labels[ndxs[0]],
            'object_trash_class': conversion_dict[str(ndxs[0])]['bin_class']
        }
        results.append(result)

    return results
