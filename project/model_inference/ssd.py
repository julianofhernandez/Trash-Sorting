
def ssd_preds(images, model_name):
    # TODO: NEED TO LOAD ACTUAL MODEL
    if not isinstance(images, list):
        images = [images]

    result = {
        'bounding_box': [((.5, .5), (.7, .7))],
        'class_probs': [1.0, 0, 0, 0, 0]
    }

    return [result for image in images]
