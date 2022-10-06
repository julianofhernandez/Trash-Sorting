# Object Detection
Steps
1. Pretrained model

Find a pretrained model that is for trash to enable the front end team to quickly see what the connecting API will look like.

2. Collect images

This is a manual process where we need lots of pictures of trash (~10,000 unique images). We can then to preprocessing to increase the total size. Preprocessing is a technique where translations, scaling, and color changes are applied to make new versions of these original images. This is an incredibly important step, there have been many researchers before us but each one creates their own dataset instead of adding to what has already been done, making no progress. We have to add to the common knowledge NOT create a slightly different version of what already exists.

3. Annotate images

We can use [MakesenseAI](https://www.makesense.ai/) to make new annoations quickly. Once we have a basic pretrained model (mAP > 80%) we can use this to make suggested edits for more annotations. Annotations are time consuming and require a lot of people, if we use this tool we can speed this up, we should make sure we start to measure how long this process takes (# annotations per hour) so that we can make predicions like: "If a univserity dedicates 1000 student hours we could improve precision by X%."

5. train model

We should be able to use Google Collab for access free online hardware (GPU or TPU) to process newly annotated images and update previous models to increase their accuracy and generalization.

7. Perform evaluation

We will need to make new visualizations of how training is working to compare against other results to see if we area ctually doing good. 

9. Return result from api

This steps seeks to answer what data do we need to return to the program. 

JSON Example:
```
{
  "c:\\image_path": {
    "detections": {
      "plastic": [x, y, width, height]
      "paper bag": [x, y, width, height]
    }
  }
}
```
## Instance Segmentation: [Mask R-CNN](https://github.com/matterport/Mask_RCNN)
Outlines object

## Bounding Boxes: [YOLO](https://www.mdpi.com/2079-9292/11/9/1323/htm)
0.5 seconds for inference on a Raspberry Pi 4
Training time of 20hours in Google Colab (free?)

## Classification

# Annotations: [MakesenseAI](https://www.makesense.ai/)
Supports COCO and YOLO formats
You can upload object detection that will suggest new annotations
