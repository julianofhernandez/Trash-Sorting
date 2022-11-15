# Dataset
This will contain the code related to [this JIRA issue](https://santiagobermudez.atlassian.net/jira/software/projects/BDE/boards/2?selectedIssue=BDE-36) for creating a custom dataset. In order to make use computer vision within the field of trash management, a database of images will be needed to accurately train a neural network. It's estimated that we will need over 1,000 images per class [1], and with over 100 individual classes outlined in the Sacramento Waste Wizard, the total number of images needed quickly rises into the 100,000s. Unfortunetly, almost all researchers in this space have elected to create their own datasets instead of building off of someone elses. This has caused a great amount of fragementation, a review of current trash image datsets has shwon that over 158,000 publically available images are spread accross 20 different datasets [3]. But because they aren't considated, using them is extremely difficult and has held back the future prospects of using neural networks for trash recognition. To fix this we are going to both, combine currently existsing datasets, and utilize volunteer clean up hours to collect new data. The scripts within this folder will be used to consolidate the currently existing datasets into categories that can be searched in the Waste Wizard.

## Formats: Pascal VOC vs COCO
To tell our computer vision algorithm where objects are within an image, we will need to choose a set of formats to support that can be downloaded. Because we are starting with SSD we can use Pascal VOC or COCO format. Both of them support classification and object detection, but Pascal VOC uses XML for annotations whereas COCO uses JSON files. For the amount of files they use, Pascal needs one XML file for each image, whereas COCO uses one file per group (train, validation, and testing). 

## Pascal VOC
https://arxiv.org/abs/2104.10972
Image sample

## COCO sample
Image sample