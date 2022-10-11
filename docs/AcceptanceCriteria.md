**Front-End**

Menu Prompt

* The menu prompt must allow users to navigate to different parts of the application, mainly classifying
    * Entering option 1 will trigger the classify trash prompt
    * Entering option 2 will trigger the annotate trash prompt
    * Entering option 3 will trigger the trash info prompt
    * Entering option 4 will trigger the settings prompt
    * Entering option 5 will trigger the about prompt
    * Entering option 6 will quit the program
* If the user enters an invalid option, they will be prompted to enter a valid one.

Classify Prompt

* Must provide options for user to (1) classify from live webcam, (2) webcam capture, or (3) file/folder
* User must have ability to (4) return to menu prompt
* If the user enters an invalid option, they will be prompted to enter a valid one.
* Displays classification and its corresponding brief description to user momentarily after successful submission
    * Should display an error to the user if classification was unsuccessful 

Annotate Trash Prompt

* Provides options for (1) annotating a webcam capture or (2) file/folder
* User must have ability to (3) return to menu prompt
* If the user enters an invalid option, they will be prompted to enter a valid one.

Trash description Prompt

* Displays descriptions for the different classifications (trash, recycle, compost)
* Ability to return to menu

Settings Prompt

* Displays settings view
* Ability to return to menu

About Prompt

* Displays brief description of interface and functions
* Credits for developers and dataset(s)
* Ability to return to menu

**Backend**

* Model Inference: Able to take an image from front-end and return classifications
    * Must be able to handle invalid input (wrong file type, etc.) without crashing or displaying unknown behavior 
* Model download: Able to send current optimized model to front-end
    * Must do so in reasonable amount of time; otherwise timeout
* Data annotation: Able to serve front-end in performing annotations to expand dataset
    * Uploading images
    * Uploading images with annotations
    * Downloading images
    * Downloading images with annotations (to validate)
        * Previous annotations user made or machine predicted

**Model**

* Accuracy above 75% for validation and test sets or a 5% improvement on pretrained weights
* Efficient can run on CPU with at least FPS of 1
* Classifies objects at least into trash, recycle, compost. But preferably first into categories which are then binned into these three options. 
* Classifies an image or preferably can put bounding boxes on objects within images

**Dataset**

* Dataset contains many different objects in different contexts labeled with bounding boxes and images
* Dataset should either label individual objects or should be labeled as the classification they fall into whether trash, recycle, or compost
