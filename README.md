![image](https://github.com/julianofhernandez/Trash-Sorting/assets/39971693/27b97de6-c49f-4a99-8e29-f9d85d636a0c)


## Problem:

The mismanagement of waste is a pressing concern that affects the environment and human health. Despite efforts to promote proper waste disposal, many individuals continue to dispose of trash haphazardly, resulting in significant pollution and harm to natural ecosystems.

One of the main issues is the mis-sorting of trash, where items are placed in the wrong bin. This problem is particularly urgent as it can lead to contamination of recycling, rendering it unusable and increasing the amount of waste that ends up in landfills.

Another problem is the lack of awareness around composting and the need for new compost requirements. This results in organic waste being sent to landfills where it releases methane, a potent greenhouse gas that contributes to climate change.

## Proposed Solution:

To address these pressing issues, our team proposes the development of an application that utilizes deep neural network models to locate and classify trash in images. To compare results from multiple models we are using EfficientNetV2 and CLIP. The application allows users to either upload an image, or use their camera live for classification. For privacy we allowed all model to be downloaded and inference to be performed locally, while it performs best on a device with a GPU, we tested it on CPU only as well. The application will help individuals properly dispose of waste by identifying the appropriate bin for each item, reducing the risk of contamination and increasing the amount of waste that can be recycled or composted. Our solution will not only mitigate the impact of waste on the environment but also promote healthier communities and cleaner cities.

Website:

![image](https://user-images.githubusercontent.com/39971693/232918769-e3da0eeb-07ef-4e7b-abe6-251e3450ef8a.png)

Terminal App:

`python projects/front-end_apps/main.py`

![image](https://user-images.githubusercontent.com/47340315/235587196-e1dd438b-1d8e-45f9-8b39-ffa71f7b101a.png)

Terminal Script:

`python projects/front-end_apps/trashsorting.py`

![image](https://user-images.githubusercontent.com/47340315/235587100-d3fe83ab-d060-4aab-91c0-39a30bb0287c.png)


## Installation

### Python 

Python Version 3.9 available at https://www.python.org/downloads/release/python-390/. Then run ```python3 -m pip install -r requirements.txt```

OR

Conda: install Anaconda, create an environment `conda create --name trashsorting`, activate environment `conda activate trashsorting`, install requirements `pip install -r requirements.txt`

For GPU torch, instead of installing torch in requirements.txt do the below:

Windows/Linux: `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`

### NodeJS 

Download and install the latest version of Node.js from the official website: https://nodejs.org/en/download/. Follow the installation instructions for your operating system.

### Yarn

Once NPM is installed with NodeJS you can run
```npm install --global yarn```

## How to run the server

Change to react-website directory and run following commands:
```cd project\front-end_apps\website\react-website\```

1. `yarn install` (may also need to run `yarn add tabler-react tabler-icons-react`)

2. `yarn build` generates a build folder in the current directory

2. `yarn start` to run on http://localhost:3000/

3. start model_inference_app.py in separate terminal to test submitting images


## Testing
#### Front End
```
cd project/front-end-apps
pytest
```
![image](https://user-images.githubusercontent.com/47340315/235586207-a89e2338-49ca-4b29-a3f1-da1273373592.png)

```
cd project\front-end_apps\website\react-website
yarn test
```

![image](https://user-images.githubusercontent.com/39971693/232924965-5f8c446e-5adc-4520-b1f3-067fbd0cf3db.png)


#### Backend
```
cd project/backend/annotation_database
pytest
```

![image](https://user-images.githubusercontent.com/47340315/235585828-c5991630-7889-47c0-8fba-3b12884d8e86.png)


```
cd project/backend/model_inference
pytest
```

![image](https://user-images.githubusercontent.com/47340315/235586790-cc21b96c-d91a-4dcf-9b70-18b50042c3ad.png)


## Development Instructions

### Environment
We recommend using Anaconda to create a Python environment with all the requirements outlined in the requirements.txt file. If you're not familiar with Anaconda, please see the official website or refer to this cheat sheet for the proper commands.

### Code Editor
We recommend using Visual Studio Code (VS Code) with the Python Language and running Jupyter Notebook for this project.

### OS/System Independent
The front-end and backend work on Windows, Mac, and Linux operating systems. However, to run the backend smoothly, we recommend your system to have at least 8GB of RAM and for fast response, a Nvidia GPU with at least 8GB of VRAM.

Please make sure that your system meets these requirements before proceeding with the setup.

## Awards

1st place at the Annual CSU Wide Research Competition

2nd place at the Sacramento Spring Research Symposium

![image](https://github.com/julianofhernandez/Trash-Sorting/assets/39971693/eba30d5e-a52b-4ede-a5c1-e30b495f8530)


## Client

Dr. Clark Fitzgerald

## Contributors

Travis Hammond
⁢

Kenta Miyahara
⁢

Julian Hernandez
⁢

Jeffrey de Jesus
⁢

Christopher Allen


Bryan Burch
⁢

Daniel Smagly
⁢

Santiago Bermudez

