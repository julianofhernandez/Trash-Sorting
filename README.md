![image](https://user-images.githubusercontent.com/39971693/232920379-6389abe8-5b61-4448-8dcc-5bc3ec5e1cd7.png)

## Proposed solution

The World Bank’s 2012 study of MSW reported that the most important way to reduce consumer greenhouse gas emissions was to educate consumers on best reduction and sorting practices [D Hoornweg]. To help encourage Californian’s to properly separate recyclables and now organic materials, we will use a neural network based program to suggest which bin an item belongs to. Our precision of 82% is better than current Califorian sorting rates of 42% [CalEPA]. Using just your smartphone camera makes searching an item more accessible than using current search tools [Waste Wizard]. This technology can also be deployed in trash cans to provide suggestions to the user [P Tiyajamorn], or to measure waterway plastic reduction efforts [T Hale].

![image](https://user-images.githubusercontent.com/39971693/232918769-e3da0eeb-07ef-4e7b-abe6-251e3450ef8a.png)

## Installation

Python Version: 3.9

Run ```python3 -m pip install -r requirements.txt```

## How to run the server
# Setup Local Copy

Make sure you have Node.js 8+ and yarn installed.

Change to react-website directory and run following commands:

1. `yarn install` (may also need to run `yarn add tabler-react tabler-icons-react`)

# Running

1. `yarn start` to run on http://localhost:3000/

2. start model_inference_app.py in separate terminal to test submitting images

# Building

1. `yarn build` generates a build folder in the current directory

# Testing
```cd project\front-end_apps\website\react-website\```
```yarn test``` runs the test for the this directory


## Testing
#### Front End
```
cd project/front-end-apps
pytest
```
![image](https://user-images.githubusercontent.com/39971693/232919996-99350a1e-305a-4129-ae45-52511b1debb3.png)
#### Backend
```
cd project/backend
pytest
```
![image](https://user-images.githubusercontent.com/39971693/232920270-739e8288-9244-4b6f-b444-d81e917d336e.png)

## Development Instructions

### Environment
We recommend using Anaconda to create a Python environment with all the requirements outlined in the requirements.txt file. If you're not familiar with Anaconda, please see the official website or refer to this cheat sheet for the proper commands.

### Code Editor
We recommend using Visual Studio Code (VS Code) with the Python Language and running Jupyter Notebook for this project.

### OS/System Independent
The front-end and backend work on Windows, Mac, and Linux operating systems. However, to run the backend smoothly, we recommend your system to have at least 8GB of RAM and for fast response, a Nvidia GPU with at least 8GB of VRAM.

Please make sure that your system meets these requirements before proceeding with the setup.

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

Bryan Burch
⁢

Daniel Smagly
⁢

Santiago Bermudez

![image](https://user-images.githubusercontent.com/39971693/232920553-e0294477-c6a2-4bb7-aa7c-44de416e9c5a.png)

