# Setup for Development

## Setup Local Copy

Make sure you have Node.js 8+ and yarn installed.

Change to react-website directory and run following commands:

1. `yarn install` (may also need to run `yarn add tabler-react tabler-icons-react`)

## Running

1. `yarn start` to run on http://localhost:3000/

2. start model_inference_app.py in separate terminal to test submitting images

## Testing

1. 'yarn test' runs the test for the current directory


# Setup for Production 

1. If you haven't yarn install the dependencies do the `yarn install` in the directory `Trash-Sorting\project\front-end_apps\website\react-website`

2. `yarn build` generates a build folder in the current directory

3. Move the `build` folder to directory `Trash-Sorting\project\backend\website_backend`

4. In the terminal, go to the URL `http://localhost:8000`, this should start the website correctly