[LogoImage]()
## Statement of Problem

Over the past decade global municipal solid waste (MSW) has nearly doubled, and is expected to continue increasing at this rate. In 2002 we produced 0.68 billion tonnes of MSW, which is expected to increase to 2.2 billion by 2025, tripling in about 20 years. This will increase total costs to $375 billion, which is heavily paid for at the city level [D Hoornweg]. Even more importantly, post-consumer waste accounts for 1,460 mtCO2e of our collective green-house gas (GHG) emissions. This is primarily from methane that is produced when food decomposes anaerobically, but this can be avoided if it’s sorted and composted [D Hoornweg]. To accomplish this, in 2022 California passed SB 1383 mandating counties to provide this service for residents across the state [CalRecycle]. However, as recently as 2016, California’s landfills were still made up of 40% organic waste, and 30% recycled material [CalEPA]. Recovering improperly sorted recyclables adds additional cost and complexity, so source separation should be prioritized [D Hoornweg].
  
## Proposed solution

The World Bank’s 2012 study of MSW reported that the most important way to reduce consumer greenhouse gas emissions was to educate consumers on best reduction and sorting practices [D Hoornweg]. To help encourage Californian’s to properly separate recyclables and now organic materials, we will use a neural network based program to suggest which bin an item belongs to. We expect the precision [P Tiyajamorn] to be better than current Califorian sorting rates [CalEPA], and using just your smartphone camera makes searching an item more accessible than using current search tools [Waste Wizard]. This technology can also be deployed in trash cans to provide suggestions to the user [P Tiyajamorn], or to measure waterway plastic reduction efforts [T Hale].

## Installation

Python Version: 3.9

Run ```python3 -m pip install requirements.txt```

## Start training 
Navigate to project/model/pytorch-ssd/

Run ```python3 train.py``` this will install the dataset which may take a while (1500 images).

usage: train.py [-h] [-b BATCH_SIZE] [-i ITERATIONS] [-j WORKERS] [-pf PRINT_FREQUENCY] [-lr LEARNING_RATE] [-ckpt CHECKPOINT] [-d DATA_DIR]

## Run validation
Navigate to project/model/pytorch-ssd/

Run ```python3 validation.py```

## Database Design ERD
```json
[
{"file_path": "TACO\\data\\batch_1/000006.jpg", "Original category": "Glass bottle", "New category": "Glass bottles & jars", "x": 517.0, "y": 127.0, "width": 447.0, "height": 1322.0}, 
{"file_path": "TACO\\data\\batch_1/000008.jpg", "Original category": "Meal carton", "New category": "Paper egg carton", "x": 1.0, "y": 457.0, "width": 1429.0, "height": 1519.0}, 
{"file_path": "TACO\\data\\batch_1/000008.jpg", "Original category": "Other carton", "New category": "Paper egg carton", "x": 531.0, "y": 292.0, "width": 1006.0, "height": 672.0}, 
{"file_path": "TACO\\data\\batch_1/000010.jpg", "Original category": "Clear plastic bottle", "New category": "Clear plastic \"clam shell\" containers", "x": 632.0, "y": 987.0, "width": 500.0, "height": 374.0}
]
```

## Prototype

## Testing
TODO: Spring 2023

## Deployement
TODO: Spring 2023

## Developer Instructions
TODO: Spring 2023

## Contributors
![image](https://user-images.githubusercontent.com/39971693/200933337-ff2dfdd5-7323-474b-9fe8-197513f16cdd.png)

## License
