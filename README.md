## Introduction

### Statement of Problem

Over the past decade global municipal solid waste has doubled, and is expected to continue to increase at this rate. In 2002 we produced 0.68 billion tonnes, which is expected to increase to 2.2 billion by 2025, tripling in about 20 years. This will increase total costs to $375 billion, which is heavily paid for at the city level [D Hoornweg]. Even more importantly, post-consumer waste accounts for 1,460 mtCO2e of our collective green-house gas (GHG) emissions, which is primarily made up of methane with 21 times more heating potential than carbon dioxide. Methane is produced when food decomposes anaerobically, but this can be avoided if it’s sorted and composted [D Hoornweg]. In 2022 California passed SB 1383 mandating counties to provide this service to residents across the state [CalRecycle]. However, as recently as 2016 California’s landfills were still made up of 40% organic waste, and 30% recycled material [CalEPA]. Additionally, recovering recyclables adds a lot of additional cost and complexity, so source separation should be prioritized [D Hoornweg].
  
### Proposed solution

The World Bank’s 2012 study of municipal solid waste (MSW) reported that the most important way to reduce consumer greenhouse gas emissions was to educate consumers on reduction and sorting [D Hoornweg]. To help encourage Californian’s to properly separate recyclables and now organic materials, we will use a neural network based program to suggest which bin an item belongs to. We expect the precision [P Tiyajamorn] to be better than current Califorian sorting rates [CalEPA], and using a smartphone just your smartphone camera makes searching an item more accessible than using current search tools [Waste Wizard]. This technology can also be deployed in trash cans to provide suggestions to the user [P Tiyajamorn], or to measure waterway plastic reduction efforts [T Hale].

### Scope and Limitations

Previous efforts leveraging computer vision for automated trash sorted have failed to scale their datasets to the size needed to teach the algorithm a properly generalized understanding of trash [NV Kumsetty]. Because different municipalities sort waste in different ways, previous researchers continually make new specialized datasets rather than adding to someone else’s. Combined, 17 open datasets contained well over 200,000 images, but using them together requires far too much overhead to realically do [NEEDED]. Continuing this trend will prevent meaningful deployment of this technology. In contrast, this study seeks to outline best known methods for organizing volunteer events to image collection, providing freely available pre-trained models, and allowing inference to be done using an API. Having the widest array of images will increase the accuracy and generalization of the model, but only if datasets are scaled rather than dispersed. 




[Fig1]					[Fig2]					[Fig3]

There are three primary levels of computer vision we can use with increasing accuracy: classification [Fig 1], object detection [fig 2], and instance segmentation [fig 3]. Classification is simplest, returning one predicted class per image, but it can’t differentiate between two objects in the same picture. Object detection takes this one step further, by predicting a class and an associated bounding box around it. This makes annotating training data more complex, but allows all pictures to be used regardless of how many objects it has. This study will be annotating and training data using object detection, but there’s another level of detail that can be obtained using instance segmentation. This predicts pixel by pixel masks outlining the exact shape of the object, this could enable automated sorting to be done by a machine [TACO]. To do this the problem of dataset scaling has to be overcome first, because annotating these images takes a lot more time than simply dragging a box around each piece of litter. 


References
[SP Gundupalli] A review on automated sorting of source-separated municipal solid waste for recycling
[P Tiyajamorn] Automatic Trash Classification using Convolutional Neural Network Machine Learning
[C Liu] A Domestic Trash Detection Model Based on Improved YOLOX
[NV Kumsetty] TrashBox: Trash Detection and Classification using Quantum Transfer Learning
[D Hoornweg] What a Waste : A Global Review of Solid Waste Management
[CalEPA] Enhancing Organic Materials Management by Improving Coordination, Increasing Incentives & Expediting Decision-Making
[T Hale] Field Testing Report for the Statewide Trash Monitoring Methods Project
[CalRecycle] New Statewide Mandatory Organic Waste Collection
[Waste Wizard] Sacramento Waste Wizard
[TACO] Trash Annotations in Context

