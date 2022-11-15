![image](https://user-images.githubusercontent.com/39971693/200933337-ff2dfdd5-7323-474b-9fe8-197513f16cdd.png)

## Introduction

### Statement of Problem

Over the past decade global municipal solid waste (MSW) has nearly doubled, and is expected to continue increasing at this rate. In 2002 we produced 0.68 billion tonnes of MSW, which is expected to increase to 2.2 billion by 2025, tripling in about 20 years. This will increase total costs to $375 billion, which is heavily paid for at the city level [D Hoornweg]. Even more importantly, post-consumer waste accounts for 1,460 mtCO2e of our collective green-house gas (GHG) emissions. This is primarily from methane that is produced when food decomposes anaerobically, but this can be avoided if it’s sorted and composted [D Hoornweg]. To accomplish this, in 2022 California passed SB 1383 mandating counties to provide this service for residents across the state [CalRecycle]. However, as recently as 2016, California’s landfills were still made up of 40% organic waste, and 30% recycled material [CalEPA]. Recovering improperly sorted recyclables adds additional cost and complexity, so source separation should be prioritized [D Hoornweg].
  
### Proposed solution

The World Bank’s 2012 study of MSW reported that the most important way to reduce consumer greenhouse gas emissions was to educate consumers on best reduction and sorting practices [D Hoornweg]. To help encourage Californian’s to properly separate recyclables and now organic materials, we will use a neural network based program to suggest which bin an item belongs to. We expect the precision [P Tiyajamorn] to be better than current Califorian sorting rates [CalEPA], and using just your smartphone camera makes searching an item more accessible than using current search tools [Waste Wizard]. This technology can also be deployed in trash cans to provide suggestions to the user [P Tiyajamorn], or to measure waterway plastic reduction efforts [T Hale].

### Scope and Limitations

Previous efforts leveraging computer vision for automated trash sorted have failed to scale their datasets to the size needed to teach the algorithm a properly generalized understanding of trash [NV Kumsetty]. Because different municipalities sort waste in different ways, previous researchers continually make new specialized datasets rather than adding to someone else’s. Combined, 17 open datasets contained well over 200,000 images, but using many different sources adds a great deal of complexity. Continuing this trend will prevent meaningful deployment of this technology at scale. In contrast, this study seeks to outline best known methods for organizing volunteer events for image collection, providing freely available pre-trained models, and allowing inference to be done using an API. Having the widest array of images will increase the accuracy and generalization of the model, but only if datasets are scaled rather than dispersed. 

![classification](https://user-images.githubusercontent.com/39971693/196785253-71e2d4eb-f1bc-48a3-8c18-ab56ff5fcfe3.png)

[Fig 1] Classification [TACO]

![Screenshot 2022-10-19 121823](https://user-images.githubusercontent.com/39971693/196784358-e82e143e-fda5-4afe-9b0d-6ec584dbfd96.png)

[Fig 2] Object Detetion[TACO]

![image](https://user-images.githubusercontent.com/39971693/196784748-452d9202-69b0-4d2c-b14a-debbaefeaf6c.png)

[Fig 3] Instance segmentation [TACO]

There are three primary levels of computer vision we can use with increasing accuracy: classification [Fig 1], object detection [Fig 2], and instance segmentation [Fig 3]. Classification is simplest, returning one predicted class per image, but it can’t differentiate between two objects in the same picture. Object detection takes this one step further, by predicting a class and an associated bounding box around it. This makes annotating training data more complex, but allows all pictures to be used regardless of how many objects it has. This study will be annotating and training data using object detection, but there’s another level of detail that can be obtained using instance segmentation. This predicts pixel by pixel masks outlining the exact shape of the object, this could enable automated sorting to be done by a machine [TACO]. However, the problem of dataset scaling has to be overcome first, and using bounding boxes instead of masks will simplify this process. A future proof dataset should support all three types of annotations, so that it can be deployed for multiple use cases.  


### References

[SP Gundupalli] [A review on automated sorting of source-separated municipal solid waste for recycling](https://www.academia.edu/29489023/A_review_on_automated_sorting_of_source_separated_municipal_solid_waste_for_recycling?auto=citations&from=cover_page)

[P Tiyajamorn] [Automatic Trash Classification using Convolutional Neural Network Machine Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9095775)

[C Liu] [A Domestic Trash Detection Model Based on Improved YOLOX](https://www.mdpi.com/1424-8220/22/18/6974/htm)

[NV Kumsetty] [TrashBox: Trash Detection and Classification using Quantum Transfer Learning](https://ieeexplore.ieee.org/abstract/document/9770922)

[D Hoornweg] [What a Waste : A Global Review of Solid Waste Management](https://openknowledge.worldbank.org/handle/10986/17388)

[CalEPA] [Enhancing Organic Materials Management by Improving Coordination, Increasing Incentives & Expediting Decision-Making](https://calepa.ca.gov/wp-content/uploads/sites/6/2018/11/CalEPA-Report-Enhancing-Organic-Materials-Management.pdf)

[T Hale] [Field Testing Report for the Statewide Trash Monitoring Methods Project](https://www.sfei.org/sites/default/files/biblio_files/Field%20Testing%20Report%20-%20Trash%20Monitoring%20Methods%202021%20rev1.pdf)

[CalRecycle] [New Statewide Mandatory Organic Waste Collection](https://calrecycle.ca.gov/organics/slcp/collection/)

[Waste Wizard] [Sacramento Waste Wizard](http://www.cityofsacramento.org/public-works/rsw/waste-wizard)

[TACO] Trash Annotations in Context


