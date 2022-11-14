## Overview

Computer vision using artificial neural networks has propagated throughout our culture and has begun having an array of technical implementations that make our lives easier. However, in order to make use of this technology within the field of trash management, a database of images will be needed to accurately train it. It's estimated that we will need over 1,000 images per class [1], and with over 100 individual classes outlined in the Sacramento Waste Wizard, the total number of images needed quickly rises into the 100,000s. Unfortunetly, almost all researchers in this space have elected to create their own datasets instead of building off of someone elses. This has caused a great amount of fragementation, a review of current trash image datsets has shwon that over 158,000 publically available images are spread accross 20 different datasets [3]. But because they aren't considated, using them is extremely difficult and has held back the future prospects of using neural networks for trash recognition. To fix this we are going to both, combine currently existsing datasets, and utilize volunteer clean up hours to collect new data. You can be a part of this data collection process and help train new computer vision algorithms, just follow the steps below.

## How to participate

1. Sign waiver

If you are over 18, ensure that you sign [this form](https://docs.google.com/forms/d/e/1FAIpQLSdCMGCegU_LGRD8PWMNdHLlymxKUoUU8md7ebfKLOxcx2ySaw/viewform?usp=sf_link), there will also be physical copies available during the cleanup event if you'd like to sign up there. If you are under 18 we cannot use your images but appreciate your help with volunteering. This waiver ensures that you have given us permission to use your photos and that you won't upload any personally identifiable information or anything that would require us to throw out the pictures without using them.

2. Turn on location services

Ensure location services are enabled on your phone, a big part of this project is trying to understand the geographic layout of where trash accumulates. This is usually on by default, but it's good to check before data collection.

<details><summary>iOS</summary>
  
  ![image](https://user-images.githubusercontent.com/39971693/201779532-020a6154-f9e7-4caa-b25f-64c4d0ff536d.png)
</details>

<details><summary>Android</summary>
  
  [More Android devices](https://support.google.com/photos/answer/9921876?hl=en)
  
![image](https://user-images.githubusercontent.com/39971693/201780171-f964895f-cb20-47e7-a5f7-9c33093a5b01.png)

</details>

2. Take the photo

Complete your clean up activities as your normally would while taking pictures. Make sure you are as close as possible to the item, you should be able to pick it up after taking the photo without having to reposition yourself. Next, ensure that no one's face is in the photo, this counts are PPI and needs much stronger data protections then what is necessary for this project. Remember, you don't have to take pictures of everything, if an object is hard to see or there's too many in one spot to take each photo just skip past them and prioritize removing them rather than taking photos.

Good examples:

Bad examples:

3. Submission

Upload your images to [this Google Drive folder](https://drive.google.com/drive/folders/1EbfJxHWg2oZslDjLVXYaTJ5rq1mIWdNM?usp=sharing), ensuring that you are following the guidelines above. 


If you are interested in following this project you can star this GitHub repository. Additionally feel free to reach out to me at julianofhernandez@gmail.com for more details or for more opportunities to help out. Thank you for helping to collect our dataset and remove litter from your community, keep being awesome!


## References

[1] How many images do you need to train a neural network? https://petewarden.com/2017/12/14/how-many-images-do-you-need-to-train-a-neural-network/

[2] Sacramento Waste Wizard http://www.cityofsacramento.org/public-works/rsw/waste-wizard#!rc-cpage=wizard_material_list

[3] Waste Datasets Review https://github.com/AgaMiko/waste-datasets-review
