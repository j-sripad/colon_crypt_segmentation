# colon crypt segmentation 
- **2 - Image Segmentation Task**


**Deep Learning Framework used** : Pytorch

This repository contains pytorch codes to do colon crypt segmentation.

Repository struture
```
models/
overlays/
Datapreparation.ipynb
Inference.ipynb
Mask_from_RLE.ipynb
Training.ipynb
Viz.ipynb
utils.py
patch_extractor.py 
requirements.txt
submission.csv
.>
.>
.>
```

**Data Description:**

**Dataset** - https://drive.google.com/file/d/1RHQjRav-Kw1CWT30iLJMDQ2hPMyIwAUi/view?usp=sharing

After downloading the data and extracting the files in the above directory structure, it has the following directory structure

```
Colonic_crypt_dataset

     .  train/
     .  train_mask/
     .  test/
     .  test_mask/
     .  train.csv
     .  test.csv
     .  colon-dataset_information.csv

```
**train/** has training H & E images and the annotation jsons
**test/** has test H & E images and the annotation jsons
**train_mask/** has training mask for all the images
**test_mask/** has training mask for all the images
**train.csv** has training image ids and their corresponding Run-Length Annotations
**test,csv** has test image ids and their corresponding Run-Length Annotations.

**Data Preprocessing**

- Mask Generation

    - The dataset already has masks generated for training and test sets. But if required, use **Mask_from_RLE.ipynb** notebook to generate masks from RLR using the train and test csv files


