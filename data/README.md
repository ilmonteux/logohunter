# Datasets

## Logos In The Wild Dataset: first-time setup

The original `Logos In The Wild` dataset was developed by the authors of [arXiv:1710.10891](https://arxiv.org/abs/1710.10891). The dataset has been uploaded to Zenodo, https://zenodo.org/record/5101018. 

> Originally directions were to ask the original authors for the dataset. As of December 2021, the original website (https://www.iosb.fraunhofer.de/servlet/is/78045/) returns a 404 not found, and people have reported not receiving an answer from the original authors. I have therefore uploaded the dataset to the zenodo link above.

- Download both `LogosInTheWild-v2.zip` (the original dataset, with images URLs and bounding box annotations) and `litw_cleaned.tar.gz` (the pre-processed images). 
- Place `LogosInTheWild-v2.zip` in the `data/` directory, unzip it 
- Place `litw_cleaned.tar.gz` in the newly created directory `data/LogosInTheWild-v2/`. Un-archive this file too.

Your directory structure should be the following:
```
REPO
├── src/  
├── data/
     ├── LogosInTheWild-v2.zip
     └── LogosInTheWild-v2/
            ├── data/
            │     ├── 0samples/
            │     │     ├── img000001.jpg
            │     │     ├── img000001.xml
            │     │     ....            
            │     ├── abus/
            │     │     ├── img000000.xml
            │     │     ├── img000001.xml
            │     │     ....
            │     │     └── urls.txt
            │     ....
            ├── data_cleaned/
            │     ├── brandROIs/
            │     │     ....
            │     ├── brands.txt
            │     └── voc_format/
            │           ├── 0samples/
            │           │     ├── img000001_1.jpg
            │           │     ├── img000001_1.xml
            │           │     ....
            │           ├── abus/
            │           │     ├── img000000_2.jpg
            │           │     ├── img000000_2.xml
            │           │     ├── img000001_2.jpg
            │           │     ├── img000001_2.xml
            │           │     ....
            │           │     └── urls.txt
            │           ....            
            ├── litw_cleaned.tar.gz
            ├── licence.txt
            ├── Readme.txt
            └── scripts/
```

Quick explanation about the folder structure:
- `data/LogosInTheWild-v2/data` contains the original dataset. Each directory has a `urls.txt` file with links to the images themselves, and one XML file with object annotations for each link (note that some of the URLs are no longer online since the dataset was created). If one retrieved the images from the URLs using the provided python script (`src/fetch_LogosInTheWild.py`), the images would be downloaded here.
- `data/LogosInTheWild-v2/data_cleaned` contains the cleaned-up dataset (created from the above via `src/create_clean_dataset.py`), with bounding box annotations in the VOC format in the `voc_format` folder. The file `brands.txt` contains a list of all 788 brands and brand variants present in the annotations (for example, the PUMA symbol vs the PUMA text logo count as different logos). Additionally, the `brandROI` folder has the logos extracted for each brand (this is not used in this project, but can be useful to train a classifier to learn different logos).

Finally, run this command:
```
# read XML annotations and transfer to a text file in the keras-yolo3 format
python litw_annotation.py --in ../data/LogosInTheWild-v2/data_cleaned/voc_format
```
This reads all the XML files and generates the `data_train.txt` and `data_test.txt` files in the format expected by `keras-yolo3`, with on each line the path to the images and the bounding box specification for each object in the image:
```
path-to-file1.jpg xmin,ymin,xmax,ymax,class_id xmin,ymin,xmax,ymax,class_id
path-to-file2.jpg xmin,ymin,xmax,ymax,class_id
```
By default, we have `class_id=0` for all logos, as we train the network to learn the general concept of a logo for open-set object detection. Specifying the option `-closedset` would keep all the classes. The default test/train split is 30% kept aside for testing.


### [DEPRECATED] dataset retrieval from original URLs
[DEPRECATED] ~~After contacting the authors of [arXiv:1710.10891](https://arxiv.org/abs/1710.10891) you will receive a download link to the [Logos In The Wild Dataset](https://www.iosb.fraunhofer.de/servlet/is/78045/), which is a zip file.~~

Download `LogosInTheWild-v2.zip` from https://zenodo.org/record/5101018 and place it in this directory (`data/`), unzip it. Each directory in `data/LogosInTheWild-v2//data` contains a `urls.txt` file with links to the images themselves, and one XML file with object annotations for each link (note that some of the URLs are no longer online since the dataset was created).

To download the images and clean up the dataset run the following:
```
cd src/
# download images from their respective URLs
python fetch_LogosInTheWild.py  --dir_litw ../data/LogosInTheWild-v2/data
# clean dataset and generate cutouts of each annotated brand
python create_clean_dataset.py  --in ../data/LogosInTheWild-v2/data/ --out ../data/LogosInTheWild-v2/data_cleaned --roi
# read XML annotations and transfer to a text file in the keras-yolo3 format
python litw_annotation.py --in ../data/LogosInTheWild-v2/data_cleaned/voc_format
```
The resulting folder structure should be the same as detailed above.

### Next steps

Now that we have the dataset, we can train the neural network. Head back to [the main directory](/)

## Test dataset
This directory contains images to test your installation and run the inference on a sample. The structure is the following:
```
data/
├── goldenstate/
├── lexus/
├── sample_in/  
└── test_brands/
```
The `goldenstate` and `lexus` folders contain a sample of images downloaded from Google Images (and are respective property of the companies, used here only for testing purposes), namely 68 and 54 images. `sample_in` contains a sample of 20 images with various logos in them, while `test_brands` contains a selection of 17 brands, most of which are new to the model (not part of the Logos In The Wild Dataset).
