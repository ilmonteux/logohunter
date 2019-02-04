# Insight AI project: Logo detection as a service

Machine learning project developed at Insight Data Science, 2019 AI session.

## Project description
Companies and advertisers need to know their customers to assess their business and marketing strategies. While the amount of information shared on social media platforms is gargantuan, a lot of it is unstructured and untagged, and particularly so for visual data. Users can voluntarily tag their preferred brands in their posts, but wouldn't it be much better for the companies to know every time their brand is being publicly shared?

In this project, I built a general-purpose logo detection API. To avoid re-training the network for each new company using the service, logo detection and identification are split in two logically and operationally separate parts: first, we find all logos in the image with a YOLO detector, and then we check for similarity between the proposed logos and an input uploaded by the customer (for example, the company owning the logo), by computing cosine similarity between features extracted by a pre-trained Inception network.

![IMAGE COMING SOON](AAAAA)

## Repo structure
+ `build`: scripts to build environment
+ `configs`: configuration files
+ `data`: input data
+ `notebooks`: exploratory analysis, visualization
+ `src`: source code
+ `tests`: unit tests

## Getting Started

#### Data
This project uses the [Logos In The Wild dataset](https://www.iosb.fraunhofer.de/servlet/is/78045/) which can be requested via email directly from the authors of the paper, [https://arxiv.org/abs/1710.10891](arXiv:1710.10891). This dataset includes 11,054 images with 32,850 bounding boxes for a total of 871 brands.

The dataset is licensed under the CC-by-SA 4.0 license. The images themselves were crawled from Google Images and are property of their respective copyright owners. For legal reasons, we do not provide the raw images: while this project would fall in the "fair use" category, any commercial application would likely need to generate their own dataset.

#### Requisites

#### Installation

## Usage

## Build Environment

#### clone, setup conda environment

#### Optional: download, process and clean dataset
After contacting the authors of [https://arxiv.org/abs/1710.10891](arXiv:1710.10891) you will receive the download link to the [Logos In The Wild Dataset](https://www.iosb.fraunhofer.de/servlet/is/78045/), which is a zip file. Place it in the `data/` directory, unzip it and your directory tree should be the following
```
REPO
├── src/
│   
├── data/
     ├── LogosInTheWild-v2.zip
     └── LogosInTheWild-v2/
            ├── data/
            │     ├── 0samples/
            │     ├── abus/
            │     ├── accenture/
            │     ├── adidas/
            │     ....
            │
            ├── licence.txt
            ├── Readme.txt
            └── scripts/
```
Each directory contains a `urls.txt` file with links to the images themselves, and one XML file with object annotations for each link (some of the URLs are no longer online since the dataset was created).

To download the images and clean up the dataset run the following:
```
cd src/
# download images from their respective URLs
python fetch_LogosInTheWild.py  --path ../data/LogosInTheWild-v2/
# clean dataset and generate cutouts of each annotated brand
python create_clean_dataset.py  --in ../data/LogosInTheWild-v2/data/ --out ../data/LogosInTheWild-v2/data_cleaned --roi
# read XML annotations and transfer to a text file in the keras-yolo3 format
python litw_annotation.py --in ../data/LogosInTheWild-v2/data_cleaned/voc_format
```
The resulting folder structure is:

```
── LogosInTheWild-v2/
       ├── data/
       │     ├── 0samples/
       │     ├── abus/
       │     ....
       │
       ├── data_cleaned/
       │     ├── brandROIs/
       │         ├── 1fcköln/
       │         ├── 24fitness/
       │         ....
       │     │
       │     ├── brands.txt
       │     └── voc_format/
       │         ├── 0samples/
       │         ├── abus/
       │         ....
       │
       ├── licence.txt
       ├── Readme.txt
       └── scripts/
```
The images and XML annotations have been copied to `data_cleaned/voc_format/`. The file `brands.txt` contains a list of all 788 brands and brand variants present in the annotations (for example, the PUMA symbol vs the PUMA text logo count as different logos). The `--roi` option in the second command extracts the logos from each image and saves them into the `data_cleaned/brandROIs/` directory.

Finally, the last command reads the XML files and generates  `data_train.txt` and `data_test.txt` files, with on each line the path to the images and the bounding box specification for each object in the image:
```
path-to-file1.jpg xmin,ymin,xmax,ymax,class_id xmin,ymin,xmax,ymax,class_id
path-to-file2.jpg xmin,ymin,xmax,ymax,class_id
```
By default, `class_id=0` for all logos, as we train the network to learn all possible logos.

#### Optional: train object detection model
After the previous steps, the `data_train.txt` and `data_test.txt` files have all the info necessary to train the model. To train the model, we follow the instructions of the [keras-yolo3](https://github.com/qqwweee/keras-yolo3) package: first we download pre-trained YOLO weights from the YOLO official website, and then we convert them to the HDF5 format used by keras.
```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
```
Training detail such as paths to train/text files, log directory, number of epochs, learning rates and so on are specified in `train.py`.
```
python train.py
```
The training is performed in two runs, first with all the layers except the last three frozen, and then with all layers trainable.
