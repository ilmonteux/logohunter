# Dataset

## Testing dataset
This directory contains images to test your installation and run the inference on a sample. The structure is the following:
```
data/
├── goldenstate/
├── lexus/
├── sample_in/  
└── test_brands/
```
The `goldenstate` and `lexus` folders contain a sample of images downloaded from Google Images (and are respective property of the companies, used here only for testing purposes), namely 68 and 54 images. `sample_in` contains a sample of 20 images with various logos in them, while `test_brands` contains a selection of 17 brands, most of which are new to the model (not part of the Logos In The Wild Dataset).

## Logos In The Wild Dataset: first-time setup

After contacting the authors of [arXiv:1710.10891](https://arxiv.org/abs/1710.10891) you will receive a download link to the [Logos In The Wild Dataset](https://www.iosb.fraunhofer.de/servlet/is/78045/), which is a zip file. Place it in this directory (`data/`), unzip it and your directory tree should be the following
```
REPO
├── src/  
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

Each directory contains a `urls.txt` file with links to the images themselves, and one XML file with object annotations for each link (note that some of the URLs are no longer online since the dataset was created).

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
       │     │    ├── 1fcköln/
       │     │    ├── 24fitness/
       │     │    ....
       │     │
       │     ├── brands.txt
       │     └── voc_format/
       │          ├── 0samples/
       │          ├── abus/
       │         ....
       │
       ├── licence.txt
       ├── Readme.txt
       └── scripts/
```
The images and XML annotations have been copied to `data_cleaned/voc_format/`. The file `brands.txt` contains a list of all 788 brands and brand variants present in the annotations (for example, the PUMA symbol vs the PUMA text logo count as different logos). The `--roi` option in the second command above extracts the logos from each image and saves them individually into the `data_cleaned/brandROIs/` directory.

Finally, the last command reads the XML files and generates  `data_train.txt` and `data_test.txt` files, with on each line the path to the images and the bounding box specification for each object in the image:
```
path-to-file1.jpg xmin,ymin,xmax,ymax,class_id xmin,ymin,xmax,ymax,class_id
path-to-file2.jpg xmin,ymin,xmax,ymax,class_id
```
By default, we have `class_id=0` for all logos, as we train the network to learn the general concept of a logo for open-set object detection. Specifying the option `-closedset` would keep all the classes. The default test/train split is 30% kept for testing.

## Next steps

Now that we have the dataset, we can train the neural network. Head back to [the main directory](/)
