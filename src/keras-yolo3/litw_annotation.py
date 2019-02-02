import xml.etree.ElementTree as ET
import os
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Process image annotation from VOC style to keras-yolo3 style.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-im_path', type=str, default='../../data_litw/LogosInTheWild-v2/data_cleaned/voc_format',
                    help='path to directory containing images and xml annotations')
parser.add_argument('-classes_names', type=str, default='../../data_litw/LogosInTheWild-v2/data_cleaned/brands.txt',
                    help='path to txt file listing all possible classes')
parser.add_argument('-train_test_split', type=float, default = 0.3,
                    help='fraction of dataset set apart for test')
parser.add_argument('-split_class_or_file', type=bool, default = True,
                    help='Do train/test split at class (0) or file level (1)')
parser.add_argument('-openset', type=bool, default = True,
                    help='If openset=True, annotate logo objects as one class instead of class by class')
args = parser.parse_args()

path_to_images, path_to_classes  = args.im_path, args.classes_names
train_test_split, split_class_or_file, flag_open = args.train_test_split, args.split_class_or_file, args.openset

with open(path_to_classes, 'r') as f:
    classes = [ line.replace('\n','') for line in f.readlines()]

def convert_annotation(xml_file):
    """Extract VOC-style annotations from XML input fileself.

    Returns:
      list of [int, bbox], one for each object, where int is the class id and
      bbox=(xmin,ymin,xmax,ymax) is the bounding box specification.


    """
    with open(xml_file, 'r') as in_file:
        tree = ET.parse(in_file)
        root = tree.getroot()

        output = []
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult)==1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            bboxes = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
            #list_file.write(" " + ",".join([str(a) for a in bboxes]) + ',' + str(cls_id))
            output.append([cls_id, bboxes])
    return output

wd = os.getcwd()

with open('my_classes.txt', 'w') as classes_file:
    if flag_open:
        classes_file.write('logo\n')
    else:
        for c in classes:
            classes_file.write('{}\n'.format(c))

train_file = open('data_train.txt', 'w')
test_file = open('data_test.txt', 'w')

for folder in os.listdir(path_to_images):
    folder_path = os.path.abspath(os.path.join(path_to_images, folder))
    if ' ' in folder_path:
        os.rename(folder_path, folder_path.replace(' ',''))
        folder_path = folder_path.replace(' ','')
    if not os.path.isdir(folder_path):
        continue

    if split_class_or_file == 0: rand = np.random.rand()
    for file in os.listdir(folder_path):
        filename = os.path.splitext(file)[0]
        if (not file.endswith('.xml')) or (not os.path.exists(os.path.join(folder_path,filename+'.jpg'))):
            continue

        if split_class_or_file == 1: rand = np.random.rand()

        list_file = train_file  if rand > train_test_split else test_file

        list_file.write(os.path.normpath(os.path.join(folder_path, filename+'.jpg')))

        for cls_id, bboxes in convert_annotation(os.path.join(folder_path, file)):
            # squash all logos to one class
            if flag_open:
                cls_id = 0

            list_file.write(" " + ",".join([str(a) for a in bboxes]) + ',' + str(cls_id))

        list_file.write('\n')

train_file.close()
test_file.close()
