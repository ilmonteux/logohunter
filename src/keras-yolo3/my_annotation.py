import xml.etree.ElementTree as ET
import os
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Process image annotation from VOC style to keras-yolo3 style.')
parser.add_argument('-im_path', type=str, default='../../data_phishtank/annotated', 
                    help='path to directory containing images and xml annotations')
parser.add_argument('-train_test_split', type=float, default = 0.3, 
                    help='fraction of dataset set apart for test')

args = parser.parse_args()
path_to_images, train_test_split = args.im_path, args.train_test_split

# path_to_images = '../../data_phishtank/annotated'
# train_test_split = 0.3


classes = [ "Paypal", "Facebook", "Dropbox", "Netflix", "Microsoft" ]

def convert_annotation(image_id, list_file):
    in_file = open(os.path.join(path_to_images,'%s.xml'%(image_id)))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = os.getcwd()


all_images = os.listdir(path_to_images)
np.random.seed(100)
np.random.shuffle(all_images)


train_file = open('data_train.txt', 'w')
test_file = open('data_test.txt', 'w')
for file in os.listdir(path_to_images):
    image_id, ext = os.path.splitext(file)
    if ext != '.jpg':
        continue


    if not os.path.exists(os.path.join(path_to_images, image_id+'.xml')):
        continue

    rand = np.random.rand()
    list_file = train_file  if rand > train_test_split else test_file

    list_file.write(os.path.normpath(os.path.join(wd, path_to_images, image_id+'.jpg')))
    
    convert_annotation(image_id, list_file)
    
    list_file.write('\n')
    
train_file.close()
test_file.close()
