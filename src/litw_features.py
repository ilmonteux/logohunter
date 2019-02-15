# import matplotlib.pyplot as plt
import cv2
import h5py
import matplotlib as mpl
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import metrics
import utils

def extract_litw_features(filename, model, my_preprocess):
    """
    Given Logos in The Wild dataset, extract all logos from images and exract
    features by applying truncated InceptionV3 model.
    """
    img_list_lbl, bbox_list_lbl = metrics.read_txt_file(filename)

    with open('brands.txt') as f:
        all_classes = [ s.strip('\n') for s in f.readlines() ]

    all_logos = []
    brand_map = []
    for idx in range(len(bbox_list_lbl)):

        im = cv2.imread(img_list_lbl[idx])[:,:,::-1]

        for bb in bbox_list_lbl[idx]:
            if bb[3]-bb[1] < 10 or bb[2]-bb[1] < 10 or bb[3]>im.shape[0] or bb[2]> im.shape[0]:
                continue
            all_logos.append(im[bb[1]:bb[3], bb[0]:bb[2]])
            brand_map.append(bb[-1])

    features = utils.features_from_image(all_logos, model, my_preprocess)

    return features, all_logos, brand_map

if __name__ == '__main__':
    model, my_preprocess = utils.load_extractor_model()

    features, all_logos, brand_map = extract_litw_features('data_all_train.txt', model, my_preprocess)

    print('Processed {} logos, transformed into feature vectors'.format(len(features)))

    utils.save_features('inception_logo_features.hdf5', features, brand_map)
