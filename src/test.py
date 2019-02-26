
import argparse
import cv2
from keras_yolo3.yolo import YOLO
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from PIL import Image
from timeit import default_timer as timer

from logos import detect_logo, match_logo
from similarity import load_brands_compute_cutoffs
from utils import load_extractor_model, load_features, model_flavor_from_name, parse_input
import utils

sim_threshold = 0.95
output_txt = 'out.txt'


def test(filename):
    """
    Test function: runs pipeline for a small set of input images and input
    brands.
    """
    yolo = YOLO(**{"model_path": 'keras_yolo3/yolo_weights_logos.h5',
                "anchors_path": 'keras_yolo3/model_data/yolo_anchors.txt',
                "classes_path": 'data_classes.txt',
                "score" : 0.05,
                "gpu_num" : 1,
                "model_image_size" : (416, 416),
                }
               )
    save_img_logo, save_img_match = True, True

    test_dir = os.path.join(os.path.dirname(__file__), os.path.pardir, 'data/test')

    # get Inception/VGG16 model and flavor from filename
    model_name, flavor = model_flavor_from_name(filename)
    ## load pre-processed features database
    features, brand_map, input_shape = load_features(filename)

    ## load inception model
    model, preprocess_input, input_shape = load_extractor_model(model_name, flavor)
    my_preprocess = lambda x: preprocess_input(utils.pad_image(x, input_shape))

    ## load sample images of logos to test against
    input_paths = ['test_batman.jpg', 'test_robin.png', 'test_lexus.png', 'test_champions.jpg',
                   'test_duff.jpg', 'test_underarmour.jpg', 'test_golden_state.jpg']
    input_labels = [ s.split('test_')[-1].split('.')[0] for s in input_paths]
    input_paths = [os.path.join(test_dir, 'test_brands/', p) for p in input_paths]

    # compute cosine similarity between input brand images and all LogosInTheWild logos
    ( img_input, feat_input, sim_cutoff, (bins, cdf_list)
    ) = load_brands_compute_cutoffs(input_paths, (model, my_preprocess), features, sim_threshold, timing=True)

    images = [ p for p in os.listdir(os.path.join(test_dir, 'sample_in/')) if p.endswith('.jpg')]
    images_path = [ os.path.join(test_dir, 'sample_in/',p) for p in images]

    start = timer()
    times_list = []
    img_size_list = []
    candidate_len_list = []
    for i, img_path in enumerate(images_path):
        outtxt = img_path

        ## find candidate logos in image
        prediction, image = detect_logo(yolo, img_path, save_img = True,
                                          save_img_path = test_dir, postfix='_logo')

        ## match candidate logos to input
        outtxt, times = match_logo(image, prediction, (model, my_preprocess),
                outtxt, (feat_input, sim_cutoff, bins, cdf_list, input_labels),
                save_img = save_img_match, save_img_path=test_dir, timing=True)

        img_size_list.append(np.sqrt(np.prod(image.size)))
        candidate_len_list.append(len(prediction))
        times_list.append(times)

    end = timer()
    print('Processed {} images in {:.1f}sec - {:.1f}FPS'.format(
            len(images_path), end-start, len(images_path)/(end-start)
           ))

    fig, axes = plt.subplots(1,2, figsize=(9,4))
    for iax in range(2):
        for i in range(len(times_list[0])):
            axes[iax].scatter([candidate_len_list, img_size_list][iax], np.array(times_list)[:,i])

        axes[iax].legend(['read img','get box','get features','match','draw','save'])
        axes[iax].set(xlabel=['number of candidates', 'image size'][iax], ylabel='Time [sec]')
    plt.savefig(os.path.join(test_dir, 'timing_test.png'))


if __name__ == '__main__':
    filename = 'inception_logo_features_200_trunc2.hdf5'
    test(filename)
