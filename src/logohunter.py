#
# MASTER SCRIPT TO DETECT LOGOS IN IMAGE AND FIND MATCHES
# for each image find and pass logos, compute features, see if logo passes cutoff, save image
#
import matplotlib.pyplot as plt

import sys, os
from PIL import Image
import cv2
import numpy as np

import argparse

from keras_yolo3.yolo import YOLO
from utils import contents_of_bbox, load_features, bbox_colors, parse_input, chain_preprocess, load_extractor_model
from similarity import features_from_image, similarity_cutoff, load_brands_compute_cutoffs, similar_matches, draw_matches

from timeit import default_timer as timer

input_shape = (299,299,3)
sim_threshold = 0.95
save_img_logo, save_img_match = True, True

def detect_logo(yolo, img_path, save_img = save_img_logo, save_img_path='./', postfix=''):
    """
    Call YOLO logo detector on input image, optionally save resulting image.

    Args:
      yolo: keras-yolo3 initialized YOLO instance
      img_path: path to image file
      save_img: bool to save annotated image
      save_img_path: path to directory where to save image
      postfix: string to add to filenames
    """
    try:
        image = Image.open(img_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
    except:
        print('File Open Error! Try again!')
        return None, None

    prediction, r_image = yolo.detect_image(image)

    img_out = postfix.join(os.path.splitext(os.path.basename(img_path)))
    if save_img:
        r_image.save(os.path.join(save_img_path, img_out))

    return prediction, r_image

def match_logo(img_path, prediction, model_preproc, input_features_cdf_cutoff_labels,
               save_img = save_img_match, save_img_path='./', timing=False):

    start = timer()
    model, my_preprocess = model_preproc
    feat_input, sim_cutoff, bins, cdf_list, input_labels = input_features_cdf_cutoff_labels
    img_test = cv2.imread(img_path)
    t_read = timer()-start
    candidates = contents_of_bbox(img_test, prediction)
    t_box = timer()-start
    features_cand = features_from_image(candidates, model, my_preprocess)
    t_feat = timer()-start
    matches, cos_sim = similar_matches(feat_input, features_cand, sim_cutoff, bins, cdf_list)
    t_match = timer()-start

    outtxt = img_path
    for idx in matches:
        bb = prediction[idx]
        label = input_labels[matches[idx][0]]
        print('Logo #{} - {} {} - classified as {} {:.2f}'.format(idx,
          tuple(bb[:2]), tuple(bb[2:4]), label, matches[idx][1]))

        outtxt += ' {},{},{},{},{},{:.2f},{:.3f}'.format(*bb[:4], label,bb[-1], matches[idx][1])
    outtxt += '\n'

    new_img = draw_matches(img_test, input_labels, prediction, matches)
    t_draw = timer()-start
    if save_img == True:
        save_img_path = os.path.abspath(save_img_path)
        a = cv2.imwrite(os.path.join(save_img_path, os.path.basename(img_path)), new_img)

    if timing:
        return outtxt, (t_read, t_box-t_read, t_feat-t_box, t_match-t_feat, t_draw-t_match)

    return outtxt


def test():
    yolo = YOLO(**{"model_path": 'keras_yolo3/yolo_weights_logos.h5',
                "anchors_path": 'keras_yolo3/model_data/yolo_anchors.txt',
                "classes_path": 'keras_yolo3/data_classes.txt',
                "score" : 0.1,
                "gpu_num" : 1,
                "model_image_size" : (416, 416),
                }
               )

    test_dir = os.path.join(os.path.dirname(__file__), os.path.pardir, 'data/test')

    ## load pre-processed features database
    filename = 'inception_logo_features.hdf5'
    brand_map, features = load_features(filename)

    ## load inception model
    model, preprocess_input = load_extractor_model()
    my_preprocess = lambda x: chain_preprocess(x, input_shape, preprocess_input)

    ## load sample images of logos to test against
    input_paths = ['test_batman.jpg', 'test_robin.png', 'test_lexus.png', 'test_champions.jpg',
                   'test_duff.jpg', 'test_underarmour.jpg', 'test_mustang.png', 'test_golden_state.jpg']
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

        ## find candidate logos in image
        prediction, r_image = detect_logo(yolo, img_path, save_img = save_img_logo,
                                          save_img_path = test_dir, postfix='_logo')

        ## match candidate logos to input
        outtxt, times = match_logo(img_path, prediction, (model, my_preprocess),
                (feat_input, sim_cutoff, bins, cdf_list, input_labels),
                save_img = save_img_match, save_img_path=test_dir, timing=True)

        img_size_list.append(np.sqrt(np.prod(r_image.size)))
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

        axes[iax].legend(['read img','get box','get features','match','draw'])
        axes[iax].set(xlabel=['number of candidates', 'image size'][iax], ylabel='Time [sec]')
    plt.savefig(os.path.join(test_dir, 'timing_test.png'))



FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )

    parser.add_argument(
        "--input_images", type=str, default='input',
        help = "path to image directory or video to find logos in"
    )

    parser.add_argument(
        "--input_brands", type=str, default='input',
        help = "path to directory with all brand logos to find in input images"
    )

    parser.add_argument(
        '--batch', default=False, action="store_true",
        help='Image detection mode for each file specified in input txt'
    )

    parser.add_argument(
        '--test', default=False, action="store_true",
        help='Test routine: run on few images in /data/test/ directory'
    )

    parser.add_argument(
        "--output", type=str, default="../data/test/",
        help = "output path: either directory for single/batch image, or filename for video"
    )

    parser.add_argument(
        "--outtxt", default=False, action="store_true",
        help = "save text file with inference results"
    )

    parser.add_argument(
        '--yolo_model', type=str, dest='model_path', default = 'keras_yolo3/yolo_weights_logos.h5',
        help='path to YOLO model weight file'
    )

    parser.add_argument(
        '--anchors', type=str, dest='anchors_path', default = 'keras_yolo3/model_data/yolo_anchors.txt',
        help='path to YOLO anchors'
    )

    parser.add_argument(
        '--classes', type=str, dest='classes_path', default = 'keras_yolo3/data_classes.txt',
        help='path to YOLO class specifications'
    )

    parser.add_argument(
        '--gpu_num', type=int, default = 1,
        help='Number of GPU to use'
    )

    parser.add_argument(
        '--confidence', type=float, dest = 'score', default = 0.1,
        help='YOLO confidence threshold above which to show predictions'
    )

    parser.add_argument(
        '--features', type=str, dest='features', default = 'inception_logo_features.hdf5',
        help='path to LogosInTheWild logos features extracted by InceptionV3'
    )

    parser.add_argument(
        '--fpr', type=float, dest = 'fpr', default = 0.95,
        help='False positive rate target to define similarity cutoffs'
    )

    FLAGS = parser.parse_args()

    if FLAGS.test:
        test()
        exit()
    save_to_txt = FLAGS.outtxt
    output_txt = os.path.abspath(os.path.join(FLAGS.output, 'out.txt'))
    if FLAGS.image:
        """
        Image detection mode, either prompt user input or was passed as argument
        """
        print("Image detection mode")

        if FLAGS.input_brands == 'input':
            print('Input logos to search for in images: (file-by-file or entire directory)')

            FLAGS.input_brands = parse_input()

        elif os.path.isfile(FLAGS.input_brands):
            FLAGS.input_brands = [ os.path.abspath(FLAGS.input_brands)  ]

        elif os.path.isdir(FLAGS.input_brands):
            FLAGS.input_brands = [ os.path.abspath(os.path.join(FLAGS.input_brands, f)) for f in os.listdir(FLAGS.input_brands) if f.endswith(('.jpg', '.png')) ]
        else:
            exit('Error: path not found:', FLAGS.input_brands)


        if FLAGS.batch and FLAGS.input_images.endswith('.txt'):
            print("Batch image detection mode: reading "+FLAGS.input_images)
            with open(FLAGS.input_images, 'r') as file:
                file_list = [line.split(' ')[0] for line in file.read().splitlines()]
            FLAGS.input_images = [os.path.abspath(f) for f in file_list]

            output_txt = FLAGS.input_images.split('.txt')+'_pred.txt'
            save_to_txt = True

        elif FLAGS.input_images == 'input':
            print('Input images to be scanned for logos: (file-by-file or entire directory)')
            FLAGS.input_images = parse_input()

        elif os.path.isdir(FLAGS.input_images):
            FLAGS.input_images = [ os.path.abspath(os.path.join(FLAGS.input_images, f)) for f in os.listdir(FLAGS.input_images) if f.endswith(('.jpg', '.png')) ]
        elif os.path.isfile(FLAGS.input_images):
            FLAGS.input_images = [ os.path.abspath(FLAGS.input_images)  ]
        else:
            exit('Error: path not found:', FLAGS.input_images)


        print('Found {} input brands: {}...'.format(len(FLAGS.input_brands), [ os.path.basename(f) for f in FLAGS.input_brands[:5]]))
        print('Found {} input images: {}...'.format(len(FLAGS.input_images), [ os.path.basename(f) for f in FLAGS.input_images[:5]]))

        output_path = FLAGS.output
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # define YOLO logo detector
        yolo = YOLO(**{"model_path": FLAGS.model_path,
                    "anchors_path": FLAGS.anchors_path,
                    "classes_path": FLAGS.classes_path,
                    "score" : FLAGS.score,
                    "gpu_num" : FLAGS.gpu_num,
                    "model_image_size" : (416, 416),
                    }
                   )


        input_paths = sorted(FLAGS.input_brands)
        input_labels = [ os.path.basename(s).split('test_')[-1].split('.')[0] for s in input_paths]

        ## load pre-processed features database
        brand_map, features = load_features(FLAGS.features)

        ## load inception model
        model, preprocess_input = load_extractor_model()
        my_preprocess = lambda x: chain_preprocess(x, input_shape, preprocess_input)

        # compute cosine similarity between input brand images and all LogosInTheWild logos
        ( img_input, feat_input, sim_cutoff, (bins, cdf_list)
        ) = load_brands_compute_cutoffs(input_paths, (model, my_preprocess), features, sim_threshold)

        start = timer()
        # cycle trough input images, look for logos and then match them against inputs
        text_out = ''
        for i, img_path in enumerate(FLAGS.input_images):
            prediction, r_image = detect_logo(yolo, img_path, save_img = save_img_logo,
                                              save_img_path = FLAGS.output,
                                              postfix='_logo')

            text = match_logo(img_path, prediction, (model, my_preprocess),
                    (feat_input, sim_cutoff, bins, cdf_list, input_labels),
                    save_img = save_img_match, save_img_path=FLAGS.output)
            print(text)
            text_out += text

        if save_to_txt:
            with open(output_txt,'w') as txtfile:
                txtfile.write(text_out)

        end = timer()
        print('Processed {} images in {:.1f}sec - {:.1f}FPS'.format(
             len(FLAGS.input_images), end-start, len(FLAGS.input_images)/(end-start)
             ))

    # video mode
    # elif FLAGS.video:
    else:
        print("Must specify either --image or --video.  See usage with --help.")
