#
# MASTER SCRIPT TO DETECT LOGOS IN IMAGE AND FIND MATCHES
# for each image, pass logos, compute features, see if logo passes cutoff
#

import sys, os
from PIL import Image
import cv2
import numpy as np

import argparse
import readline
readline.parse_and_bind("tab: complete")

from keras_yolo3.yolo import YOLO
from utils import contents_of_bbox, load_features, pad_image, bbox_colors, draw_matches
from similarity import features_from_image, similarity_cutoff, similar_matches

from timeit import default_timer as timer

input_shape = (299,299,3)

def load_extractor_model():
    from keras.applications.inception_v3 import InceptionV3
    from keras.applications.inception_v3 import preprocess_input
    model = InceptionV3(weights='imagenet', include_top=False)

    return model, preprocess_input

def preprocess(img, input_shape, preprocess_func):
    return preprocess_func(pad_image(img, input_shape))

def detect_logo(yolo, img_path, save_img = False, save_img_path='./'):
    """
    Call YOLO logo detector on input image, optionally save resulting image.

    Args:
      yolo: keras-yolo3 initialized YOLO instance
      img_path: path to image file
      save_img: bool to save annotated image
      save_img_path: path to directory where to save image
    """
    try:
        image = Image.open(img_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
    except:
        print('File Open Error! Try again!')
        return None, None

    prediction, r_image = yolo.detect_image(image)

    if save_img:
        r_image.save(os.path.join(save_img_path, os.path.basename(img)))

    return prediction, r_image


yolo = YOLO(**{"model_path": 'keras_yolo3/logs/003/trained_weights_final.h5',
        "anchors_path": 'keras_yolo3/model_data/yolo_anchors.txt',
        "classes_path": 'keras_yolo3/data_classes.txt',
        "score" : 0.1,
        "model_image_size" : (416, 416),
            }
           )

def test():

    ## load pre-processed features database
    filename = 'inception_logo_features.hdf5'
    brand_map, features = load_features(filename)

    ## load inception model
    model, preprocess_input = load_extractor_model()
    my_preprocess = lambda x: preprocess(x, input_shape, preprocess_input)


    ## load sample images of logos to test against
    input_paths = ['test_batman.jpg', 'test_robin.png','test_wellsfargo.png', 'test_aldi.jpg', 'test_adidas1.png', 'test_adidas1a.jpg']
    input_paths = [os.path.join('keras_yolo3', p) for p in input_paths]


    #
    img_input = []
    for path in input_paths:
        img = cv2.imread(path)[:,:,::-1]
        img_input.append(img)

    feat_input = features_from_image(np.array(img_input), model, my_preprocess)
    sim_cutoff = similarity_cutoff(feat_input, features, threshold=0.95)

    print('Resulting similarity threshold for targets:')
    for path, cutoff in zip(input_paths, sim_cutoff):
        print('    {}  {:.2f}'.format(path, cutoff))



    ## test on batman image
    # img_path = 'keras_yolo3/test11.jpg'

    images = ['test.jpg', 'testb.jpg', 'test1.jpg',  'test1a.jpg', 'test1b.jpg', 'test2.jpg', 'test3.jpg', 'test4.jpg',
             'test5.jpg', 'test6.jpg', 'test7.jpg', 'test8.jpg', 'test9.jpg', 'test10.jpg', 'test11.jpg', 'test12.jpg',
             'test13.jpg', 'test14.jpg', 'test15.jpg', 'test16.jpg', 'test17.jpg']
    images_path = [os.path.join('keras_yolo3', p) for p in images]

    start = timer()
    for i, img_path in enumerate(images_path):

        ## find candidate logos in image
        prediction, r_image = detect_logo(yolo, img_path, save_img = False)

        ## match candidate logos to input
        img_test = cv2.imread(img_path)

        candidates = contents_of_bbox(img_test, prediction)
        features_cand = features_from_image(candidates, model, my_preprocess)

        matches, cc = similar_matches(feat_input, features_cand, sim_cutoff)


        new_img = draw_matches(img_test, input_paths, prediction, matches)
        cv2.imwrite(images[i], img_test)
    end = timer()
    print('Processed {} images in {:.1f}sec - {:.0f}'.format(len(images_path), end-start, len(images_path)/(end-start) ))













test()



sys.exit()




FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str, dest='model_path',
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str, dest='anchors_path',
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str, dest='classes_path',
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )
    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    parser.add_argument(
        '--batch', type=str,
        help='Image detection mode for each file specified in input txt, will ignore all positional arguments'
    )
    parser.add_argument(
        '--confidence', type=float, dest = 'score', default = 0.3,
        help='Model confidence threshold above which to show predictions'
    )
    parser.add_argument(
        '--iou_min', type=float, dest = 'iou_threshold', default = 0.45,
        help='IoU threshold for pruning object candidates with higher IoU than higher score boxes'
    )
    parser.add_argument(
        "--input", nargs='?', type=str, required=False, default='input',
        help = "Image or video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="output",
        help = "output path: either directory for single/batch image, or filename for video"
    )

    FLAGS = parser.parse_args()

    if not os.path.isdir('output'):
        os.makedirs('output')
    if FLAGS.image:
        """
        Image detection mode, either prompt user input or was passed as argument
        """
        print("Image detection mode")

        yolo = YOLO(**vars(FLAGS))
        if FLAGS.input == 'input':
            while True:
                FLAGS.input = input('Input image filename (q to quit):')
                if FLAGS.input in ['q','quit']:
                    yolo.close_session()
                    exit()

                img = FLAGS.input
                prediction, r_image = detect_logo(yolo, img, save_img = True, save_img_path=FLAGS.output)
                if prediction is None:
                    continue

        else:
            img = FLAGS.input
            prediction, r_image = detect_logo(yolo, img, save_img = True, save_img_path=FLAGS.output)

        yolo.close_session()

    elif 'batch' in FLAGS:
        print("Batch image detection mode: reading "+FLAGS.batch)

        with open(FLAGS.batch, 'r') as file:
            file_list = [line.split(' ')[0] for line in file.read().splitlines()]
        out_txtfile = os.path.join(FLAGS.output, 'data_pred.txt')
        txtfile = open(out_txtfile,'w')

        yolo = YOLO(**vars(FLAGS))

        for img in file_list[:10]:
            prediction, r_image = detect_logo(yolo, img, save_img = True, save_img_path='output')

            txtfile.write(img+' ')
            for pred in prediction:
                txtfile.write(','.join([str(p) for p in pred])+' ')
            txtfile.write('\n')

        txtfile.close()
        yolo.close_session()
    # elif "input" in FLAGS:
    #     detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
