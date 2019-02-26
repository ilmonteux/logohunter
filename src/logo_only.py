"""
Run generic logo detection on input images, without matching to a specific brand
"""
import argparse
from keras_yolo3.yolo import YOLO
import os
import sys
from timeit import default_timer as timer

from logos import detect_logo, match_logo, detect_video
import utils
from utils import parse_input


output_txt = 'out.txt'
FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode'
    )

    parser.add_argument(
        '--video', default=False, action="store_true",
        help='Video detection mode'
    )

    parser.add_argument(
        "--input_images", type=str, default='input',
        help = "path to image directory or video to find logos in"
    )

    parser.add_argument(
        "--output", type=str, default="../data/test/",
        help = "output path: either directory for single/batch image, or filename for video"
    )

    parser.add_argument(
        "--outtxt", default=False, dest='save_to_txt', action="store_true",
        help = "save text file with inference results"
    )

    parser.add_argument(
        "--no_save_img", default=False, action="store_true",
        help = "do not save output images with annotated boxes"
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
        '--classes', type=str, dest='classes_path', default = 'data_classes.txt',
        help='path to YOLO class specifications'
    )

    parser.add_argument(
        '--gpu_num', type=int, default = 1,
        help='Number of GPU to use'
    )

    parser.add_argument(
        '--confidence', type=float, dest = 'score', default = 0.1,
        help='YOLO object confidence threshold above which to show predictions'
    )

    FLAGS = parser.parse_args()
    save_img_logo = not FLAGS.no_save_img

    # define YOLO logo detector
    yolo = YOLO(**{"model_path": FLAGS.model_path,
                "anchors_path": FLAGS.anchors_path,
                "classes_path": FLAGS.classes_path,
                "score" : FLAGS.score,
                "gpu_num" : FLAGS.gpu_num,
                "model_image_size" : (416, 416),
                }
               )

    # image detection mode
    if FLAGS.image:

        if FLAGS.input_images.endswith('.txt'):
            print("Batch image detection mode: reading "+FLAGS.input_images)
            output_txt = FLAGS.input_images.split('.txt')[0]+'_pred_logo.txt'
            FLAGS.save_to_txt = True
            with open(FLAGS.input_images, 'r') as file:
                file_list = [line.split(' ')[0] for line in file.read().splitlines()]
            FLAGS.input_images = [os.path.abspath(f) for f in file_list]


        elif FLAGS.input_images == 'input':
            print('Input images to be scanned for logos: (file-by-file or entire directory)')
            FLAGS.input_images = parse_input()

        elif os.path.isdir(FLAGS.input_images):
            FLAGS.input_images = [ os.path.abspath(os.path.join(FLAGS.input_images, f)) for f in os.listdir(FLAGS.input_images) if f.endswith(('.jpg', '.png')) ]
        elif os.path.isfile(FLAGS.input_images):
            FLAGS.input_images = [ os.path.abspath(FLAGS.input_images)  ]
        else:
            exit('Error: path not found: {}'.format(FLAGS.input_images))

        start = timer()
        # cycle trough input images, look for logos and then match them against inputs
        text_out = ''
        for i, img_path in enumerate(FLAGS.input_images):
            text_out += (img_path+' ')
            prediction, image = detect_logo(yolo, img_path, save_img = save_img_logo,
                                              save_img_path = FLAGS.output,
                                              postfix='_logo')
            for pred in prediction:
                text_out += ','.join([str(p) for p in pred])+' '
            text_out += '\n'

        if FLAGS.save_to_txt:
            with open(output_txt,'w') as txtfile:
                txtfile.write(text_out)

        end = timer()
        print('Processed {} images in {:.1f}sec - {:.1f}FPS'.format(
             len(FLAGS.input_images), end-start, len(FLAGS.input_images)/(end-start)
             ))

    # video detection mode
    elif FLAGS.video:
        if FLAGS.input_images == 'input':
            print('Input video to be scanned for logos: enter one file')
            FLAGS.input_images = parse_input()[0]

        elif os.path.isfile(FLAGS.input_images):
            FLAGS.input_images = os.path.abspath(FLAGS.input_images)
        else:
            exit('Error: path not found: {}'.format(FLAGS.input_images))

        if FLAGS.output == "../data/test/":
            FLAGS.output = os.path.splitext(FLAGS.input_images)[0]+'.mp4'

        detect_video(yolo, video_path = FLAGS.input_images, output_path = FLAGS.output)
    else:
        print("Must specify either --image or --video.  See usage with --help.")
