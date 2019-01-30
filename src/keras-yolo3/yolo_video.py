import sys, os
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import readline
readline.parse_and_bind("tab: complete")

def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        if img == 'quit':
            break
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            prediction, r_image = yolo.detect_image(image)
            r_image.show()
            r_image.save(os.path.join('output', os.path.basename(img)))
            
            out_txtfile = os.path.join('output', os.path.splitext(os.path.basename(img))[0]+'.txt')
            with open(out_txtfile,'w') as txtfile:
                for pred in prediction:
                    txtfile.write(' '.join([str(p) for p in pred]))
                    txtfile.write('\n')                                                                                
    yolo.close_session()


def detect_img_batch(yolo, batchfile):
    with open(batchfile, 'r') as file:
        file_list = [line.split(' ')[0] for line in file.read().splitlines()]
    for img in file_list:
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            prediction, r_image = yolo.detect_image(image)
            r_image.show()
            r_image.save(os.path.join('output', os.path.basename(img)))
            out_txtfile = os.path.join('output', os.path.splitext(os.path.basename(img))[0]+'.txt')
            with open(out_txtfile,'w') as txtfile:
                for pred in prediction:
                    txtfile.write(' '.join([str(p) for p in pred]))
                    txtfile.write('\n')                                                                                                   
    yolo.close_session()
                                                                
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
        '--batchfile', type=str,
        help='Image detection mode for each file specified in input txt, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif 'batchfile' in FLAGS:
        print("Batch image detection mode: reading "+FLAGS.batchfile)
        print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img_batch(YOLO(**vars(FLAGS)), FLAGS.batchfile)
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
