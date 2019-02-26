import cv2
import numpy as np
import os
from PIL import Image
from timeit import default_timer as timer

import utils
from utils import contents_of_bbox, features_from_image
from similarity import load_brands_compute_cutoffs, similar_matches, similarity_cutoff, draw_matches


def detect_logo(yolo, img_path, save_img, save_img_path='./', postfix=''):
    """
    Call YOLO logo detector on input image, optionally save resulting image.

    Args:
      yolo: keras-yolo3 initialized YOLO instance
      img_path: path to image file
      save_img: bool to save annotated image
      save_img_path: path to directory where to save image
      postfix: string to add to filenames
    Returns:
      prediction: list of bounding boxes in format (xmin,ymin,xmax,ymax,class_id,confidence)
      image: unaltered input image as (H,W,C) array
    """
    try:
        image = Image.open(img_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_array = np.array(image)
    except:
        print('File Open Error! Try again!')
        return None, None

    prediction, new_image = yolo.detect_image(image)

    img_out = postfix.join(os.path.splitext(os.path.basename(img_path)))
    if save_img:
        new_image.save(os.path.join(save_img_path, img_out))

    return prediction, image_array

def match_logo(img_test, prediction, model_preproc, outtxt, input_features_cdf_cutoff_labels,
               save_img, save_img_path='./', timing=False):
    """
    Given an a path to an image and a list of predicted bounding boxes,
    extract features and check each against input brand features. Declare
    a match if the cosine similarity is smaller than an input-dependent
    cutoff. Draw and annotate resulting boxes on image.

    Args:
      img_test: input image
      prediction: bounding box candidates
      model_preproc: (model, preprocess) tuple of the feature extractor model
        and the preprocessing function to be applied to image before the model
      input_features_cdf_cutoff_labels = (feat_input, sim_cutoff, bins, cdf_list, input_labels)
        tuple of lists related to input brand, giving pre-computed features,
        similarity cutoffs, cumulative similarity distribution and relative bins
        specifications, and labels to be drawn when matches are found.
      save_img: bool flag to save annotated image
      save_img_path: path to directory where to save image
      timing: bool flag to output timing information for each step, make plot
    Returns:
      outtxt: one line detailing input file path and resulting matched bounding
        boxes, space-separated in format
        (xmin,ymin,xmax,ymax,class_label,logo_confidence,similarity_percentile)
      timing: timing for each step of the pipeline, namely image read, logog candidate
        extraction, feature computation, matching to input brands
        (optional, only if timing=True)
    """

    start = timer()
    model, my_preprocess = model_preproc
    feat_input, sim_cutoff, bins, cdf_list, input_labels = input_features_cdf_cutoff_labels
    # from PIL image to np array
    #img_test = np.array(image)

    # img_test = cv2.imread(img_path) # could be removed by passing previous PIL image
    t_read = timer()-start
    candidates, i_candidates_too_small = contents_of_bbox(img_test, prediction)
    # filter predicted bboxes to discard small logos
    prediction = [ pred for i, pred in enumerate(prediction) if i not in i_candidates_too_small]
    t_box = timer()-start
    features_cand = features_from_image(candidates, model, my_preprocess)
    t_feat = timer()-start
    matches, cos_sim = similar_matches(feat_input, features_cand, sim_cutoff, bins, cdf_list)
    t_match = timer()-start

    img_path = outtxt
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
        saved = Image.fromarray(new_img).save(os.path.join(save_img_path, os.path.basename(img_path)))
        # save with opencv, remember to flip RGB->BGR
        # saved = cv2.imwrite(os.path.join(save_img_path, os.path.basename(img_path)), new_img[...,::-1])
    t_save = timer()-start
    if timing:
        return outtxt, (t_read, t_box-t_read, t_feat-t_box, t_match-t_feat,
                        t_draw-t_match, t_save-t_draw)

    return outtxt


def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open video")
    video_FourCC    = cv2.VideoWriter_fourcc(*'mp4v') #int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print(output_path, video_FourCC, video_fps, video_size)
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while vid.isOpened():
        return_value, frame = vid.read()
        if not return_value:
            break
        # opencv images are BGR, translate to RGB
        frame = frame[:,:,::-1]
        image = Image.fromarray(frame)
        out_pred, image = yolo.detect_image(image)
        result = np.asarray(image)
        if isOutput:
            out.write(result[:,:,::-1])
    vid.release()
    out.release()
    yolo.close_session()
