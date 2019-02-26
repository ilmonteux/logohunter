import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils import bbox_colors, chunks, draw_annotated_box, features_from_image
from timeit import default_timer as timer
from PIL import Image


def similarity_cutoff(feat_input, features, threshold=0.95, timing=False):
    """
    Given list of input feature and feature database, compute distribution of
    cosine similarityof the database with respect to each input. Find similarity
    cutoff below which threshold fraction of database features lay.

    Args:
      feat_input: (n_input, N) array of features for input
      features: (n_database, N) array of features for logo database
      threshold: fractional threshold for setting the cutoff
    Returns:
      cutoff_list: list of cutoffs for each input
      (bins, cdf_list): bins specifications and list of CDF distributions
        for similarity of the logo database against each input.
    """

    start = timer()
    cs = cosine_similarity(X = feat_input, Y = features)

    cutoff_list = []
    cdf_list = []
    for i, cs1 in enumerate(cs):
        hist, bins = np.histogram(cs1, bins=np.arange(0,1,0.001))
        cdf = np.cumsum(hist)/len(cs1)
        cutoff = bins[np.where(cdf < threshold)][-1]
        cutoff_list.append(cutoff)
        cdf_list.append(cdf)
    end = timer()
    print('Computed similarity cutoffs given inputs in {:.2f}sec'.format(end - start))

    return cutoff_list, (bins, cdf_list)


def load_brands_compute_cutoffs(input_paths, model_preproc, features, threshold = 0.95, timing=False):
    """
    Given paths to input brand images, this is a wrapper to features_from_image()
    and similarity_cutoff().

    Args:
      input_paths: list of paths to input images
      model_preproc: (model, preprocess) tuple of model extractor and
        image preprocessing function
      features: (n_database, N) array of features for logo database
      threshold: fractional threshold for setting the cutoff
    Returns:
      img_input: list of iamges (3D np.arrays)
      feat_input: (n_input, F) array of 1D features extracted from input images
      cutoff_list: list of cutoffs for each input
      (bins, cdf_list): bins specifications and list of CDF distributions
        for similarity of the logo database against each input.
    """

    start = timer()
    img_input = []
    for path in input_paths:
        img = cv2.imread(path)
        # apppend images in RGB color ordering
        if img is not None:
            img_input.append(img[:,:,::-1])
        else:
            print(path)

    t_read  = timer()-start
    model, my_preprocess = model_preproc
    img_input = np.array(img_input)
    feat_input = features_from_image(img_input, model, my_preprocess)
    t_feat = timer()-start

    sim_cutoff, (bins, cdf_list)= similarity_cutoff(feat_input, features, threshold, timing)
    t_sim_cut = timer()-start

    if timing:
        print('Time spent in each section:')
        print('-reading images: {:.2f}sec\n-features: {:.2f}sec\n-cosine similarity: {:.2f}sec'.format(
          t_read, t_feat-t_read, t_sim_cut-t_feat
          ))

    print('Resulting 95% similarity threshold for targets:')
    for path, cutoff in zip(input_paths, sim_cutoff):
        print('    {}  {:.2f}'.format(path, cutoff))

    return img_input, feat_input, sim_cutoff, (bins, cdf_list)



def similar_matches(feat_input, features_cand, cutoff_list, bins, cdf_list):
    """
    Given features of inputs to check candidates against, compute cosine
    similarity and define a match if cosine similarity is above a cutoff.

    Args:
      feat_input:    (n_input, N) array of features for input
      features_cand: (n_candidates, N) array of features for candidates

    Returns:
      matches: dictionary mapping each logo match to its input brand and its CDF value.
      cos_sim: (n_input, n_candidates) cosine similarity matrix between inputs and candidates.
    """

    if len(features_cand)==0:
        print('Found 0 logos from 0 classes')
        return {}, np.array([])

    assert feat_input.shape[1] == features_cand.shape[1], 'matrices should have same columns'
    assert len(cutoff_list) == len(feat_input), 'there should be one similarity cutoff for each input logo'

    cos_sim = cosine_similarity(X = feat_input, Y = features_cand)

    # similarity cutoffs are defined 3 significant digits, approximate cos_sim for consistency
    cos_sim = np.round(cos_sim, 3)

    # for each input, return matches if above threshold
    # matches = []
    matches = {}
    for i in range(len(feat_input)):
        # matches = [ c for c in range(len(features_cand)) if cc[i] > cutoff_list[i]]
        # alternatively in numpy, get indices of
        match_indices = np.where(cos_sim[i] >= cutoff_list[i])

        # to avoid double positives if candidate is above threshold for multiple inputs,
        # will pick input with better cosine_similarity, meaning the one at the highest percentile
        for idx in match_indices[0]:
            cdf_match = cdf_list[i][bins[:-1] < cos_sim[i, idx]][-1]
            # if candidate not seen previously, current brand is best guess so far
            if idx not in matches:
                matches[idx] = (i, cdf_match)
            # if previously seen at lower confidence, replace with current candidate
            elif matches[idx][1] < cdf_match:
                matches[idx] = (i, cdf_match)
            else:
                continue

    n_classes = len(np.unique([v[0] for v in matches.values()]))
    print('Found {} logos from {} classes'.format(len(matches), n_classes))

    return matches, cos_sim


def draw_matches(img_test, inputs, prediction, matches):
    """
    Draw bounding boxes on image for logo candidates that match against user input.

    Args:
      img_test: input image as 3D np.array (assuming RGB ordering)
      inputs: list of annotations strings that will appear on top of each box
      prediction: logo candidates from YOLO step
      matches: array of prediction indices, prediction[matches[i]]
    Returns:
      annotated image as 3D np.array  (opencv BGR ordering)

    """

    if len(prediction)==0:
        return img_test

    image = Image.fromarray(img_test)

    colors = bbox_colors(len(inputs))
    # for internal consistency, colors in BGR notation
    colors = np.array(colors)[:,::-1]

    # for each input, look for matches and draw them on the image
    match_bbox_list_list = []

    for i in range(len(inputs)):
        match_bbox_list_list.append([])
        for i_cand, (i_match, cdf) in matches.items():
            if i==i_match:
                match_bbox_list_list[i].append(prediction[i_cand])

        # print('{} target: {} matches found'.format(inputs[i], len(match_bbox_list_list[i]) ))

    new_image = draw_annotated_box(image, match_bbox_list_list, inputs, colors)

    return np.array(new_image)
















def main():
    print('FILL ME')



if __name__ == '__main__':
    main()
