import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils import chunks
from timeit import default_timer as timer


def features_from_image(img_array, model, preprocess, batch_size = 100):
    """
    Extract features from image array given a decapitated keras model.
    Use a generator to avoid running out of memory for large inputs.

    Args:
      img_array: (N, H, W, C) list/array of input images
      model: keras model, outputs
    Returns:
      (N, F) array of 1D features
    """
    if len(img_array) == 0:
        return np.array([])

    steps = len(img_array)//batch_size + 1
    img_gen = chunks(img_array, batch_size, preprocessing_function = preprocess)
    features = model.predict_generator(img_gen, steps = steps)

    # if the generator has looped past end of array, cut it down
    features = features[:len(img_array)]

    # reshape features: flatten last three dimensions to one
    features = features.reshape(features.shape[0], np.prod(features.shape[1:]))
    return features

def similarity_cutoff(feat_input, features, threshold=0.95):
    """
    Given list of input feature and feature database, compute distribution of
    cosine similarityof the database with respect to each input. Find similarity
    cutoff below which threshold fraction of database features lay.

    Args:
      feat_input: (n_input, N) array of features for input
      features: (n_database, N) array of features for logo database
      threshold: fractional threshold
    Returns:
      cutoff_list: list of cutoffs for each input
    """

    start = timer()
    cs = cosine_similarity(X = feat_input, Y = features)
    cutoff_list = []
    # assume only one input? otherwise list
    for i, cs1 in enumerate(cs):
        hist, bins = np.histogram(cs1, bins=np.arange(0,1,0.001))
        cutoff = bins[np.where(np.cumsum(hist)< threshold*len(cs1))][-1]
        cutoff_list.append(cutoff)
    end = timer()
    print('Computed similarity cutoffs given inputs {:.2f}'.format(end - start))

    return cutoff_list

def similar_matches(feat_input, features_cand, cutoff_list):
    """
    Given features of inputs to check candidates against, compute cosine
    similarity and define a match if cosine similarity is above a cutoff.

    Args:
      feat_input:    (n_input, N) array of features for input
      features_cand: (n_candidates, N) array of features for candidates

    Returns:
      matches: (n_input, ) list of indices (from 0 to n_candidates) where a match occurred.
      cc: (n_input, n_candidates) cosine similarity matrix between inputs and candidates

    """
    if len(features_cand)==0:
        return np.array([]), np.array([])
    assert feat_input.shape[1] == features_cand.shape[1], 'matrices should have same columns'
    cc = cosine_similarity(X = feat_input, Y = features_cand)
    cc = np.round(cc, 3)

    assert len(cutoff_list) == len(feat_input)

    # for each input, return matches if
    match_indices = []
    for i in range(len(feat_input)):
        # matches = [ c for c in range(len(features_cand)) if cc[i] > cutoff_list[i]]
        # alternatively in numpy, get indices of
        matches = np.where(cc[i] >= cutoff_list[i])
        match_indices.append(matches)

    return match_indices, cc
















def main():
    print('FILL ME')



if __name__ == '__main__':
    main()
