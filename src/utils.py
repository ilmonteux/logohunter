import cv2
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import h5py
import colorsys

from timeit import default_timer as timer


def bbox_colors(n):
    hsv_tuples = [(x / n, 1., 1.) for x in range(n)]
    colors = 255 * np.array([ colorsys.hsv_to_rgb(*x) for x in hsv_tuples])

    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.

    return colors.astype(int)

def contents_of_bbox(img, bbox_list, expand=1.):
    """
    Extract portions of image inside one of the bounding boxes list.

    Args:
      img: 3D image array
      bbox_list: list of bounding box specifications, with first 4 elements
      specifying box corners in (xmin, ymin, xmax, ymax) format.
    """

    candidates =[]
    for xmin, ymin, xmax, ymax, *_ in bbox_list:

        xmin, ymin = int(xmin//expand), int(ymin//expand)
        xmax, ymax = int(np.round(xmax//expand)), int(np.round(ymax//expand))
        candidates.append(img[ymin:ymax, xmin:xmax])

    return np.array(candidates)


def pad_image(img, shape, mode = 'constant_mean'):
    """
    Resize and pad image to given size.

    Args:
      img: (H, W, C) input numpy array
      shape: (H', W') destination size
      mode: filling mode for new padded pixels. Default = 'constant_mean' returns
        grayscale padding with pixel intensity equal to mean of the array. Other
        options include np.pad() options, such as 'edge', 'mean' (by row/column)...
    Returns:
      new_im: (H', W', C) padded numpy array
    """
    if mode == 'constant_mean':
        mode_args = {'mode': 'constant', 'constant_values': np.mean(img)}
    else:
        mode_args = {'mode': mode}

    ih, iw = img.shape[:2]
    h, w = shape[:2]

    # first rescale image so that largest dimension matches target
    scale = min(w/iw, h/ih)
    nw, nh = int(iw * scale), int(ih * scale)
    img = cv2.resize(img, (nw, nh))

    # center-pad rest of image: compute padding and split in two
    xpad, ypad = shape[1]-nw, shape[0]-nh
    xpad = (xpad//2, xpad//2+xpad%2)
    ypad = (ypad//2, ypad//2+ypad%2)

    new_im = np.pad(img, pad_width=(ypad, xpad, (0,0)), **mode_args)

    return new_im


def chunks(l, n, preprocessing_function = None):
    """Yield successive n-sized chunks from l.

    Modification to work with Keras: made infinite loop,
    add preprocessing, returns np.array

    Args:
      l: iterable
      n: number of items to take for each chunk
      preprocessing_function: function that processes image (3D array)
    Returns:
      generator with n-sized np.array preprocessed chunks of the input
    """

    func = (lambda x: x) if (preprocessing_function is None) else preprocessing_function

    # in predict_generator, steps argument sets how many times looped through "while True"
    while True:
        for i in range(0, len(l), n):
            yield np.array([func(el) for el in l[i:i + n]])


def load_features(filename):
    """
    Load pre-saved features for all logos in the LogosInTheWild database
    """

    start = timer()
    # get database features
    with  h5py.File(filename, 'r') as hf:
        brand_map = list(hf.get('brand_map'))
        features = hf.get('features')
        features = np.array(features)
    end = timer()
    print('Loaded {} features from {} in {:.2f}sec'.format(features.shape, filename, end-start))

    return brand_map, features
