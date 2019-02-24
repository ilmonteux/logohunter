import numpy as np
import os
import argparse
import readline
readline.parse_and_bind("tab: complete")

import matplotlib.pyplot as plt


def read_txt_file(filename):
    """
    Reads text files in the keras-yolo3 format, one file per line, with
    all objects space-separated within each line, object details comma-separated:

    path-to-file1.jpg xmin,ymin,xmax,ymax,class_id[,confidence] xmin,ymin,xmax,ymax,class_id[,confidence]
    path-to-file2.jpg xmin,ymin,xmax,ymax,class_id[,confidence]

    If file is detailing ground truths, each object info has 5 elements,
    that is, 4 coordinates + class index; if detailing model predictions,
    the prediction confidence is the last element for each box.

    Args:
      filename: path to text file
    Returns:
      img_list: list of paths to images
      bbox_list: list of bounding boxes
    """
    with open(filename, 'r') as file:
        img_list = []
        bbox_list = []
        for line in file.read().splitlines():
            img, bbox = line.split(' ')[0],  line.split(' ')[1:]
            img_list.append(img)

            bbox = [ bb for bb in bbox if bb != '' ]

            # skip if no predictions made
            if len(bbox)==0:
                bbox_list.append([])
                continue

            if len(bbox[0].split(','))==5:
                bbox = [[int(x) for x in bb.split(',')] for bb in bbox]
            elif len(bbox[0].split(','))==6:
                bbox = [[int(x) for x in bb.split(',')[:-1]] + [float(bb.split(',')[-1])] for bb in bbox]
            else:
                print(bbox[0])
                # raise Exception('Error: bounding boxes should be defined by either 5 or 6 comma-separated entries!')

            # sort objects by prediction confidence
            bbox = sorted(bbox, key = lambda x: x[-1], reverse=True)
            bbox_list.append(bbox)
        return img_list, bbox_list


def iou_from_bboxes(bb1, bb2):
    """
    bbox = (xmin,ymin,xmax,ymax)
    """
    assert bb1[0]<bb1[2], bb1
    assert bb1[1]<bb1[3], bb1
    assert bb2[0]<bb2[2], bb2
    assert bb2[1]<bb2[3], bb2

    # find intersection
    xmin = max(bb1[0], bb2[0])
    xmax = min(bb1[2], bb2[2])
    ymin = max(bb1[1], bb2[1])
    ymax = min(bb1[3], bb2[3])


    if xmax < xmin or ymax < ymin:
        return 0.0

    area1 = (bb1[2] - bb1[0]) * (bb1[3]-bb1[1])
    area2 = (bb2[2] - bb2[0]) * (bb2[3]-bb2[1])
    area_inters = (xmin - xmax) * (ymin - ymax)
    area_union = area1 + area2 - area_inters
    iou = area_inters / area_union

    assert iou >= 0.0
    assert iou <= 1.0
    return iou


# https://tarangshah.com/blog/2018-01-27/what-is-map-understanding-the-statistic-of-choice-for-comparing-object-detection-models/
def count_tpfpfn_from_bboxes(bbox_list_true, bbox_list_pred, conf_thr=0.5, iou_thr=0.5):
    """
    Compute true positives, false positives, false negatives for a given prediction
    confidence threshold and IoU threshold. Procedure follows PASCAL VOC format.

    Each prediction above the confidence threshold count as one positive.
    If IoU is above threshold with any of the ground truth objects, count as
    true positive, otherwise count as false positive. Any remaining true objects
    not matched to a predicted object count as fals negatives.

    Args:
      bbox_list_true: list of true bounding boxes, given as (xmin, xmax, ymin, ymax)
      bbox_list_pred: list of true bounding boxes, given as (xmin, xmax, ymin, ymax, score)
                      where score is the prediction confidence.
    Returns:
      tp, fp, fn: counts for true positives, false positives, false negatives.
      match_dict: map of matches, from true boxes to matching predicted boxes.
        Nested dictionary keys are image index, and true box index mapped to predicted
        box index.
    """
    tp, fp, fn = 0, 0, 0
    # iterate over different input images
    assert len(bbox_list_pred) == len(bbox_list_true)

    n = len(bbox_list_pred)
    match_dict = { i:{} for i in range(n)}
    for i in range(n):
        # iterate over true boxes in given image
        for j, bb1 in enumerate(bbox_list_true[i]):
            # iterate over predicted boxes in given image (sorted by highest confidence)
            for k, bb2 in enumerate(bbox_list_pred[i]):

                # discard prediction if below confidence threshold
                if bb2[-1] < conf_thr:
                    continue
                # discard if true object has already been matched to higher-confidence prediction
                if j in match_dict[i]:
                    continue

                iou = iou_from_bboxes(bb1, bb2)
                # discard prediction if there is no overlap with ground truth
                if iou == 0:
                    continue

                # call it a match if above IOU threshold, mark true box with corresponding match
                if iou > iou_thr:
                    tp += 1
                    match_dict[i][j] = k
                else:
                    fp += 1
        # after going through all predictions, count any unmatched true objects
        fn += len(bbox_list_true[i]) - len(match_dict[i])

    return (tp, fp, fn), match_dict

def prec_recalls_from_bboxes(bbox_list_true, bbox_list_pred, conf_thr_list = np.arange(0,1.01,0.05), iou_thr_list = [0.5]):
    """
    Compute precision-recall given true positives, false positives, false negatives.
    Each is computed at a given confidence threshold and IoU threshold

    Args:
      bbox_list_true: list of true bounding boxes, given as (xmin, xmax, ymin, ymax)
      bbox_list_pred: list of true bounding boxes, given as (xmin, xmax, ymin, ymax, score)
                      where score is the prediction confidence.
    Returns:
      precision = TP / ( TP + FP )
      recall = TP / ( TP + FN )
    """
    # regularize to avoid 0/0 errors
    eps = 0.01

    prec_mat, rec_mat = [], []
    for iou_thr in iou_thr_list:
        prec_r, rec_r = [], []
        for conf_thr in reversed(conf_thr_list):
            (tp, fp, fn), _ = count_tpfpfn_from_bboxes(bbox_list_true, bbox_list_pred, conf_thr=conf_thr, iou_thr=iou_thr)

            prec, rec = (tp + eps) / ( tp + fp + eps), (tp + eps) / ( tp + fn + eps)

            prec_r.append(prec)
            rec_r.append(rec)
        # append rows
        prec_mat.append(prec_r)
        rec_mat.append(rec_r)

    return prec_mat, rec_mat



def main(test_file, pred_file, fig_out):
    # check if files exist
    if not os.path.isfile(test_file):
        raise Exception('File {} not found! Check and try again.'.format(test_file))
    if not os.path.isfile(pred_file):
        raise Exception('File {} not found! Check and try again.'.format(pred_file))

    # import text files and get resulting object bounding boxes
    img_list, bbox_list_true = read_txt_file(test_file)
    img_list_pred, bbox_list_pred = read_txt_file(pred_file)

    # compute precision-recall curves for different IoU thresholds
    iou_thr_list = np.arange(0.1,0.91,0.1)
    conf_thr_list = np.arange(0,1.01,0.01)
    print('Computing precision, recall from ground truth objects and model predictions')
    prec, rec = prec_recalls_from_bboxes(bbox_list_true, bbox_list_pred,
                                         conf_thr_list = conf_thr_list,
                                         iou_thr_list = iou_thr_list
                                         )

    # plot precision-recall curves, find mean Average Precision
    print('Mean Average Precision for different IoU thresholds...')
    plt.gca().set(xlim=(0,1), ylim=(0,1), xlabel='Recall', ylabel='Precision')
    for i in range(len(prec)):
        auc = np.trapz(prec[i], rec[i])
        lbl = 'iou_min = {:.1f}, mAP={:.2f}'.format(iou_thr_list[i], auc)
        print(lbl)
        plt.plot(rec[i], prec[i], label = lbl, lw=2)
    plt.legend()
    plt.savefig('prec_recall')
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    '''
    Command line options
    '''
    parser.add_argument(
        '--test_file', type=str, dest='test_file', default='data_test.txt',
        help='path to ground truth text file in keras-yolo3 format'
    )
    parser.add_argument(
        '--pred_file', type=str, dest='pred_file', default=  'data_test_pred.txt',
        help='path to predictions text file in keras-yolo3 format'
    )
    parser.add_argument(
        '--fig_out', type=str, dest='fig_out', default='prec_recall.png',
        help='path to save location of precision-recall figure'
    )

    args = parser.parse_args()

    test_file = args.test_file
    pred_file = args.pred_file
    fig_out = args.fig_out


    main(test_file, pred_file, fig_out)
