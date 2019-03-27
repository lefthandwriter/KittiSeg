"""
label = "data/data_CRACK500/testcrop/20160222_080933_361_1.png"
pred = "data/data_CRACK500/20160222_080933_361_1.jpg"

# crack = 1, background = 0
"""

from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import sys
import logging

sys.path.append("/content/KittiSeg/submodules/TensorVision/")
import tensorvision.analyze as ana

def ComputeMetrics(preds_list, labels_list, threshold):
    """
    Computes metrics (Precision, Recall, Accuracy and F1 Score) on test segmentation

    preds_list: path to grayscale images of segmented images
    labels_list: path to ground truth
    thresold: 0-255. if img[r][c] > threshold, label pixel as crack
    """
    cm = None
    for pred, label in zip(preds_list, labels_list):
        logging.info("pred image: %s" % pred)
        logging.info("ground truth: %s" % label)

        img = (imread(pred) > threshold).astype(np.uint8)
        gt  = imread(label).astype(np.uint8)
        # plt.imshow(img, 'gray'); plt.show()
        # plt.imshow(gt, 'gray'); plt.show()

        if cm is None:
            cm = ana.get_confusion_matrix(gt, img)
        else:
            cm_curr = ana.get_confusion_matrix(gt, img)
            cm = ana.merge_cms(cm, cm_curr)
        ## TODO: Hausdorff distance metric

    precision = ana.get_precision(cm)
    recall = ana.get_recall(cm)
    acc = ana.get_accuracy(cm)
    f1 = ana.get_f_score(cm)
    logging.info("Precision: %f" % precision)
    logging.info("Recall: %f" % recall)
    logging.info("Accuracy: %f" % acc)
    logging.info("F1: %f" % f1)
    logging.info("ComputeMetrics completed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help='path to test images - output from softmax')
    parser.add_argument('--gt',   help='path ground truth labels')
    parser.add_argument('--threshold',   help='threshold for softmax, range 0-255')
    args = parser.parse_args()

    if args.test is None:
        # ValueError()
        pass
    if args.gt is None:
        # ValueError()
        pass
    if args.threshold is None:
        args.threshold = 127 # same as 0.5 for range 0-1

    labels_list = sorted(os.listdir(args.gt))
    files_unique = [os.path.splitext(f)[0] for f in labels_list if f.endswith(".png")]

    labels_list = [os.path.join(args.gt, l) + ".png" for l in files_unique]
    preds_list  = [os.path.join(args.test, p) + ".jpg" for p in files_unique]

    ComputeMetrics(preds_list, labels_list, int(args.threshold))

if __name__ == '__main__':
    main()
