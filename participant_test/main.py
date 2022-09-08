import os

from eval_functions import iou, biou

import numpy as np
import cv2 as cv

def calculate_score(preds: np.array, tars: np.array) -> float:

    assert preds.shape == tars.shape, f"pred shape {preds.shape} does not match tar shape {tars.shape}"
    assert len(preds.shape) != 4, f"expected shape is (bs, ydim, xdim), but found {preds.shape}"
    assert type(preds) == np.ndarray, f"preds is a {type(preds)}, but should be numpy.ndarray"
    assert type(tars) == np.ndarray, f"tars is a {type(tars)}, but should be numpy.ndarray"
    assert type(preds[0][0][0]) == np.uint8, f"preds is not of type np.uint8, but {type(preds[0][0][0])}"
    assert type(tars[0][0][0]) == np.uint8, f"tars is not of type np.uint8, but {type(tars[0][0][0])}"

    bs = preds.shape[0]

    t_iou = 0
    t_biou = 0

    for i in range(bs):
        t_iou += iou(preds[i], tars[i])
        t_biou += biou(preds[i], tars[i])

    t_iou /= bs
    t_biou /= bs

    return (t_iou + t_biou) / 2


if __name__ == '__main__':

    datapath = "data"
    predfolder = "test"
    targetfolder = "test_labels"

    imgsize = 1500

    predpath = os.path.join(datapath, predfolder)
    targetpath = os.path.join(datapath, targetfolder)

    allowed_imagetypes = ["png", "jpg", "jpeg", "tif", "tiff"]

    assert os.path.exists(predpath), f"imagepath: '{predpath}' is not correct"
    assert os.path.exists(targetpath), f"targetpath: '{targetpath}' is not correct"

    predfiles = os.listdir(predpath)
    targetfiles = os.listdir(targetpath)

    assert len(predfiles) == len(targetfiles), f"Found {len(predfiles)} image(s) and {len(targetfiles)} target(s), the numbers must match"

    preds = np.ndarray((len(predfiles), imgsize, imgsize), dtype=np.uint8)
    tars = np.ndarray((len(predfiles), imgsize, imgsize), dtype=np.uint8)

    for i in range(len(predfiles)):

        predp = os.path.join(predpath, predfiles[i])
        tarp = os.path.join(targetpath, targetfiles[i])

        pred = cv.imread(tarp, cv.IMREAD_GRAYSCALE)
        tar = cv.imread(tarp, cv.IMREAD_GRAYSCALE)
        tar[tar == 255] = 1
        pred[pred == 255] = 1

        preds[i] = pred
        tars[i] = tar


    score = calculate_score(preds, tars)

    print(score)
