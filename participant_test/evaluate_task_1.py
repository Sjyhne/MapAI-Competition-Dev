import argparse

import torch
import torchvision
import numpy as np
import cv2 as cv

from yaml import load, Loader, dump, Dumper
from dataloader import create_dataloader
from tqdm import tqdm
from eval_functions import iou, biou

import os
import shutil

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="config/evaluation_task_1.yaml", help="Config")
    parser.add_argument("--task", type=int, default=1, help="Which task you are submitting for")
    parser.add_argument("--device", type=str, default="cpu", help="Which device the inference should run on")
    parser.add_argument("--weights", type=str, help="Path to weights for the specific model and task")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument("--data_percentage", type=float, default=1.0, help="Percentage of the whole dataset that is used")


    args = parser.parse_args()

    opts = load(open(args.config, "r"), Loader=Loader)
    try:
        opts = opts | vars(args)
    except Exception as e:
        opts = {**opts, **vars(args)}

    model = torchvision.models.segmentation.fcn_resnet50(pretrained=False, num_classes=opts["num_classes"])

    # Commented for testing purposes
    #model.load_state_dict(torch.load(opts["weights"]))

    submissionfolder = "submission"

    taskfolder = os.path.join(submissionfolder, "task" + str(opts["task"]))

    if not os.path.exists(submissionfolder):
        os.mkdir(submissionfolder)

    if os.path.exists(taskfolder):
        answer = input(f"Are you sure you want to delete folder: '{taskfolder}'? (y/n) ")
        if answer == "y":
            shutil.rmtree(taskfolder)
        else:
            print("Exiting because you do not want to delete folder:", taskfolder)
            exit(0)

    os.mkdir(taskfolder)

    testloader = create_dataloader(opts, "test")

    predictionfolder = os.path.join(taskfolder, "predictions")

    os.mkdir(predictionfolder)

    device = opts["device"]

    model = model.to(device)

    iou_scores = np.ndarray((len(testloader)))
    biou_scores = np.ndarray((len(testloader)))

    for idx, batch in tqdm(enumerate(testloader), total=len(testloader), desc="Inference"):
        image, label, filename = batch
        image = image.to(device)
        label = label.to(device)

        filename = filename[0]

        prediction = model(image)["out"]
        prediction = torch.argmax(torch.softmax(prediction, dim=1), dim=1).squeeze().detach().numpy()

        label = label.squeeze().detach().numpy()

        prediction = np.uint8(prediction)
        label = np.uint8(label)

        iou_score = iou(prediction, label)
        biou_score = biou(label, prediction)

        iou_scores[idx] = np.round(iou_score, 6)
        biou_scores[idx] = np.round(biou_score, 6)

        filepath = os.path.join(predictionfolder, filename)

        prediction_visual = np.copy(prediction)

        for idx, value in enumerate(opts["classes"]):
            prediction_visual[prediction_visual == idx] = opts["class_to_color"][value]

        cv.imwrite(filepath, prediction)
        cv.imwrite(filepath.split(".")[0] + ".png", prediction_visual)

    print("iou_score:", np.round(iou_scores.mean(), 5), "biou_score:", np.round(biou_scores.mean(), 5))

    dump(opts, open(os.path.join(taskfolder, "opts.yaml"), "w"), Dumper=Dumper)




