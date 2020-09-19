from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.colour_estimation import guess_color_better

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        trained_on_cuda = True # TODO
        if trained_on_cuda:
            model.load_state_dict(torch.load(opt.weights_path, map_location=device))
        else:
            model.load_state_dict(torch.load(opt.weights_path))
    

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    file_name = 'data/videos/3.mp4'
    cap = cv2.VideoCapture(file_name)
    width  = int(cap.get(3))
    height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    file_name = file_name.split('.')[0]
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(f'output/{file_name}.avi', fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, img = cap.read()
        if ret == True:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_tensor = transforms.ToTensor()(rgb)
            img_tensor, _ = pad_to_square(img_tensor, 0)
            img_tensor = resize(img_tensor, 416)
            img_tensor = img_tensor.unsqueeze(0)
            img_tensor = Variable(img_tensor.type(Tensor))

            # Get detections
            with torch.no_grad():
                detections = model(img_tensor)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

            for detection in detections:
                if detection is not None:
                    detection = rescale_boxes(detection, opt.img_size, rgb.shape[:2])
                    unique_labels = detection[:, -1].cpu().unique()
                    n_cls_preds = len(unique_labels)
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                        x1 = int(x1.numpy())
                        y1 = int(y1.numpy())
                        x2 = int(x2.numpy())
                        y2 = int(y2.numpy())

                        box_w = x2 - x1
                        box_h = y2 - y1
                        start_point = (x1, y1 + box_h)
                        end_point = (x2, y1)


                        detection_frame = img[y1:y1+box_h, x1:x1+box_w]
                        #start = time.time()
                        color_name = guess_color_better(detection_frame)
                        #end = time.time()
                        #print(f'Elapsed time: {(end-start)*1000}ms')
                        # bgr color space
                        if color_name == 'blue':
                            color = (255, 0, 0)
                        elif color_name == 'yellow':
                            color = (0, 255, 255)
                        elif color_name == 'red':
                            color = (0, 0, 255)
                        else:
                            color = (169, 169, 169)

                        img = cv2.rectangle(img, start_point, end_point, color, 2)
                        cv2.putText(img, f'{color_name} {classes[int(cls_pred)]}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            out.write(img)
            cv2.imshow('Cone detection', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
