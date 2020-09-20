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


class ConeDetector:
    """Detect cones from a video in real-time."""
    def __init__(
        self, model_def: str, weights_path: str, class_path: str,
        conf_thres: int = 0.8, nms_thres: int = 0.4) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs("output", exist_ok=True)

        self.img_size = 416

        # Set up model
        trained_on_cuda = True
        self.model = Darknet(model_def, img_size=self.img_size).to(device)
        if weights_path.endswith(".weights"):
            # Load darknet weights
            self.model.load_darknet_weights(weights_path)
        else:
            # Load checkpoint weights
            trained_on_cuda = True  # TODO
            if trained_on_cuda:
                self.model.load_state_dict(
                    torch.load(weights_path, map_location=device)
                )
            else:
                self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()

        # get class labels
        self.classes = load_classes(opt.class_path)
        self.nms_thres = nms_thres
        self.conf_thres = conf_thres

        self.Tensor = (
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        )

    def analyse(self, file_name: str) -> None:
        """Analyse given video and predict cones."""
        cap = cv2.VideoCapture(f"data/videos/{file_name}")
        width = int(cap.get(3))
        height = int(cap.get(4))
        fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")
        file_name = file_name.split(".")[0]
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(f"output/{file_name}.avi", fourcc, fps, (width, height))

        while cap.isOpened():
            start = time.time()  # time
            ret, img = cap.read()
            if ret == True:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                init_start = time.time()

                """ faster but does not work
                rgb = pad_to_square_new(img)
                rgb = cv2.resize(rgb, (self.img_size, self.img_size))
                img_tensor = torch.from_numpy(rgb).float()
                img_tensor = img_tensor[np.newaxis, :] 
                img_tensor = img_tensor.permute(0, 3, 1, 2)
                print(img_tensor.size())
                #img_tensor = Variable(img_tensor.type(self.Tensor))
                """
                img_tensor = transforms.ToTensor()(rgb)
                img_tensor, _ = pad_to_square(img_tensor, 0)
                img_tensor = resize(img_tensor, 416)
                img_tensor = img_tensor.unsqueeze(0)
                img_tensor = Variable(img_tensor.type(self.Tensor))

                init_end = time.time()

                # Get detections
                model_start = time.time()  # time
                with torch.no_grad():
                    detections = self.model(img_tensor)
                    detections = non_max_suppression(
                        detections, self.conf_thres, self.nms_thres
                    )
                model_end = time.time()  # time

                detection_start = time.time()  # time
                for detection in detections:
                    if detection is not None:
                        detection = rescale_boxes(detection, 416, rgb.shape[:2])
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

                            detection_frame = img[y1 : y1 + box_h, x1 : x1 + box_w]
                            color_name = guess_color_better(detection_frame)
                            # bgr color space
                            if color_name == "blue":
                                color = (255, 0, 0)
                            elif color_name == "yellow":
                                color = (0, 255, 255)
                            elif color_name == "red":
                                color = (0, 0, 255)
                            else:
                                color = (169, 169, 169)

                            img = cv2.rectangle(img, start_point, end_point, color, 2)
                            cv2.putText(img, f"{color_name} {self.classes[int(cls_pred)]}", (x1, y1 - 5), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                            )
                detection_end = time.time()  # time
            end = time.time()  # time
            print(
                f"Frame time: {round((end-start)*1000, 2)}ms \t {round(1/(end-start), 2)}fps \t"
                f"model init time: {round((init_end-init_start)*1000, 2)}ms \t"
                f"model eval time: {round((model_end-model_start)*1000, 2)}ms \t"
                f"detection iteration time: {round((detection_end-detection_start)*1000, 2)}ms \t"
            )
            out.write(img)
            cv2.imshow("Cone detection", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--video", type=str, default="1.mp4", help="video file name")
    opt = parser.parse_args()
    print(opt)
    detector = ConeDetector(
        opt.model_def, opt.weights_path, opt.class_path, opt.conf_thres, opt.nms_thres
    )
    detector.analyse(opt.video)
