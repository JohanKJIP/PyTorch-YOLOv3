import os
from shutil import copyfile

import numpy as np
import cv2 as cv

for i, (image_filename, label_filename) in enumerate(zip(os.listdir('images'), os.listdir('labels'))):
    img = cv.imread(f'images/{image_filename}')
    if not os.path.exists('augmentation'):
        os.makedirs('augmentation')
    if not os.path.exists('augmentation/images'):
        os.makedirs('augmentation/images')
    if not os.path.exists('augmentation/labels'):
        os.makedirs('augmentation/labels')  

    # make blue cones
    red = img[:,:,2].copy()
    blue = img[:,:,0].copy()

    img[:,:,0] = red
    img[:,:,2] = blue

    cv.imwrite(f'augmentation/images/{i+600}.jpeg', img)
    copyfile(f'labels/{label_filename}', f'augmentation/labels/{i+600}.txt')

    # redo to original
    img[:,:,2] = red
    img[:,:,0] = blue

    # make yellow cones
    red = img[:,:,2].copy()
    green = img[:,:,1].copy()
    blue = img[:,:,0].copy()

    img[:,:,1] = red
    cv.imwrite(f'augmentation/images/{i+2000}.jpeg', img)
    copyfile(f'labels/{label_filename}', f'augmentation/labels/{i+2000}.txt')
