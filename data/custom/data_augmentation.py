import os
from shutil import copyfile

import numpy as np
import cv2 as cv

# [blue green red]

def r2b(image) -> None:
    """Convert red colors to blue."""
    img = image.copy()
    red = img[:,:,2].copy()
    blue = img[:,:,0].copy()
    img[:,:,0] = red
    img[:,:,2] = blue
    return img

def r2y(image) -> None:
    """Convert red colors to yellow."""
    img = image.copy()
    red = img[:,:,2].copy()
    img[:,:,1] = red
    return img

def y2r(image) -> None:
    """Convert yellow colors to red."""
    img = image.copy()
    imgHLS = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    imgHLS[:,:,0] -= 20
    img = cv.cvtColor(imgHLS, cv.COLOR_HSV2BGR)
    return img

def y2b(image) -> None:
    """Convert yellow colors to blue."""
    img = image.copy()
    imgHLS = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    imgHLS[:,:,0] += 90
    img = cv.cvtColor(imgHLS, cv.COLOR_HSV2BGR)
    return img

def b2y(image) -> None:
    """Convert blue colors to yellowish."""
    img = image.copy()
    imgHLS = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    imgHLS[:,:,0] -= 80
    img = cv.cvtColor(imgHLS, cv.COLOR_HSV2BGR)
    return img

def main() -> None:
    """Perform data augmentation."""
    for cone_color in os.listdir('sources/images'):
        for image_filename in os.listdir(f'sources/images/{cone_color}'):
            img = cv.imread(f'sources/images/{cone_color}/{image_filename}')
            cv.imwrite(f'images/orig{image_filename}', img)
            name = image_filename.split('.')[0]
            copyfile(f'sources/labels/{cone_color}/{name}.txt', f'labels/orig{name}.txt')

            # make all cones in colors red, yellow and blue
            if cone_color == 'red':
                cv.imwrite(f'images/blue{image_filename}', r2b(img))
                copyfile(f'sources/labels/{cone_color}/{name}.txt', f'labels/blue{name}.txt')
                cv.imwrite(f'images/yellow{image_filename}', r2y(img))
                copyfile(f'sources/labels/{cone_color}/{name}.txt', f'labels/yellow{name}.txt')
            elif cone_color == 'yellow':
                cv.imwrite(f'images/red{image_filename}', y2r(img))
                copyfile(f'sources/labels/{cone_color}/{name}.txt', f'labels/red{name}.txt')
                cv.imwrite(f'images/blue{image_filename}', y2b(img))
                copyfile(f'sources/labels/{cone_color}/{name}.txt', f'labels/blue{name}.txt')
            elif cone_color == 'blue':
                cv.imwrite(f'images/red{image_filename}', r2b(img))
                copyfile(f'sources/labels/{cone_color}/{name}.txt', f'labels/red{name}.txt')
                cv.imwrite(f'images/yellow{image_filename}', b2y(img))
                copyfile(f'sources/labels/{cone_color}/{name}.txt', f'labels/yellow{name}.txt')

if __name__ == "__main__":
    main()


