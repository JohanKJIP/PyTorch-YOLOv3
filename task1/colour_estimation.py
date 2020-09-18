import os

import cv2
import numpy as np

def guess_color(img) -> str:
    if img.shape[0] == 0 or img.shape[1] == 0:
        return 'unknown'
    top_crop = 0.5
    bottom_crop = 0.1
    sides_crop = 0.25
    x = img.shape[1]
    y = img.shape[0]
    cropped_img = img[int(y/2):y - int(y*bottom_crop), int(x*sides_crop):int(x - x*sides_crop)]
    #blurred = cv2.GaussianBlur(cropped_img, (5, 5), 0)
    #cv2.imshow('image', blurred)
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

    samples = 25
    xs = np.random.randint(hsv.shape[1], size=samples)
    ys = np.random.randint(hsv.shape[0], size=samples)
    summed_hue = 0
    count = 0
    for x, y in zip(xs, ys):
        pixel = hsv[y, x, :]
        if pixel[1] > 50:
            summed_hue += hsv[y, x, 0]
            count += 1
    if count == 0: count = 1
    average_hue = summed_hue / count

    if average_hue <= 20:
        return 'red'
    elif average_hue <= 75:
        return 'yellow'
    else:
        return 'blue'

def main() -> None:
    correct = 0
    count = 0
    for file_name in os.listdir('images'):
        img = cv2.imread(f'images/{file_name}')
        file_color = guess_color(img)
        print(f'{file_name}: {file_color}')
        if file_color == file_name.split('_')[0]:
            correct += 1
        count += 1
    print(f'\nAccuracy: {round(correct / count, 2) * 100}%')

if __name__ == "__main__":
    main()