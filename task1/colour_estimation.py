import os

import cv2
import numpy as np

def color(file_name: str):
    img = cv2.imread(f'images/{file_name}')
    
    top_crop = 0.5
    bottom_crop = 0.1
    sides_crop = 0.25
    x = img.shape[1]
    y = img.shape[0]
    cropped_img = img[int(y/2):y - int(y*bottom_crop), int(x*sides_crop):int(x - x*sides_crop)]
    blurred = cv2.GaussianBlur(cropped_img, (5, 5), 0)
    #cv2.imshow('image', blurred)
    #cv2.waitKey(0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
   
    summed_hue = 0
    count = 0
    step_size = 5
    for x in range(0, hsv.shape[0], step_size):
        for y in range(0, hsv.shape[1], step_size):
            pixel = hsv[x, y, :]
            if pixel[1] > 50:
                summed_hue += hsv[x, y, 0]
                count += 1
    average_hue = summed_hue / count

    if average_hue <= 15:
        return 'red'
    elif average_hue <= 75:
        return 'yellow'
    else:
        return 'blue'

def main() -> None:
    correct = 0
    count = 0
    for file_name in os.listdir('images'):
        file_color = color(file_name)
        print(f'{file_name}: {file_color}')
        if file_color == file_name.split('_')[0]:
            correct += 1
        count += 1
    print(f'\nAccuracy: {round(correct / count, 2) * 100}%')

if __name__ == "__main__":
    main()