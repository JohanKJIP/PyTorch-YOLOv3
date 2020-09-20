import os

import cv2
import numpy as np

def guess_color_better(img) -> str:
    """Estimate the dominant colour in the image."""
    if img.shape[0] == 0 or img.shape[1] == 0:
        return 'unknown'
        
    top_crop = 0.15
    bottom_crop = 0.15
    sides_crop = 0.25
    x = img.shape[1]
    y = img.shape[0]
    cropped_img = img[int(y*top_crop):y - int(y*bottom_crop), int(x*sides_crop):int(x - x*sides_crop)]
    blurred = cv2.GaussianBlur(cropped_img, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    red_mask = cv2.inRange(hsv,(0, 100, 20), (20, 255, 255))
    yellow_mask = cv2.inRange(hsv,(20, 100, 20), (35, 255, 255))
    blue_mask = cv2.inRange(hsv,(85, 120, 20), (140, 255, 255))
    mask_list = [('red', red_mask), ('yellow', yellow_mask), ('blue', blue_mask)]

    number_samples = 20
    ys = np.random.randint(hsv.shape[0], size=number_samples)
    xs = np.random.randint(hsv.shape[1], size=number_samples)

    WHITE_PIXEL = 255
    mask_ratios = []
    # sample for masked values in the middle of the picture
    for mask_name, mask in mask_list:
        white_pixels = 0
        count = 0
        for y, x in zip(ys, xs):
            if (mask[y, x] == WHITE_PIXEL):
                white_pixels += 1
            count += 1
        ratio = white_pixels / count
        mask_ratios.append((mask_name, ratio))

    # color mask with highest ratio is the dominant colour
    sorted_mask_rations = sorted(mask_ratios, key = lambda x: x[1])
    mask_name, ratio = sorted_mask_rations[-1]
    if ratio == 0:
        return 'unknown'
    return mask_name

def guess_color(img) -> str:
    """Estimate the color based on average hue in the image."""
    if img.shape[0] == 0 or img.shape[1] == 0:
        return 'unknown'
    top_crop = 0.5
    bottom_crop = 0.1
    sides_crop = 0.25
    x = img.shape[1]
    y = img.shape[0]
    cropped_img = img[int(y/2):y - int(y*bottom_crop), int(x*sides_crop):int(x - x*sides_crop)]
    blurred = cv2.GaussianBlur(cropped_img, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    samples = 10
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
    """Run task1 image tests."""
    correct = 0
    count = 0
    for file_name in os.listdir('images'):
        img = cv2.imread(f'images/{file_name}')
        file_color = guess_color_better(img)
        print(f'{file_name}: {file_color}')
        if file_color == file_name.split('_')[0]:
            correct += 1
        count += 1
    print(f'\nAccuracy: {round(correct / count, 2) * 100}%')

if __name__ == "__main__":
    main()