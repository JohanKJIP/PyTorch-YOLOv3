import os
import random

validation_split = 0.2

images = []
for i, image_filename in enumerate(os.listdir('images')):
    images.append(image_filename)

random.shuffle(images)

split_number = int(len(images) * (1 - 0.2))
train = images[0:split_number]
validation = images[split_number:]

with open('train.txt', 'w') as f:
    for image in train:
        f.write(f'data/custom/images/{image}\n')

with open('valid.txt', 'w') as f:
    for image in validation:
        f.write(f'data/custom/images/{image}\n')

print(len(images))
print(len(train))
print(len(validation))
