# Overview

**Task 1**

To solve task 1, I researched different methods to do colour estimation. My first approach was to convert the image to HSL colour space and measure the average hue of the picture. From this information I would then guess the colour based on some interval/threshold. However, I quickly noticed that the average colour is not the same as the dominant colour. In this particular problem, the dominant colour of the picture is more likely to be the colour of the cone. Furthermore, the estimation of the colour was 'unstable', i.e. it was flickering between colours a lot. 

Therefore, I started to experiment with k-means clustering but quickly abandonded this idea because it was too computationally expensive for live detection. 

In my final version of colour estimation, I created masks in HSV colour space for the three colours red, blue and yellow with the help of opencv's `inRange` function. I then randomly sampled pixels from each mask and calculated the ratio of white pixels. The mask with the highest ratio of white pixels was deemed the dominant colour. 

**Task 2**

For task 2 I coverted a minimal PyTorch YOLOv3 example for picture into a video detector. I chose this approach since I remembered that YOLO was mentioned during the last interview and I thought it was good idea to get more familiar with it. 

The dataset used can be found in the README file. It is a combination of two different datasets. Since all of the cones in the dataset are red, I augmented the data by making them yellow and blue as well. This is done with the `data_augmentation.py` script under the `data/custom` directory.

The model was trained on GCP with a NVIDIA Tesla T4 for around 8-12 hours. 

# Quiz answers

### Question 1: Which category is the most important in the Business Plan Presentation?

**Aswer:** *Content* is the most important category in the Business Plan Presentaiton since it gives the most amount of points, 20. 

### Question 2: The driver must be able to leave the car quickly in an emergency. What does the regulations state about driver egress time?

**Aswer:** Egress is considered complete when the driver stands next to the car both feet on the ground.

### Question 3: Is it okay to adjust the angle of the winglets after technical inspection?

**Answer:** Yes, adjustment of winglet angles, but not the position of the complete aerodynamic device in relation to the vehicle.

### Question 4: How many lateral g's are simulated during the tilt test?

**Answer:** None of the above. 


### Question 5: How should the DV log data during the race?

**Answer:** Teams must install the standardised data logger piece of hardware provided by the officials on their vehicle.

### Question 6: What level of wireless communication with the vehicle (exclusing Remote Emergency System) is allowed during the race?

**Answer:** Only one-way-telemetry for information retrieval is allowed.
