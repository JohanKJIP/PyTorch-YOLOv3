# Real-time cone detection on videos

Using YOLOv3 implementation from ![here](https://github.com/eriklindernoren/PyTorch-YOLOv3).

Short project report and rule quiz answers can be found in the file `report.md`.

### Results
<div style='position:relative;padding-bottom:54.921%;'><iframe src="//gifs.com/embed/final2-ANpVPj" scrolling="no" style="backface-visibility: hidden; transform: scale(1); position: absolute; height: 100%; width: 100%;" width="630" height="346" frameborder="0"></iframe></div>

Final iteration:

- ![Close cones](https://youtu.be/63t9inNP2ao)
- ![Cones on track](https://youtu.be/zXURDLLD3wE)
- ![Other cones](https://youtu.be/vtaUvGwF3Dc)

Old version, more stable but worse on long range:

- ![Close cones](https://youtu.be/TvAsGWcQL0c)
- ![Cones on track](https://youtu.be/IbSd45sqzYo)
- ![Other cones](https://youtu.be/hRu43fHbzWM)

# KTH Formula specifics

**Task 1** can be found under `utils/colour_estimation.py`. To test with the dataset for task 1, simply add a folder in this directory called `images` with the pictures inside. Run the program inside the utils folder with `python colour_estimation.py` and it will print the approximated colour and the total accuracy.

**Task 2** results can be found above. Used another YOLOv3 implementation linked above. 

# Usage

1. Install latest PyTorch
2. `pip install requirements.txt`

To detect video:
`python video_detect.py --weights_path <path_to_weights> --video <video name in data/videos>`

To train network:
1. Download sources.zip, link (todo)
2. Extract `sources` folder into `data/custom`
3. Run `data_augmentation.py` inside `custom`
4. Run `train_divider.py` inside `custom`
5. Go back to root workspace and do `python train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data`

