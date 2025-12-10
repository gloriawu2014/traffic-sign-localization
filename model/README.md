# Model for Detection of Traffic Sign Bounding Boxes

**Layout Overview**

```
├── data/                                   # Folder containing previous attempts at training
├── images/                                 # Examples of images with ground truth and model predicted boxes
├── parse_coco.py                           # Script to parse and prepare data for training and testing
├── test.py                                 # Script to evaluate model performance on the test data
├── test.sb                                 # Slurm job script to submit test.py
├── train.py                                # Script to train the model
├── train.sb                                # Slurm job script to submit train.py
├── visualize_images_with_boxes.py          # Script to generate images with their bounding boxes
```

**Reproduce Results**

```
sbatch train.sb EPOCHS LR       # Train the model; outputs model to data/*.pth
sbatch test.sb IOU              # Evaluate model performance
```

**Show image with bounding box**

```
python3 visualize_images_with_boxes.py --iou IOU --num_images NUM_IMAGES        # Outputs images to images/, make sure a venv is activated
```