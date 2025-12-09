# Safety Specifications

# Weather Perturbations

There are three weather types: **snow**, **frost**, and **fog**, and five severity types: 1, 2, 3, 4, 5.

**Layout Overview**
```
├── images/                                     # Example images of weather perturbations
├── plots/                                      # Graphs with resulting accuracies
├── visualize_weather_images_with_boxes.py      # Script to generate images with weather perturbations and bounding boxes
├── weather_graphs.py                           # Script to generate graphs
├── weather.err                                 # Error output of weather.sb
├── weather.out                                 # Standard output of weather.sb
├── weather.py                                  # Script to generate weather perturbations and evaluate model accuracy
├── weather.sb                                  # Slurm job script to submit weather.py with each (weather, severity) pair tested 5 times
```

**Recreate Results**
```
sbatch weather.sb                               # Submits 75 trials of weather.py
python3 visualize_weather_images_with_boxes.py  # Generates images with weather perturbations; make sure a venv is activated
python3 weather_graphs.py                       # Generates graphs depicted output from weather.sb; make sure a venv is activated
```
To run a single trial of weather.py (ensure a venv is activated):
```
python3 weather.py --iou IOU --corruption CORRUPTION --severity SEVERITY --num_test NUM_TEST
```

# Lighting Perturbations