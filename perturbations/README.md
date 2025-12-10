# Safety Specifications

**Layout Overview**
```
├── images/                                     # Example images of weather and lighting perturbations
├── plots/                                      # Graphs with resulting accuracies
├── graphs.py                                   # Script to generate graphs
├── lighting.py                                 # Script to generate lighting perturbations and evaluate model accuracy
├── lighting.sb                                 # Slrum job script to submit lighting.py with each (lighting, severity) pair tested 5 times
├── visualize_perturbed_images_with_boxes.py    # Script to generate images with weather or lighting perturbations and bounding boxes
├── weather.py                                  # Script to generate weather perturbations and evaluate model accuracy
├── weather.sb                                  # Slurm job script to submit weather.py with each (weather, severity) pair tested 5 times
```

# Weather Perturbations

There are three weather types: **snow**, **frost**, and **fog**, and five severity types: 1, 2, 3, 4, 5.

**Recreate Results**
```
sbatch weather.sb                                                               # Submits 75 trials of weather.py
python3 visualize_perturbed_images_with_boxes.py --corruption CORRUPTION        # Generates images with weather perturbations; make sure a venv is activated
python3 graphs.py --file weather.out                                            # Generates graphs depicted output from weather.sb; make sure a venv is activated
```
To run a single trial of weather.py (ensure a venv is activated):
```
python3 weather.py --iou IOU --corruption CORRUPTION --severity SEVERITY --num_test NUM_TEST
```

# Lighting Perturbations

There are three lighting types: **dusk**, **dawn**, and **bright**, and severity ranges from 0 to 1.

**Recreate Results**
```
sbatch lighting.sb                                                              # Submits 60 trials of weather.py
python3 visualize_perturbed_images_with_boxes.py --corruption CORRUPTION        # Generates images with lighting perturbations; make sure a venv is activated
python3 graphs.py --file lighting.out                                           # Generates graphs depicted output from weather.sb; make sure a venv is activated
```
To run a single trial of lighting.py (ensure a venv is activated):
```
python3 lighting.py --iou IOU --lighting_type CORRUPTION --severity SEVERITY --num_test NUM_TEST
```
