# traffic-sign-localization

COSC 227 Final Project: Implementing, evaluating, and improving a localization model for detecting traffic signs

Chloe Lee, Isabella Niemi, Marah Sami, Gloria Wu

# Setup (HPC)

From the root directory of the repository, follow the subsequent steps to set everything up.

Create and enter a virtual environment:
```
python3 -m venv .venv --copies
source .venv/bin/activate
```

Install the dependencies:
```
pip install -r requirements.txt
```

Decompress the dataset:
```
cd data
tar -xvjf JPEGImages.tar.bz2
```

# Reproducing Results (HPC)

Make sure you are in a virtual environment:
```
source .venv/bin/activate
```
**Train and test the model:**
```
cd model
sbatch train.sb
sbatch test.sb
```
**Assess model's performance on perturbations:**
```
cd perturbations
sbatch lighting.sb <CORRUPTION TYPE> <SEVERITY> <IOU> <NUM_TEST>
sbatch weather.sb <LIGHTING TYPE> <SEVERITY> <IOU> <NUM_TEST>
```
**Compare to other pre-trained models:**

# Development

Make sure to commit any dependencies you add:
```
pip freeze > requirements.txt
```
Format code with Ruff (from the root directory of the repository):
```
ruff format
```