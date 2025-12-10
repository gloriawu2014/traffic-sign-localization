# Traffic Sign Localization

This project aims to train and evaluate a model for detecting the bounding box of a traffic sign. Additionally, we compare the model to a state-of-the-art model, YOLOv11. This project uses the DFG dataset for training and testing: https://www.vicos.si/resources/dfg/

COSC 227 Final Project: Chloe Lee, Isabella Niemi, Marah Sami, Gloria Wu

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

# Layout Overview

**Instructions on reproducing results are included in separate README.mds in the following folders**

```
├── data/                       # All data and trained models
├── deliverables/               # Project deliverables
├── model/                      # Training and evaluation of model on DFG dataset
├── newultralytics/             # Testing on YOLO model
├── perturbations/              # Evaluation of safety specifications on trained model
```

# Development

Before committing, make sure to commit any added dependencies and format code with Ruff:
```
pip freeze > requirements.txt
ruff format
```