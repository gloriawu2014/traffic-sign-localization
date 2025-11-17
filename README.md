# traffic-sign-localization

COSC 227 Final Project: Implementing, evaluating, and improving a localization model for detecting traffic signs

Chloe Lee, Isabella Niemi, Marah Sami, Gloria Wu

# Setup

From the root directory of the repository (`./traffic-sign-localization`), follow the subsequent steps to set everything up.

Create and enter a virtual environment:
```
python3 -m venv .venv
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

# Run the Model

Make sure you are in a virtual environment:
```
source .venv/bin/activate
```
Train and evaluate the model:
```
python3 main.py
```

# Development

Make sure to commit any dependencies you add:
```
pip freeze > requirements.txt
```
