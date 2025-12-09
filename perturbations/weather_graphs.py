"""
Script to generate graphs for weather perturbations results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

data = pd.read_csv("weather_perturbations.csv")

data['severity'] = pd.to_numeric(data['severity'], errors='coerce')
data['accuracy'] = pd.to_numeric(data['accuracy'], errors='coerce')

corruption_types = data['type'].unique()

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

for ctype in corruption_types:
    subset = data[data['type'] == ctype]

    avg_accuracy = subset.groupby('severity')['accuracy'].mean()

    plt.figure(figsize=(10,6))
    plt.plot(avg_accuracy.index, avg_accuracy.values, marker='o', label=ctype.capitalize())

    plt.xlabel("Severity")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs Severity for: {ctype.capitalize()}")
    plt.xticks([1, 2, 3, 4, 5])
    plt.ylim(0, 0.2)
    plt.grid(True)
    
    save_path = os.path.join(output_dir, f"{ctype}.jpg")
    plt.savefig(save_path, format='jpg', dpi=300)
    plt.close()