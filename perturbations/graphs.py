"""
Script to generate graphs for weather perturbations results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", default="weather.out")
args = parser.parse_args()

data = pd.read_csv(args.file)

data["severity"] = pd.to_numeric(data["severity"], errors="coerce")
data["accuracy"] = pd.to_numeric(data["accuracy"], errors="coerce")

corruption_types = data["type"].unique()

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

for ctype in corruption_types:
    subset = data[data["type"] == ctype]

    avg_accuracy = 100 * subset.groupby("severity")["accuracy"].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(
        avg_accuracy.index, avg_accuracy.values, marker="o", label=ctype.capitalize()
    )

    plt.xlabel("Severity")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs Severity for: {ctype.capitalize()}")
    if args.file == "weather.out":
        plt.xticks([1, 2, 3, 4, 5])
    else:
        plt.xticks([0.25, 0.50, 0.75, 1.0])
    plt.grid(True)

    save_path = os.path.join(output_dir, f"{ctype}.jpg")
    plt.savefig(save_path, format="jpg", dpi=300)
    plt.close()
