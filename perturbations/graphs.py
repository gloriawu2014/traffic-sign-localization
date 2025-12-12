"""
Script to generate graphs for weather perturbations results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def separate_graphs():
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


def single_graph():
    # if you want all three lines on the same graph
    plt.figure(figsize=(12, 7))

    for ctype in corruption_types:
        subset = data[data["type"] == ctype]

        avg_accuracy = 100 * subset.groupby("severity")["accuracy"].mean()

        plt.plot(
            avg_accuracy.index,
            avg_accuracy.values,
            marker="o",
            label=ctype.capitalize()
        )

    plt.xlabel("Severity")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Severity for All Corruption Types")

    # Tick options depending on file
    if args.file == "weather.out":
        plt.xticks([1, 2, 3, 4, 5])
    else:
        plt.xticks([0.25, 0.50, 0.75, 1.0])

    plt.grid(True)
    plt.legend(title="Corruption Types")

    if args.file == "weather.out":
        save_path = os.path.join(output_dir, "weather.jpg")
    else:
        save_path = os.path.join(output_dir, "lighting.jpg")
    plt.savefig(save_path, format="jpg", dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="weather.out")
    parser.add_argument("--all", default=False)
    args = parser.parse_args()

    data = pd.read_csv(args.file)

    data["severity"] = pd.to_numeric(data["severity"], errors="coerce")
    data["accuracy"] = pd.to_numeric(data["accuracy"], errors="coerce")

    corruption_types = data["type"].unique()

    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    if not args.all:
        separate_graphs()
    else:
        single_graph()