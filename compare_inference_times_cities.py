# Script to compare the inference times of the different models over the test sets of the cities

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from src.utils import get_model
from codecarbon import EmissionsTracker
import torch
import pickle

# Path: compare_inference_times.py

CITIES = ["gijon", "newyork", "london"]
CITIES_EN_TO_ES = {
    "gijon": "Gijón",
    "barcelona": "Barcelona",
    "madrid": "Madrid",
    "newyork": "Nueva York",
    "paris": "París",
    "london": "Londres",
}

MODELS = ["ELVis", "MF_ELVis", "PRESLEY"]

RESULTS_PATH = "results/"
FIGURES_PATH = "figures/"

PLOT_COLORS = {"ELVis": "g", "PRESLEY": "r", "MF_ELVis": "b"}

RUNS_PER_MODEL = 1

if not os.path.exists(FIGURES_PATH):
    os.makedirs(FIGURES_PATH)
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

aucs = {}

plt.figure(figsize=(16, 7))
plt.rcParams.update({"font.size": 17})

if __name__ == "__main__":
    for city in CITIES:
        for use_cuda in [True]:
            test_cases = pickle.load(open("data/" + city + "/data_10+10/TEST_IMG", "rb"))
            test_cases_sizes = test_cases.groupby("id_test").size().values
            num_of_unique_users = len(test_cases["id_user"].unique())

            test_cases_batches = []
            current_batch = []
            current_batch_size = 0

            random_users = torch.randint(0, num_of_unique_users - 1, (max(test_cases_sizes),)).to(
                "cuda" if use_cuda else "cpu"
            )

            random_images = torch.rand((max(test_cases_sizes), 1536)).to("cuda" if use_cuda else "cpu")

            for run in range(RUNS_PER_MODEL):
                for model_name in MODELS:
                    if model_name == "ELVis":
                        d = 256
                    elif model_name == "MF_ELVis":
                        d = 1024
                    elif "PRESLEY" in model_name:
                        d = 64

                    # Time the inference and track the emissions
                    emissions_tracker = EmissionsTracker(log_level="error")

                    model = get_model(
                        model_name if model_name[:7] != "PRESLEY" else "PRESLEY",
                        config={"d": d, "lr": 0.01, "dropout": 0.5},
                        nusers=num_of_unique_users,
                    )

                    model = model.to("cuda" if use_cuda else "cpu")

                    emissions_tracker = EmissionsTracker(log_level="error")

                    emissions_tracker.start()

                    for _ in range(RUNS_PER_MODEL):
                        for test_case_size in test_cases_sizes:
                            model(
                                (
                                    random_users[:test_case_size],
                                    random_images[:test_case_size],
                                )
                            )

                    emissions_tracker.stop()

                    t = emissions_tracker.final_emissions_data.duration / RUNS_PER_MODEL
                    emissions = emissions_tracker.final_emissions * 1000 / RUNS_PER_MODEL

                    if city not in aucs:
                        aucs[city] = {}
                    if str(use_cuda) not in aucs[city]:
                        aucs[city][str(use_cuda)] = {}
                    if model_name not in aucs[city][str(use_cuda)]:
                        aucs[city][str(use_cuda)][model_name] = {}
                        aucs[city][str(use_cuda)][model_name]["time"] = 0
                        aucs[city][str(use_cuda)][model_name]["emissions"] = 0

                    aucs[city][str(use_cuda)][model_name]["time"] += t
                    aucs[city][str(use_cuda)][model_name]["emissions"] += emissions

                    print(
                        f"Batch size: {city}, Cuda: {use_cuda}, Model: {model_name}, Time: {t:.2f}, Emissions: {emissions:.2f}"
                    )

    # Dump the results
    json.dump(aucs, open(RESULTS_PATH + "inference_times.json", "w"))

    width = 0.225
    # For each each use_cuda, plot a grouped bar chart (one group per batch size)
    for use_cuda in [True]:
        plt.figure(figsize=(10, 7))
        plt.rcParams.update({"font.size": 20})
        plt.title(f"Inference Times")
        plt.xlabel("City")
        plt.ylabel("Time (s)")
        plt.xticks(np.arange(len(CITIES)), [CITIES_EN_TO_ES[city] for city in CITIES])
        for i, model_name in enumerate(MODELS):
            plt.bar(
                np.arange(len(CITIES)) + i * width,
                [aucs[city][str(use_cuda)][model_name]["time"] for city in CITIES],
                width=width,
                color=PLOT_COLORS[model_name] if model_name in PLOT_COLORS else "grey",
                label=model_name if model_name != "PRESLEY" else "BRIE",
            )

        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_PATH + f"inference_times_cuda_{use_cuda}.pdf")

        # Same for the emissions
        plt.figure(figsize=(9, 5))
        plt.rcParams.update({"font.size": 20})
        plt.title(f"Emisiones en Inferencia")
        plt.xlabel("Ciudad")
        plt.ylabel("Emisiones (gCO2)")
        plt.xticks(np.arange(len(CITIES)), [CITIES_EN_TO_ES[city] for city in CITIES])
        for i, model_name in enumerate(MODELS):
            plt.bar(
                np.arange(len(CITIES)) + i * width,
                [aucs[city][str(use_cuda)][model_name]["emissions"] for city in CITIES],
                width=width,
                color=PLOT_COLORS[model_name] if model_name in PLOT_COLORS else "grey",
                label=model_name if model_name != "PRESLEY" else "BRIE",
            )

        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_PATH + f"inference_emissions_cuda_{use_cuda}.pdf")
