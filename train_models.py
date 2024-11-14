# Script to call through command line to train models

import os

# Path: train_models.py


def train_model_through_command_line(model_name: str, city: str, use_validation: bool):
    command = (
        "python main.py --stage train --model "
        + model_name
        + " --city "
        + city
        + " --use_train_val --log_to_csv"
    )
    if not use_validation:
        command += " --no_validation"

    if model_name == "ELVis":
        command += " --lr 1e-4 --batch_size 32768 --max_epochs 100 -d 256"

    elif model_name == "PRESLEY":
        command += " --lr 1e-3 --batch_size 16384 --max_epochs 15 -d 64 --dropout 0.75"

    elif model_name == "MF_ELVis":
        command += " --lr 1e-3 --batch_size 32768 --max_epochs 25 -d 1024"

    if city in ["barcelona", "gijon", "madrid"]:
        command += " --workers 4"
    elif city in ["paris", "newyork"]:
        command += " --workers 2"
    elif city in ["london"]:
        command += " --workers 1"

    print(command)
    os.system(command)


if __name__ == "__main__":
    # Cities to train
    cities = ["gijon", "barcelona", "madrid", "newyork", "paris", "london"]
    # Models to train
    models = ["PRESLEY"]
    for city in cities:
        for model in models:
            train_model_through_command_line(model, city, True)
            train_model_through_command_line(model, city, False)
