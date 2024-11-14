from src.models.isle import ISLE
from src.models.mf_elvis import MF_ELVis
from src.models.elvis import ELVis
from src.models.presley import PRESLEY
from src.models.collei import COLLEI
from src.models.diedue import DIEDUE

from src.datamodule import *


def get_model(model_name, config, nusers):
    if model_name == "MF_ELVis":
        model = MF_ELVis(d=config["d"], nusers=nusers, lr=config["lr"])
    elif model_name == "ELVis":
        model = ELVis(d=config["d"], nusers=nusers, lr=config["lr"])
    elif model_name == "PRESLEY":
        model = PRESLEY(d=config["d"], nusers=nusers, lr=config["lr"], dropout=config["dropout"])
    elif model_name == "COLLEI":
        model = COLLEI(d=config["d"], nusers=nusers, lr=config["lr"], tau=config["tau"])
    elif model_name == "ISLE":
        model = ISLE(d=config["d"], lr=config["lr"], dropout=config["dropout"])

    elif model_name == "DIEDUE":
        model = DIEDUE(d=config["d"], lr=config["lr"], dropout=config["dropout"])
    return model


def get_presley_config(config, nusers):
    return PRESLEY(config=config, nusers=nusers)


def get_dataset_constructor(model_name):
    if model_name in ["MF_ELVis", "ELVis"]:
        dataset = TripadvisorImageAuthorshipBCEDataset
    elif model_name in ["COLLEI"]:
        dataset = TripadvisorImageAuthorshipCLDataset
    elif model_name in ["PRESLEY"]:
        dataset = TripadvisorImageAuthorshipBPRDataset
    elif model_name in ["ISLE", "DIEDUE"]:
        dataset = TripadvisorImageAuthorshipISLEDataset
    return dataset
