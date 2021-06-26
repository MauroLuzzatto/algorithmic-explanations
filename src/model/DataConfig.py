# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 17:25:59 2021

@author: maurol
"""
import json
import os

import pandas as pd

from src.model.utils import get_column_selection
from src.model.config import path_base


class DataConfig(object):
    def __init__(self, path_config):
        self.path_config = path_config

    def create_data_config(self, path_load, dataset_name):
        """
        generate the data configuration dictionary
        """

        print(os.path.join(path_load, dataset_name))
        df = pd.read_csv(
            os.path.join(path_load, dataset_name),
            sep=";",
            encoding="utf-8-sig",
            index_col=0,
        )
        columns = list(df)

        data_config = {}
        data_config["dataset"] = dataset_name

        targets_academic = ["academic"]
        features_academic = [
            "SAT",
            "ACT",
            "GPA",
            "Major",
            "Grade",
            "College rank",
            "Favorite subject",
        ]

        data_config["academic"] = get_column_selection(
            targets_academic, features_academic, columns
        )

        targets_democraphic = ["democraphic"]
        features_democraphic = [
            "Age",
            "Gender - ",
            "Ethnicity - ",
            "State of residence",
        ]

        data_config["democraphic"] = get_column_selection(
            targets_democraphic, features_democraphic, columns
        )

        targets_all = (
            targets_academic
            + targets_democraphic
            + ["essay.player"]
            + ["extracurricular.player"]
        )
        features_all = (
            features_academic + features_democraphic + ["Essay score"] + ["Number of"]
        )

        data_config["all"] = get_column_selection(targets_all, features_all, columns)
        return data_config

    def load_config(self):
        with open(
            os.path.join(self.path_config, "data_config.json"), "r", encoding="utf-8"
        ) as f:
            data_config = json.load(f)
        return data_config

    def save_config(self, data_config):
        with open(
            os.path.join(self.path_config, "data_config.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(data_config, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":

    path_load = os.path.join(path_base, "dataset", "training")
    path_config = os.path.join(path_base, "src", "resources")
    dataset_name = r"applications-website-up-to-20April-clean.csv_postprocessed.csv"

    data = DataConfig(path_config)
    data_config = data.create_data_config(path_load, dataset_name)
