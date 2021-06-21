# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 17:25:59 2021

@author: maurol
"""
import json
import os

import pandas as pd

from src.model.utils import get_column_selection


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

        targets_1 = ["academic"]
        features_1 = [
            "SAT",
            "ACT",
            "GPA",
            "Major",
            "Grade",
            "College rank",
            "Favorite subjects",
        ]
        targets_2 = ["extracurricular.player"]
        features_2 = ["Number of"]
        targets_3 = ["democraphic"]
        features_3 = ["Age", "Gender - ", "Ethnicity - ", "State of residence"]
        targets = targets_1 + targets_2 + targets_3 + ["essay.player"]
        features = features_1 + features_2 + features_3 + ["Essay score"]

        data_config["academic"] = get_column_selection(targets_1, features_1, columns)
        data_config["extracurricular"] = get_column_selection(
            targets_2, features_2, columns
        )
        data_config["democraphic"] = get_column_selection(
            targets_3, features_3, columns
        )
        data_config["all"] = get_column_selection(targets, features, columns)
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
