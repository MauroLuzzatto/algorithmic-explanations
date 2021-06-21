# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 21:15:30 2020

@author: mauro
"""

import os
from typing import Dict

import pandas as pd

from explanation.ExplanationMixin import ExplanationMixin
from explanation.Logger import Logger
from explanation.utils import create_folder


class ExplanationBase(ExplanationMixin):
    """
    Explanation base class
    """

    def __init__(
        self, sparse: bool = False, save: bool = True, config: Dict = None
    ) -> None:

        self.sparse = sparse
        self.save = save

        if not config:
            config = {}

        self.folder = config.get("folder", "explanation")
        self.file_name = config.get("file_name", "explanations.csv")

        self.setup_paths()
        self.get_number_to_string_dict()

        self.natural_language_text = (
            "In your case, the {} features which "
            "contributed most to the mechanism’s "
            "decision were the features {}."
        )
        self.method_text = (
            "To help you understand this decision, here are "
            "the {} features which were most important for "
            "how the mechanism made its decision in your specific case:"
        )

    def setup_paths(self):

        self.path = os.path.join(os.path.dirname(os.getcwd()), "reports", self.folder)
        self.path_plot = create_folder(os.path.join(self.path, "plot"))
        self.path_result = create_folder(os.path.join(self.path, "results"))
        self.path_log = create_folder(os.path.join(self.path, "logs"))

    def setup_logger(self, logger_name: str) -> object:
        logger = Logger(logger_name, self.path_log)
        return logger.get_logger()

    def calculate_explanation(self):
        raise NotImplementedError("Subclasses should implement this!")

    def plot(self):
        raise NotImplementedError("Subclasses should implement this!")

    def get_feature_values(self):
        raise NotImplementedError("Subclasses should implement this!")

    def sparse_to_num_features(
        self, sparse_features: int = 2, dense_features: int = 6
    ) -> int:
        """
        Convert sparse bool into the number of selected features
        """
        if self.sparse:
            num_features = sparse_features
        else:
            num_features = dense_features
        return num_features

    def get_prediction(self, sample: int = 0) -> float:
        """
        Get the model prediction

        Args:
            sample (TYPE, optional): DESCRIPTION. Defaults to 0.

        Returns:
            None.

        """
        assert hasattr(self, "model")
        # x = self.X.values[sample, :].reshape(1, -1)
        self.prediction = self.model.predict(self.X.values)[sample]

    def get_method_text(self, feature_values: list) -> None:
        """
        Generate the output of the method explanation.

        Args:
            feature_values -> list(tuple(name, value))
        Returns:
            None
        """
        self.method_text = self.method_text.format(self.num_to_str[len(feature_values)])

    def get_natural_language_text(self, feature_values: list) -> None:
        """
        Generate the output of the explanation in natural language.

        Args:
            feature_values -> list(tuple(name, value))
        Returns:
            None
        """

        sentence = "'{}' with an average contribution of {:.2f}"

        values = []
        for feature_name, feature_value in feature_values:
            value = sentence.format(feature_name, feature_value)
            values.append(value)

        values = self.join_text_with_comma_and_and(values)

        return values

    def save_csv(self, sample: int) -> None:
        """
        Save the explanation to a csv. The columns contain the method_text,
        the natural_language_text, the name of the plot and the predicted
        value. The index is the Entry ID.

        Args:
            sample (TYPE, optional): DESCRIPTION.

        Returns:
            None.

        """
        assert hasattr(self, "method_text"), "instance lacks method_text"
        assert hasattr(
            self, "natural_language_output"
        ), "instance lacks natural_language_text"
        assert hasattr(self, "plot_name"), "instance lacks plot_name"
        assert hasattr(self, "prediction"), "instance lacks prediction"

        output = {
            "method": self.method_text,
            "explanation": self.natural_language_output,
            "plot": self.plot_name,
            "sparse": self.sparse,
            "prediction": self.prediction,
        }

        df = pd.DataFrame(output, index=[sample])

        # def add_double_quotes(string):
        #     return f"'{string}'"

        # df['method'] = df['method'].apply(add_double_quotes)
        # df['explanation'] = df['explanation'].apply(add_double_quotes)

        df["explanation"] = df["explanation"].astype(str)
        df["method"] = df["explanation"].astype(str)

        # df['method'] = df['method'].str.replace(',', '[comma]')
        # df['explanation'] = df['explanation'].str.replace(',', '[comma]')

        df["method"] = df["method"].str.replace("\n", "\\n")
        df["explanation"] = df["explanation"].str.replace("\n", "\\n")

        import csv

        # check if the file is already there, if not, create it
        if not os.path.isfile(os.path.join(self.path_result, self.file_name)):
            df.to_csv(
                os.path.join(self.path_result, self.file_name),
                sep=";",
                encoding="utf-8-sig",
                index_label=["Entry ID"],
                escapechar="\\",
                quotechar='"',
                quoting=csv.QUOTE_NONNUMERIC,
            )
        else:
            # append row to the file
            df.to_csv(
                os.path.join(self.path_result, self.file_name),
                sep=";",
                encoding="utf-8-sig",
                index_label=["Entry ID"],
                mode="a",
                header=False,
                escapechar="\\",
                quotechar='"',
                quoting=csv.QUOTE_NONNUMERIC,
            )
