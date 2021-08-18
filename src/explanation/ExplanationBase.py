# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 21:15:30 2020

@author: mauro
"""

import csv
import os
from typing import Dict

import pandas as pd

from src.explanation.ExplanationMixin import ExplanationMixin
from src.explanation.Logger import Logger
from src.explanation.utils import create_folder


class ExplanationBase(ExplanationMixin):
    """
    Explanation base class
    """
    def __init__(
        self,
        number_of_features: int = 4,
        save: bool = True,
        config: Dict = None,
    ) -> None:
        """


        Args:
            number_of_features (int, optional): DESCRIPTION. Defaults to False.
            save (bool, optional): DESCRIPTION. Defaults to True.
            config (Dict, optional): DESCRIPTION. Defaults to None.
             (TYPE): DESCRIPTION.

        Returns:
            None: DESCRIPTION.
        """
        self.number_of_features = number_of_features
        self.save = save
        self.config = config

        if not self.config:
            self.config = {}

        self.folder = self.config.get("folder", "explanation")
        self.file_name = self.config.get("file_name", "explanations.csv")

        self.setup_paths()
        self.get_number_to_string_dict()

        self.natural_language_text_empty = (
            "In your case, the {} features which "
            "contributed most to model's output were {}."
        )
        self.method_text_empty = (
            "To help you understand the model predictions, here are "
            "the {} features which were most important for "
            "the model predictions in your specific case:"
        )

        self.score_text_empty = (
            "The {} used {} features to produce the predictions. The prediction of this sample was {:.1f}."
        )
        

    def setup_paths(self):
        """


        Returns:
            None.

        """
        self.path = os.path.join(
            os.path.dirname(os.getcwd()), "reports", self.folder
        )
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

    # def sparse_to_number_of_features(
    #     self, sparse_features: int = 4, dense_features: int = 8
    # ) -> int:
    #     """
    #     Convert sparse bool into the number of selected features
    #     """
    #     if self.sparse:
    #         number_of_features = sparse_features
    #     else:
    #         number_of_features = dense_features
    #     return number_of_features

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

    def get_method_text(self) -> None:
        """
        Generate the output of the method explanation.

        Returns:
            None
        """
        return self.method_text_empty.format(
            self.num_to_str[self.number_of_features]
        )

    def get_sentences(self, feature_values: list, sentence_empty: str) -> None:
        """
        Generate the output sentences

        Args:
            feature_values -> list(tuple(name, value))
        Returns:
            None
        """

        values = []
        for feature_name, feature_value in feature_values:
            value = sentence_empty.format(feature_name, feature_value)
            values.append(value)

        sentences = self.join_text_with_comma_and_and(values)
        return sentences

    def get_natural_language_text(self, sentences):
        """
        Generate the output of the explanation in natural language.

        Returns:
            TYPE: DESCRIPTION.

        """
        return self.natural_language_text_empty.format(
            self.num_to_str[self.number_of_features], sentences
        )

    def get_score_text(self):
        """


        Args:
            feature_values (list): DESCRIPTION.

        Returns:
            TYPE: DESCRIPTION.

        """
        number_of_dataset_features = self.X.shape[1]
        return self.score_text_empty.format(
            self.model.__class__.__name__, number_of_dataset_features, self.prediction
        )
    

    def get_model_text(self):
        
        return str(self.model)

    def get_plot_name(self, sample=None):

        if sample:
            plot_name = f"{self.explanation_name}_sample_{sample}_sparse_{self.number_of_features}.png"
        else:
            plot_name = (
                f"{self.explanation_name}_sparse_{self.number_of_features}.png"
            )
        return plot_name
    
    
    def print_output(self,  separator="---" * 20):
        
        explanation = separator.join(
            [self.score_text, self.method_text, self.natural_language_text]
        )
        return explanation
    

    def __str__(self, separator ='\n'):
        return self.print_output(separator)
        

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
        assert hasattr(self, "score_text"), "instance lacks score_text"
        assert hasattr(
            self, "natural_language_text"
        ), "instance lacks natural_language_text"
        assert hasattr(self, "plot_name"), "instance lacks plot_name"
        assert hasattr(self, "prediction"), "instance lacks prediction"

        output = {
            "score_text": self.score_text,
            "method": self.method_text,
            "explanation": self.natural_language_text,
            "plot": self.plot_name,
            "number_of_features": self.number_of_features,
            "prediction": self.prediction,
        }

        df = pd.DataFrame(output, index=[sample])

        for column in ["method", "explanation", "score_text"]:
            df[column] = df[column].astype(str)
            df[column] = df[column].str.replace("\n", "\\n")

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
