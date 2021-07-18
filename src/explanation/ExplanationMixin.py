# -*- coding: utf-8 -*-
"""
Created on Thu May 13 21:49:43 2021

@author: maurol
"""
import os

from explanation.CategoryMapper import CategoryMapper


class ExplanationMixin:
    def map_category(self, feature_name, feature_value):
        """


        Args:
            feature_name (TYPE): DESCRIPTION.
            feature_value (TYPE): DESCRIPTION.

        Returns:
            feature_value (TYPE): DESCRIPTION.

        """
        # TODO: fix path
        path_json = r"C:\Users\maurol\OneDrive\Dokumente\Python_Scripts\model_explanation_study\dataset\training\mapping"
        if f"{feature_name}.json" in os.listdir(path_json):
            mapper = CategoryMapper(path_json, feature_name)
            # print(mapper[int(feature_value)])
            feature_value = mapper[int(feature_value)]
        return feature_value

    @staticmethod
    def join_text_with_comma_and_and(values: list) -> str:
        """
        Merge values for text output with commas and only the last value
        with an "and""

        Args:
            values (list): list of values to be merged.

        Returns:
            str: new text.

        """

        if len(values) > 2:
            last_value = values[-1]
            values = ", ".join(values[:-1])
            text = values + ", and " + last_value

        else:
            text = ", and ".join(values)
        return text

    def get_number_to_string_dict(self) -> None:
        """
        map number of features to string values
        """
        self.num_to_str = {}
        self.num_to_str[1] = "one"
        self.num_to_str[2] = "two"
        self.num_to_str[3] = "three"
        self.num_to_str[4] = "four"
        self.num_to_str[5] = "five"
        self.num_to_str[6] = "six"
        self.num_to_str[7] = "seven"
        self.num_to_str[8] = "eight"
        self.num_to_str[9] = "nine"
        self.num_to_str[10] = "ten"
