# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 21:41:40 2020

@author: mauro
"""
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn

from explanation.ExplanationBase import ExplanationBase


class ShapleyExplanation(ExplanationBase):
    """
    Non-contrastive, local Explanation
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: np.array,
        model: sklearn.base.BaseEstimator,
        sparse: bool,
        config: Dict = None,
        save: bool = True,
    ) -> None:
        super(ShapleyExplanation, self).__init__(sparse, save, config)
        """
        Init the specific explanation class, the base class is "Explanation"

        Args:
            X (df): (Test) samples and features to calculate the importance for (sample, features)
            y (np.array): (Test) target values of the samples (samples, 1)
            model (object): trained (sckit-learn) model object
            sparse (bool): boolean value to generate sparse or non sparse explanation
            save (bool, optional): boolean value to save the plots. Defaults to True.

        Returns:
            None.
        """
        self.X = X
        self.y = y
        self.feature_names = list(X)
        self.model = model
        self.num_features = self.sparse_to_num_features()
        self.explanation_name = "shapely"
        self.logger = self.setup_logger(self.explanation_name)

    def calculate_explanation(self) -> None:
        """
        Explain model predictions using SHAP library

        Returns:
            None.

        """
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(self.X)

    def get_feature_values(self, sample: int = 0):
        """
        extract the feature name and its importance per sample

        Args:
            sample (int, optional): sample for which the explanation should
            be returned. Defaults to 0.

        Returns:
            feature_values (list(tuple(str, float))): list of tuples for each
            feature and its importance of a sample.

        """
        # get absolute values to get the strongst postive and negative contribution
        indexes = np.argsort(abs(self.shap_values[sample, :]))
        feature_values = []
        # sort by importance -> highst to lowest
        for index in indexes.tolist()[::-1][: self.num_features]:
            feature_values.append(
                (self.feature_names[index], self.shap_values[sample, index])
            )
        return feature_values

    def get_score(self, sample: int = 0) -> float:
        """
        calculate the overall score of the sample (output-values)

        Args:
            sample (int, optional): sample for which the explanation should
                be returned. Defaults to 0.
        Returns:
            None.
        """
        return np.sum(self.shap_values[sample, :]) + self.explainer.expected_value

    def plot(self, sample: int = 0) -> None:
        """
        Create a bar plot of the shape values for a selected sample

        Args:
            sample (int, optional): sample for which the explanation should
                be returned. Defaults to 0.
        Returns:
            None

        """
        self.plot_name = f"shapely_{sample}_{bool(self.sparse)}.png"

        indexes = np.argsort(abs(self.shap_values[sample, :]))
        sorted_idx = indexes.tolist()[::-1][: self.num_features]

        width = self.shap_values[sample, sorted_idx]
        y = np.arange(self.num_features, 0, -1)
        labels = [self.feature_names[i] for i in sorted_idx]

        fig = plt.figure(figsize=(6, max(2, int(0.5 * self.num_features))))
        plt.barh(y=y, width=width, height=0.5)
        plt.yticks(y, labels)
        plt.xlabel("Contribution")
        plt.tight_layout()
        plt.show()

        if self.save:
            fig.savefig(
                os.path.join(self.path_plot, self.plot_name), bbox_inches="tight"
            )

    def plot_shape(self, sample: int = 0) -> None:
        """
        visualize the first prediction's explanation

        Args:
            sample (int, optional): sample for which the explanation should
                be returned. Defaults to 0.
        Returns:
            None.
        """
        shap.force_plot(
            base_value=self.explainer.expected_value,
            shap_values=np.around(self.shap_values[sample, :], decimals=2),
            features=self.X.iloc[sample, :],
            matplotlib=True,
            show=False,
        )

        fig = plt.gcf()
        fig.set_figheight(4)
        fig.set_figwidth(8)
        plt.show()

        if self.save:
            fig.savefig(os.path.join(self.path_plot, self.plot_name))

    def log_output(self, sample: int) -> None:
        """
        Log the prediction values of the sample

        Args:
            sample (int): DESCRIPTION.

        Returns:
            None.
        """
        self.logger.info(
            "The expected_value was: {:.2f}".format(self.explainer.expected_value)
        )
        self.logger.info("The y_value was: {}".format(self.y.values[sample]))
        self.logger.info("The predicted value was: {}".format(self.prediction))

    def main(self, sample: int) -> None:
        """
        main function to create the explanation of the given sample. The
        method_text, natural_language_text and the plots are create per sample.

        Args:
            sample (int): number of the sample to create the explanation for

        Returns:
            None.
        """

        self.calculate_explanation()
        feature_values = self.get_feature_values(sample)

        self.natural_language_text = self.get_natural_language_text(feature_values)
        self.method_text = self.get_method_text(feature_values)

        self.plot_name = self.get_plot_name(sample)
        self.plot(sample)
        self.get_prediction(sample)
        self.save_csv(sample)
        self.log_output(sample)
        return self.method_text, self.natural_language_text


if __name__ == "__main__":

    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    diabetes = load_diabetes()

    X_train, X_val, y_train, y_val = train_test_split(
        diabetes.data, diabetes.target, random_state=0
    )

    model = RandomForestRegressor(random_state=0).fit(X_train, y_train)
    # model = sklearn.linear_model.LinearRegression().fit(X_train, y_train)
    print(model.score(X_val, y_val))

    # DF, based on which importance is checked
    X_val = pd.DataFrame(X_val, columns=diabetes.feature_names)

    sparse = False
    text = "{}"
    X = X_val
    y = y_val
    sample = 10

    for sparse in [0, 1]:
        shapely = ShapleyExplanation(X, y, model, sparse)
        shapely.main(sample)
