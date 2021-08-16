"""
Created on Tue Nov 24 21:41:40 2020

@author: mauro

Permutation importance does not reflect to the intrinsic predictive value of 
a feature by itself but how important this feature is for a particular model.

Source:
https://scikit-learn.org/stable/modules/permutation_importance.html

"""
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance

from src.explanation.ExplanationBase import ExplanationBase


class PermutationExplanation(ExplanationBase):
    """
    Non-contrastive, global Explanation
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: np.array,
        model: sklearn.base.BaseEstimator,
        sparse: bool,
        show_rating: bool = True,
        config: Dict = None,
        save: bool = True,
    ):
        super(PermutationExplanation, self).__init__(
            sparse, show_rating, save, config
        )
        """
        Init the specific explanation class, the base class is "Explanation"

        Args:
            X (df): (Test) samples and features to calculate the importance for (sample, features)
            y (np.array): (Test) target values of the samples (samples, 1)
            model (object): trained (sckit-learn) model object
            sparse (bool): boolean value to generate sparse or non sparse explanation
            show_rating
            save (bool, optional): boolean value to save the plots. Defaults to True.
           
        Returns:
            None.

        """
        self.X = X
        self.y = y
        self.model = model

        self.feature_names = list(X)
        self.num_features = self.sparse_to_num_features()

        self.natural_language_text_empty = (
            "The {} attributes which were most important for the automated"
            " mechanism's assignment of ratings and their average contributions"
            " were: {}."
        )

        self.method_text_empty = (
            "Here are the {} attributes which were most important for the"
            " automated mechanism’s assignment of ratings. Contribution is on a"
            " scale from 0 to 1."
        )

        self.sentence_text = "'{}' ({:.2f})"

        self.explanation_name = "permutation"
        self.logger = self.setup_logger(self.explanation_name)
        self.plot_name = self.get_plot_name()

        self.setup()

    def calculate_explanation(self, n_repeats=30):
        """
        conduct the Permutation Feature Importance and get the importance

         Args:
            n_repeats (int, optional): sets the number of times a feature
            is randomly shuffled

        Returns:
            None
        """
        self.r = permutation_importance(
            self.model,
            self.X.values,
            self.y.values,
            n_repeats=n_repeats,
            random_state=0,
        )

    def get_feature_values(self):
        """
        extract the feature name and its importance per sample

        Args:
            sample (int, optional): sample for which the explanation should
            be returned. Defaults to 0.

        Returns:
            feature_values (list(tuple(str, float))): list of tuples for each
            feature and its importance of a sample.

        """
        feature_values = []
        # sort by importance -> highst to lowest
        for index in self.r.importances_mean.argsort()[::-1][
            : self.num_features
        ]:
            feature_values.append(
                (self.feature_names[index], self.r.importances_mean[index])
            )
        return feature_values

    def plot_boxplot(self):
        """
        plot the sorted permutation feature importance using a boxplot

        Returns:
            None.

        """
        sorted_idx = self.r.importances_mean.argsort()
        values = self.r.importances[sorted_idx].T
        labels = [self.feature_names[i] for i in sorted_idx]

        fig, ax = plt.subplots(
            figsize=(6, max(2, int(0.5 * self.num_features)))
        )
        ax.boxplot(
            values[:, -self.num_features :],
            vert=False,
            labels=labels[-self.num_features :],
        )
        plt.tight_layout()
        plt.show(block=False)

    def plot(self):
        """
        Bar plot of the feature importance

        Returns:
            None.

        """
        sorted_idx = self.r.importances_mean.argsort()
        values = self.r.importances[sorted_idx].T
        labels = [self.feature_names[i] for i in sorted_idx][
            -self.num_features :
        ]

        width = np.median(values[:, -self.num_features :], axis=0)
        y = np.arange(self.num_features)

        fig = plt.figure(figsize=(6, max(2, int(0.5 * self.num_features))))
        plt.barh(y=y, width=width, height=0.5)
        plt.yticks(y, labels)
        plt.xlabel("Contribution")
        plt.tight_layout()
        plt.show()

        if self.save:
            fig.savefig(
                os.path.join(self.path_plot, self.plot_name),
                bbox_inches="tight",
            )

    def setup(self):
        """
        Since the plots and values are calculate once per trained model,
        the feature importance computatoin is done at the beginning
        when initating the class

        Returns:
            None.
        """
        self.calculate_explanation()
        self.feature_values = self.get_feature_values()
        self.natural_language_text = self.get_natural_language_text(
            self.feature_values, self.sentence_text
        )
        self.method_text = self.get_method_text(self.feature_values)
        self.plot()

    def main(self, sample_index, sample=None):
        """
        main function to create the explanation of the given sample. The
        method_text, natural_language_text and the plots are create per sample.

        Args:
            sample (int): number of the sample to create the explanation for

        Returns:
            None.
        """

        self.get_prediction(sample_index)
        self.score_text = self.get_score_text(self.num_features)
        self.save_csv(sample)

        return self.score_text, self.method_text, self.natural_language_text
