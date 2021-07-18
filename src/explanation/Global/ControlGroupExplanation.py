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

from explanation.ExplanationBase import ExplanationBase

class ControlGroupExplanation(ExplanationBase):
    """
    Control Group Explanation
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
        super(ControlGroupExplanation, self).__init__(sparse, show_rating, save, config)
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

        self.num_features = self.sparse_to_num_features()
        
        self.natural_language_text = None
        self.method_text = None
         
        self.explanation_name = "control_group"
        self.logger = self.setup_logger(self.explanation_name)
        self.plot_name = 'control_group'


    def main(self, sample_index, sample):
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