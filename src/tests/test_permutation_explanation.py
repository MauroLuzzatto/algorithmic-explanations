# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 21:58:56 2021

@author: maurol
"""

import pytest

import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from src.explanation.PermutationExplanation import PermutationExplanation


diabetes = load_diabetes()

X_train, X_val, y_train, y_val = train_test_split(
    diabetes.data, diabetes.target, random_state=0
)

X_val = pd.DataFrame(X_val, columns=diabetes.feature_names)
y_val = pd.DataFrame(y_val)


class Test:
    @pytest.fixture()
    def setup(self):
        model = RandomForestRegressor(random_state=0).fit(X_train, y_train)
        yield model

    def test_permuation_explanation_sparse(self, setup):

        sparse = True
        show_rating = False
        sample_index = 1

        permutation = PermutationExplanation(
            X_val, y_val, setup, sparse, show_rating
        )

        score_text, method_text, explanation_text = permutation.main(
            sample_index
        )

        assert (
            score_text
            == "The automated mechanism analyzed hundreds of applications and"
            " used 10 attributes to produce a rating for each applicant"
            " between 1 and 10."
        )
        assert (
            method_text
            == "Here are the four attributes which were most important for the"
            " automated mechanism’s assignment of ratings. Contribution is"
            " on a scale from 0 to 1."
        )
        assert (
            explanation_text
            == "The four attributes which were most important for the automated"
            " mechanism's assignment of ratings and their average"
            " contributions were: 'bmi' (0.15), 's5' (0.12), 'bp' (0.03),"
            " and 'age' (0.02)."
        )

    def test_permuation_explanation_dense(self, setup):

        sparse = False
        show_rating = False
        sample_index = 1

        permutation = PermutationExplanation(
            X_val, y_val, setup, sparse, show_rating
        )

        score_text, method_text, explanation_text = permutation.main(
            sample_index
        )

        assert (
            score_text
            == "The automated mechanism analyzed hundreds of applications and"
            " used 10 attributes to produce a rating for each applicant"
            " between 1 and 10."
        )
        assert (
            method_text
            == "Here are the eight attributes which were most important for the"
            " automated mechanism’s assignment of ratings. Contribution is"
            " on a scale from 0 to 1."
        )
        assert (
            explanation_text
            == "The eight attributes which were most important for the automated"
            " mechanism's assignment of ratings and their average"
            " contributions were: 'bmi' (0.15), 's5' (0.12), 'bp' (0.03),"
            " and 'age' (0.02)."
        )

    def test_permuation_explanation_show_rating(self, setup):

        sparse = True
        show_rating = True
        sample_index = 1

        permutation = PermutationExplanation(
            X_val, y_val, setup, sparse, show_rating
        )

        score_text, method_text, explanation_text = permutation.main(
            sample_index
        )

        assert (
            score_text
            == "The automated mechanism analyzed hundreds of applications and"
            " used 10 attributes to produce a rating for each applicant"
            " between 1 and 10. Your rating was 251.8."
        )
        assert (
            method_text
            == "Here are the four attributes which were most important for the"
            " automated mechanism’s assignment of ratings. Contribution is"
            " on a scale from 0 to 1."
        )
        assert (
            explanation_text
            == "The four attributes which were most important for the automated"
            " mechanism's assignment of ratings and their average"
            " contributions were: 'bmi' (0.15), 's5' (0.12), 'bp' (0.03),"
            " and 'age' (0.02)."
        )
