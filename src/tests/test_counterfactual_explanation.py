# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 21:58:56 2021

@author: maurol
"""

import pytest

import pandas as pd

from sklearn.datasets import load_diabetes, load_iris
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.explanation.counterfactual_explanation import CounterfactualExplanation
from xgboost import XGBRegressor  # type: ignore


def get_regression_model():

    diabetes = load_diabetes()
    
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, random_state=0
    )
    X_test = pd.DataFrame(X_test, columns=diabetes.feature_names)
    y_test = pd.DataFrame(y_test)
    
    model = RandomForestRegressor(random_state=0).fit(X_train, y_train)    
    return model, X_test, y_test


def get_classification_model():

    iris = load_iris()
    
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, random_state=0
    )
    X_test = pd.DataFrame(X_test, columns=iris.feature_names)
    y_test = pd.DataFrame(y_test)
    
    model = RandomForestClassifier(random_state=0).fit(X_train, y_train)
    return model, X_test, y_test


def test_counterfactual_explanation_4_features():

    number_of_features = 4
    sample_index = 1

    explainer = CounterfactualExplanation(
        X_test, y_test, model, number_of_features, y_desired=2
    )

    explanation = explainer.explain(
        sample_index, separator=None
    )
    score_text, method_text, natural_language_text = explanation
    
    print(score_text)
    print(method_text)
    print(natural_language_text)

    assert (
        score_text == 'The RandomForestRegressor used 10 features to produce the predictions. The prediction of this sample was 251.8.'
    )
    assert (
        method_text
        == 'The feature importance was calculated using the Permutation Feature Importance method.'
    )
    assert (
        natural_language_text
        == "The four features which were most important for the predictions were (from highest to lowest): 'bmi' (0.15), 's5' (0.12), 'bp' (0.03), and 'age' (0.02)."
    )

def test_counterfactual_explanation_8_features():

    number_of_features = 8
    sample_index = 1

    explainer = CounterfactualExplanation(
        X_test, y_test, model, number_of_features, y_desired=2
    )

    explanation = explainer.explain(
        sample_index, separator=None
    )
    score_text, method_text, natural_language_text = explanation
    print(score_text)
    print(method_text)
    print(natural_language_text)
    

    assert (
        score_text
        == "The RandomForestRegressor used 10 features to produce the predictions. The prediction of this sample was 251.8."
    )
    assert (
        method_text
        == "The feature importance was calculated using the Permutation Feature Importance method."
    )
    assert (
        natural_language_text
        == "The eight features which were most important for the predictions were (from highest to lowest): 'bmi' (0.15), 's5' (0.12), 'bp' (0.03), 'age' (0.02), 's2' (-0.00), 'sex' (-0.00), 's3' (-0.00), and 's1' (-0.01)."
    )



if __name__ == "__main__":
    
    # model, X_test, y_test = get_regression_model()
    model, X_test, y_test = get_classification_model()

    test_counterfactual_explanation_4_features()
    test_counterfactual_explanation_8_features()
    
    