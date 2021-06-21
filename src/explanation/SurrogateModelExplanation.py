"""
Created on Tue Nov 24 21:41:40 2020

@author: mauro
"""
import os
from typing import Dict

import graphviz
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from explanation.ExplanationBase import ExplanationBase
from explanation.SurrogatePlot import SurrogatePlot
from explanation.SurrogateText import SurrogateText


class SurrogateModelExplanation(ExplanationBase):
    """
    Contrastive, global Explanation (global surrogate model)
    """

    def __init__(self, X, y, model, sparse, config: Dict = None, save=True):
        super(SurrogateModelExplanation, self).__init__(sparse, save, config)
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

        self.plot_name = f"surrogate_{bool(self.sparse)}.png"
        self.logger = self.setup_logger("surrogate")
        self.precision = 2
        self.num_features = 2
        if sparse:
            self.num_features = 1

        self.setup()

    def calculate_explanation(self, max_leaf_nodes=100):
        """
        Train a surrogate model (Decision Tree) on the predicted values
        from the original model
        """
        y_hat = self.model.predict(self.X.values)
        self.surrogate_model = DecisionTreeRegressor(
            max_depth=self.num_features, max_leaf_nodes=max_leaf_nodes
        )
        self.surrogate_model.fit(self.X, y_hat)
        self.logger.info(
            "Surrogate Model R2 score: {:.2f}".format(
                self.surrogate_model.score(self.X, y_hat)
            )
        )

    def plot(self):
        """
        use garphix to plot the decision tree
        """
        surrogatePlot = SurrogatePlot()
        dot_file = surrogatePlot(
            model=self.surrogate_model,
            feature_names=self.feature_names,
            precision=self.precision,
        )

        graphviz.Source(
            dot_file,
            filename=os.path.join(self.path_plot, "{}".format(self.plot_name)),
            format="png",
        ).view()

        if self.save:
            with open(
                os.path.join(self.path_plot, "{}.dot".format(self.plot_name)), "w"
            ) as file:
                file.write(dot_file)

    def get_method_text(self):
        """
        Define the method introduction text of the explanation type.

        Returns:
            None.
        """
        self.method_text = (
            "To help you understand this decision, here is a decision tree "
            "showing you how the mechanism made its decision:"
        )

    def get_natural_language_text(self):
        """
        Define the natural language output using the feature names and its
        values for this explanation type

        Returns:
            None.
        """
        natural_language_text = (
            "Applicants received an average score of {:.2f} if the value of {}"
        )
        surrogateText = SurrogateText(
            text=natural_language_text,
            model=self.surrogate_model,
            X=self.X,
            feature_names=self.feature_names,
        )
        return surrogateText.get_text()

    def setup(self):
        """
        Calculate the feature importance and create the text once

        Returns:
            None.
        """
        self.calculate_explanation()
        self.natural_language_output = self.get_natural_language_text()
        self.get_method_text()
        self.plot()

    def main(self, sample):
        """
        main function to create the explanation of the given sample. The
        method_text, natural_language_text and the plots are create per sample.

        Args:
            sample (int): number of the sample to create the explanation for

        Returns:
            None.
        """
        print(self.method_text)
        print(self.natural_language_output)
        self.get_prediction(sample)
        self.save_csv(sample)


if __name__ == "__main__":

    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    diabetes = load_diabetes()

    X_train, X_val, y_train, y_val = train_test_split(
        diabetes.data, diabetes.target, random_state=0
    )

    model = RandomForestRegressor().fit(X_train, y_train)
    # model = sklearn.linear_model.LinearRegression().fit(X_train, y_train)
    print(model.score(X_val, y_val))

    # DF, based on which importance is checked
    X_val = pd.DataFrame(X_val, columns=diabetes.feature_names)

    sparse = True
    text = "{}"
    X = X_val
    y = y_val
    sample = 10

    surrogateExplanation = SurrogateModelExplanation(X, y, model, sparse)
    surrogateExplanation.main(sample)
