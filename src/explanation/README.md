# explanations


# Global

## non-contrastive

### Permutation feature importance
Permutation feature importance measures the increase in the prediction error of the model after we permuted the feature's values, which breaks the relationship between the feature and the true outcome [1].


## contrastive

### Surrogate model
A global surrogate model is an interpretable model that is trained to approximate the predictions of a black box model. We can draw conclusions about the black box model by interpreting the surrogate model [1].


# Local

## non-contrastive

### Shapley Values
A prediction can be explained by assuming that each feature value of  the instance is a "player" in a game where the prediction is the payout.  Shapley values (a method from coalitional game theory) tells us how  to fairly distribute the "payout" among the features. The Shapley value is the average marginal contribution of a feature value across all possible coalitions [1].


## contrastive

### Counterfactual explanations
Counterfactual explanations tell us how the values of an instance have to change to significantly change its prediction. A counterfactual explanation of a prediction describes the smallest change to the feature values that changes the prediction to a predefined output. By creating counterfactual instances, we learn about how the model makes its predictions and can explain individual predictions [1].

# Options

- dense
- sparse



# Source

[1] Molnar, Christoph. "Interpretable machine learning. A Guide for Making Black Box Models Explainable", 2019. https://christophm.github.io/interpretable-ml-book/