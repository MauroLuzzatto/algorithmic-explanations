# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 17:16:29 2020

@author: mauro
"""
import logging
import os

from explanation import (
    CounterfactualExplanation,
    PermutationExplanation,
    ShapleyExplanation,
    SurrogateModelExplanation,
)
from src.model.config import path_base
from src.model.DataConfig import DataConfig
from src.model.utils import (
    average_the_ratings,
    get_dataset,
    load_pickle,
    map_index_to_sample,
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


mode = None
if mode == "testing":

    path_load = os.path.join(path_base, "dataset", "training")
    path_model = os.path.join(path_base, "model")
    path_model = os.path.join(path_model, "2021-06-19--11-32-15")

    dataset_name = "training_data_v2.csv"

    model = load_pickle(
        path_model=path_model,
        model_name="XGBRegressor.pickle",
    )

    X, y = get_dataset(path_load=path_load, name=dataset_name)
    # TODO: fix
    X = X.fillna(X.median())  # works


else:
    path_config = os.path.join(path_base, "src", "resources")
    path_load = os.path.join(path_base, "dataset", "training")
    path_model_base = os.path.join(path_base, "model")

    data = DataConfig(path_config)
    data_config = data.load_config()


for field in ["all"]:  # 'extracurricular', 'academic', 'democraphic',

    model_name = [name for name in os.listdir(path_model_base) if field in name][0]
    path_model = os.path.join(path_model_base, model_name)

    config = data_config[field]

    model = load_pickle(
        path_model=path_model,
        model_name="XGBRegressor.pickle",
    )

    X, y = get_dataset(
        path_load=path_load,
        name=data_config["dataset"],
        target=config["target"],
        features=config["features"],
    )

    new_name = f"{field}.player.rating"
    y = average_the_ratings(y, list(y), new_name)

    config["folder"] = field

    ########################################33
    # Set: y_desired
    #########################################

    samples = X.index.tolist()[:]  # set 1

    samples_dict = {
        "permutation": samples[:10],
        "shapley": samples[10:20],
        "surrogate": samples[20:30],
        "counterfactual": samples[30:40],
    }

    for sparse in [True, False]:

        # Global, Non-contrastive
        permutation = PermutationExplanation(X, y, model, sparse, config)
        for sample in samples_dict["permutation"]:
            sample = map_index_to_sample(X, sample)
            method_text, explanation_text = permutation.main(sample)
            print(method_text, explanation_text)

        # Local, Non-contrastive
        shapely = ShapleyExplanation(X, y, model, sparse, config)
        for sample in samples_dict["shapley"]:
            sample = map_index_to_sample(X, sample)
            method_text, explanation_text = shapely.main(sample)
            print(method_text, explanation_text)

        # Global, Contrastive
        surrogate = SurrogateModelExplanation(X, y, model, sparse, config)
        for sample in samples_dict["surrogate"]:
            sample = map_index_to_sample(X, sample)
            method_text, explanation_text = surrogate.main(sample)
            print(method_text, explanation_text)

        # Local, Contrastive
        counterfactual = CounterfactualExplanation(
            X, y, model, sparse, config, y_desired=y.values.max()
        )
        for sample in samples_dict["counterfactual"]:
            sample = map_index_to_sample(X, sample)
            method_text, explanation_text = counterfactual.main(sample)
            print(method_text, explanation_text)
