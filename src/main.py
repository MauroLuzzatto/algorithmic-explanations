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

    path_base = (
        r"C:\Users\maurol\OneDrive\Dokumente\Python_Scripts\model_explanation_study"
    )
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

    path_base = (
        r"C:\Users\maurol\OneDrive\Dokumente\Python_Scripts\model_explanation_study"
    )
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

    samples = X.index.tolist()[:10]  # set 1
    sparse = False

    # Global, Non-contrastive
    permutation = PermutationExplanation(X, y, model, sparse, config)
    for sample in samples:
        sample = map_index_to_sample(X, sample)
        permutation.main(sample)

    # Local, Non-contrastive
    shapely = ShapleyExplanation(X, y, model, sparse, config)
    for sample in samples:
        print(sample)
        sample = map_index_to_sample(X, sample)
        shapely.main(sample)

    # Global, Contrastive
    surrogate = SurrogateModelExplanation(X, y, model, sparse, config)
    for sample in samples:
        print(sample)
        sample = map_index_to_sample(X, sample)
        surrogate.main(sample)

    # Local, Contrastive
    counterfactual = CounterfactualExplanation(
        X, y, model, sparse, config, y_desired=y.values.max()
    )
    for sample in samples:
        sample = map_index_to_sample(X, sample)
        counterfactual.main(sample)
