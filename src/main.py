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
    ControlGroupExplanation,
)
from model.config import path_base
from model.DataConfig import DataConfig
from model.utils import (
    average_the_ratings,
    get_dataset,
    load_pickle,
    map_index_to_sample,
    shuffle_in_unison,
    experiment_setup,
    create_treatment_dataframe,
)


logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


path_config = os.path.join(path_base, "src", "resources")
path_load = os.path.join(path_base, "dataset", "training")
path_model_base = os.path.join(path_base, "model")

data = DataConfig(path_config)
data_config = data.load_config()





def find_winner(X, y):

    y_pred = model.predict(X.values)
    y_winner = y.copy()
    y_winner["y_pred"] = y_pred
    y_winner.reset_index(inplace=True)
    index_winner = y_winner["y_pred"].argmax()
    df_winner = y_winner.iloc[index_winner]
    return df_winner


for field in ["all"]:

    model_name = [
        name for name in os.listdir(path_model_base) if field in name
    ][-1]
    print(model_name)
    path_model = os.path.join(path_model_base, model_name)

    print(os.getcwd())
    path_save = os.path.join(os.getcwd(), "reports", field)
    print(path_save)

    config = data_config[field]
    config["folder"] = field

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
    samples_dict = experiment_setup(X)
   

    # # control group
    # for samples, sparse, show_rating in samples_dict["control_group"]:
    #     control = ControlGroupExplanation(
    #         X, y, model, sparse, show_rating, config
    #     )
    #     for sample in samples:
    #         sample_index = map_index_to_sample(X, sample)
    #         output = control.main(sample_index, sample)
    #         print(sparse, show_rating)
    #         print_output(sample, output)

    # Global, Non-contrastive
    for samples, number_of_features, _ in samples_dict["permutation"]:
        
        permutation = PermutationExplanation(
           X, y, model, number_of_features, config
        )
        
        permutation.fit(X, y)
        
        for sample in samples:
            sample_index = map_index_to_sample(X, sample)
            output = permutation.explain(sample_index)
            print(number_of_features)
            print(permutation)

    # # Local, Non-contrastive
    # for samples, sparse, show_rating in samples_dict["shapley"]:
    #     shapely = ShapleyExplanation(X, y, model, sparse, show_rating, config)
    #     for sample in samples:
    #         sample_index = map_index_to_sample(X, sample)
    #         output = shapely.main(sample_index, sample)
    #         print(sparse, show_rating)
    #         print_output(sample, output)

    # # Global, Contrastive
    # for samples, sparse, show_rating in samples_dict["surrogate"]:
    #     surrogate = SurrogateModelExplanation(
    #         X, y, model, sparse, show_rating, config
    #     )
    #     for sample in samples:
    #         sample_index = map_index_to_sample(X, sample)
    #         output = surrogate.main(sample_index, sample)
    #         print(sparse, show_rating)
    #         print_output(sample, output)

    # # Local, Contrastive
    # for samples, sparse, show_rating in samples_dict["counterfactual"]:
    #     counterfactual = CounterfactualExplanation(
    #         X, y, model, sparse, show_rating, config, y_desired=8.0
    #     )
    #     for sample in samples:
    #         sample_index = map_index_to_sample(X, sample)
    #         output = counterfactual.main(sample_index, sample)
    #         print(sparse, show_rating)
    #         print_output(sample, output)
