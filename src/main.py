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
    shuffle_in_unison,
    experiment_setup,
    create_treatment_dataframe
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


path_config = os.path.join(path_base, "src", "resources")
path_load = os.path.join(path_base, "dataset", "training")
path_model_base = os.path.join(path_base, "model")

data = DataConfig(path_config)
data_config = data.load_config()


for field in ['all']:

    model_name = [name for name in os.listdir(path_model_base) if field in name][-1]
    print(model_name)
    path_model = os.path.join(path_model_base, model_name)

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
    
    # score =  y[
    #     [col for col in list(y) if "player.rating" in col]
    # ].mean(axis=1).reset_index()
    
    
    X, y = shuffle_in_unison(X, y)
            

    samples_dict = experiment_setup(X) 
    df_treatment = create_treatment_dataframe(samples_dict)
    
    path_save = os.path.join(os.path.dirname(os.getcwd()), "reports", field)
    
    df_treatment.to_csv(
        os.path.join(path_save, 'treatment_groups.csv'),
       sep=";",
       encoding="utf-8-sig",
    )
    
    # Global, Non-contrastive
    for samples, sparse in samples_dict["permutation"]:
        permutation = PermutationExplanation(X, y, model, sparse, config)
        for sample in samples:
            sample_index = map_index_to_sample(X, sample)
            method_text, explanation_text = permutation.main(sample_index, sample)
            print(sample, method_text, explanation_text)

    # Local, Non-contrastive
    for samples, sparse in samples_dict["shapley"]:
        shapely = ShapleyExplanation(X, y, model, sparse, config)
        for sample in samples:
            sample_index = map_index_to_sample(X, sample)
            method_text, explanation_text = shapely.main(sample_index, sample)
            print(method_text, explanation_text)

    # Global, Contrastive
    for samples, sparse in samples_dict["surrogate"]:
        surrogate = SurrogateModelExplanation(X, y, model, sparse, config)
        for sample in samples:            
            sample_index = map_index_to_sample(X, sample)
            method_text, explanation_text = surrogate.main(sample_index, sample)
            print(sample, method_text, explanation_text)
            
    # Local, Contrastive
    for  samples, sparse in samples_dict["counterfactual"]:
        counterfactual = CounterfactualExplanation(
            X, y, model, sparse, config, y_desired=y.values.max()
        )
        for sample in samples:
            sample_index = map_index_to_sample(X, sample)
            method_text, explanation_text = counterfactual.main(sample_index, sample)
            print(sample, method_text, explanation_text)
