# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 10:37:54 2021

@author: maurol
"""
import os
import pandas as pd

from src.model.config import path_base

import os
from collections import defaultdict

import pandas as pd

from src.model.config import path_base



meta_columns = [
    "participant.id_in_session",
    "participant.code",
    "participant.label",
    "participant._is_bot",
    "participant._index_in_pages",
    "participant._max_page_index",
    "participant._current_app_name",
    "participant._current_page_name",
    "participant.time_started",
    "participant.visited",
    "participant.mturk_worker_id",
    "participant.mturk_assignment_id",
    "participant.payoff",
    "session.code",
    "session.label",
    "session.mturk_HITId",
    "session.mturk_HITGroupId",
    "session.comment",
    "session.is_demo",
    "session.config.dataset",
    "session.config.participation_fee",
    "session.config.real_world_currency_per_point",
]


def get_columns_per_round(nr_round, prefix):
    return [
        f"{prefix}.{nr_round}.player.id_in_group",
        # f'{prefix}.{nr_round}.player.role',
        # f'{prefix}.{nr_round}.player.payoff',
        # f'{prefix}.{nr_round}.player.go_back_hidden',
        f"{prefix}.{nr_round}.player.applicant_id",
        f"{prefix}.{nr_round}.player.applicant_name",
        f"{prefix}.{nr_round}.player.rating",
        # f'{prefix}.{nr_round}.group.id_in_subsession',
        # f'{prefix}.{nr_round}.subsession.round_number'
    ]


base_columns = [
    "player.id_in_group",
    # 'player.role',
    # 'player.payoff',
    # 'player.go_back_hidden',
    "player.applicant_id",
    "player.applicant_name",
    "player.rating",
    # 'group.id_in_subsession',
    # 'subsession.round_number'
]


def fix_column_name(columns):

    columns_new = []
    for col in columns:
        print(col)
        if col[1] == "":
            columns_new.append(col[0])
        else:
            columns_new.append(f'{file.split("_")[0]}.{col[0]}.{int(col[1])}')
    return columns_new



path_base = r'C:\Users\maurol\OneDrive\Dokumente\Python_Scripts\algorithmic-explanations'
folder_name = "algorithmic_explanations"
folder_name_v2 = "algorithmic_explanations_v2"

path_ratings = os.path.join(path_base, "dataset", "ratings")


file = 'essay_all_apps_wide-2021-05-22.csv'

df = pd.read_csv(os.path.join(path_ratings, folder_name, file))

file_v2 = 'essay_batch1_P2P3.csv'

df_v2 = pd.read_csv(os.path.join(path_ratings, folder_name_v2, file_v2))


df_list = []

files = [
    'essay_batch1_P2P3.csv',
    'essay_batch2_P2P3.csv'
    
    ]
folder_name = "algorithmic_explanations_v2"



import matplotlib.pyplot as plt

dict_data = defaultdict(list)
dict_df = {}
list_data = []

for file in files:
    
    print(file)


    df = pd.read_csv(os.path.join(path_ratings, folder_name, file))
    df.index = ["player1", "player2", "player3"]

    df_metadata = df[meta_columns].T

    # TODO: is this value still relevant?
    prefix = list(df)[50].split(".")[0]

    for nr_round in range(1, 1000):
        columns = get_columns_per_round(nr_round, prefix)
        try:
            df_subset = df[columns]
        except KeyError:
            continue

        df_subset.columns = base_columns
        round_dict = df_subset.T.to_dict()
        for player in df.index:
            dict_data[player].append(round_dict[player])
            list_data.append(round_dict[player])

    for player in df.index:
        dict_df[player] = (
            pd.DataFrame(dict_data[player])
            .sort_values("player.applicant_id")
            .dropna(subset=["player.rating"])
        )

        fig = plt.figure(figsize=(4,4))
        plt.title(f'{player} - {file}')
        plt.hist(dict_df[player]['player.rating'])
        plt.show()

        # dict_df[player].to_csv(
        #     os.path.join(path_save, f"{player}_{file}.csv"), sep=",", encoding="utf-8"
        # )
        
    
    df_new = pd.DataFrame(list_data).sort_values("player.applicant_id")
    df_new.dropna(subset=["player.rating"], inplace=True)

    # df_new.to_csv(
    #         os.path.join(path_save, f'all_{file}.csv'),
    #         sep=';', encoding='utf-8'
    #     )

    df_rating = df_new.pivot(
        index=["player.applicant_id", "player.applicant_name"],
        columns="player.id_in_group",
        values=["player.rating"],
    )

    df_rating.reset_index(inplace=True)
    columns = fix_column_name(list(df_rating))
    df_rating = df_rating.droplevel(1, axis=1)

    df_rating.columns = columns
    df_rating = df_rating.rename(
        columns={"player.applicant_name": "name", "player.applicant_id": "Entry ID"}
    )


    # df_rating.to_csv(
    #     os.path.join(path_save, f"ratings_{file}.csv"), sep=",", encoding="utf-8"
    # )

    df_list.append(df_rating)

