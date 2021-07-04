# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 10:37:54 2021

@author: maurol
"""
import os
import pandas as pd

from src.model.config import path_base

path_base = r'C:\Users\maurol\OneDrive\Dokumente\Python_Scripts\algorithmic-explanations'
folder_name = "algorithmic_explanations"
folder_name_v2 = "algorithmic_explanations_v2"

path_ratings = os.path.join(path_base, "dataset", "ratings")


file = 'essay_all_apps_wide-2021-05-22.csv'

df = pd.read_csv(os.path.join(path_ratings, folder_name, file))

file = 'essay_batch1_P2P3.csv'

df_v2 = pd.read_csv(os.path.join(path_ratings, folder_name_v2, file))
