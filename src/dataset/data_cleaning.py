# -*- coding: utf-8 -*-
"""
Created on Fri May 28 20:49:31 2021

@author: maurol
"""

path = r"C:\Users\maurol\OneDrive\Dokumente\Python_Scripts\algorithmic-explanations\dataset\training"
file = "applications-website-up-to-20April-clean.csv"


import os
from datetime import datetime

import numpy as np
import pandas as pd


def make_date(string):
    return datetime.strptime(string, "%m.%d.%Y")


def get_age(date):
    diff = datetime.today() - date
    return int(diff.days / 365.0)


def replace_line_break(string):
    return string.replace("\r\n", " | ") if isinstance(string, str) else ""


df = pd.read_csv(os.path.join(path, file), sep=";", encoding="cp1252", index_col=0)
df["GPA"] = df["GPA"].str.replace(",", ".").astype(float)


df["birth.date"] = df["birth.date"].str.replace("/", ".")
df["birth.date"] = df["birth.date"].apply(make_date)
df["age"] = df["birth.date"].apply(get_age)


df["fav.subjects"] = df["fav.subjects"].apply(replace_line_break)
df["extracurriculars"] = df["extracurriculars"].apply(replace_line_break)
df["leadership"] = df["leadership"].apply(replace_line_break)
df["major"] = df["major"].apply(replace_line_break)




def get_rank_bins(rank):

    bins = [
        (1, 50),
        (51, 100),
        (101, 150),
        (151, 200),
        (201, 300),
        (301, 400),
        (401, 500),
        (501, 600),
        (601, 800),
        (801, 3000),
    ]

    # check if the value is a nan
    if rank != rank:
        return "unknown"

    elif "-" in rank:
        value = np.mean([int(value) for value in rank.split("-")])
    else:
        value = int(rank)

    for bin_tuple in bins:
        if bin_tuple[0] <= value <= bin_tuple[1]:
            if bin_tuple[1] == 3000:
                return f"{bin_tuple[0]} and above"
            return f"{bin_tuple[0]}-{bin_tuple[1]}"


df["college.rank_v2"] = df["college.rank"].apply(get_rank_bins)

# df['SAT'][df['SAT'].notnull()].precentile(5)

# ['name',
#  'email',
#  'essay',
#  'current.grade',
#  'college.text',
#  'college.rank',
#  'major',
#  'fav.subjects',
#  'extracurriculars',
#  'leadership',
#  'GPA',
#  'ACT',
#  'SAT',
#  'birth.date',
#  'state',
#  'gender',
#  'ethnicity',
#  'source.of.application',
#  'scholarship.rules',
#  'entry.ID',
#  'age']
