# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 19:28:44 2020

@author: mauro
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_cleaning_functions import (
    get_age,
    get_rank_bins,
    make_date,
    replace_line_break,
)
from sklearn.preprocessing import MultiLabelBinarizer

from src.model.config import path_base


class DataPreprocessing(object):
    def __init__(self, path_data, file_name, path_save):
        self.path_data = path_data
        self.file_name = file_name
        self.path_save = path_save

    def load(self):
        """
        load data and covert it to a dataframe
        """

        files = [
            training_file
            for training_file in os.listdir(self.path_data)
            if self.file_name in training_file
        ]

        df_list = []
        for training_file in files:
            df_list.append(
                pd.read_csv(
                    os.path.join(self.path_data, training_file),
                    sep=";",
                    encoding="utf-8-sig",
                )
            )

        self.df = pd.concat(df_list, axis=0)

        self.df.set_index("Entry ID", inplace=True, verify_integrity=True, drop=True)
        print(self.df.shape)
        return self.df

    def load_real_data(self, index_name):
        """
        load the data and conduct simple cleaning steps

        Returns:
            pd.DataFrame: dataframe containing the data

        """
        column = "entry.ID"
        self.df = pd.read_csv(
            os.path.join(self.path_data, self.file_name),
            sep=";",
            encoding="utf-8",
        )
        self.df.set_index(column, inplace=True, verify_integrity=True, drop=True)
        print(self.df.shape)
        self.data_cleaning()
        self.df.index.rename(index_name, inplace=True)
        return self.df

    def data_cleaning(self):
        """
        clean and pre-process the value in the raw data

        Returns:
            None.

        """
        self.df["GPA"] = self.df["GPA"].str.replace(",", ".").astype(float)
        self.df["birth.date"] = self.df["birth.date"].str.replace("/", ".")
        self.df["birth.date"] = self.df["birth.date"].apply(make_date)
        self.df["age"] = self.df["birth.date"].apply(get_age).astype(int)

        self.df["fav.subjects"] = self.df["fav.subjects"].apply(replace_line_break)
        self.df["extracurriculars"] = self.df["extracurriculars"].apply(
            replace_line_break
        )
        self.df["leadership"] = self.df["leadership"].apply(replace_line_break)
        self.df["major"] = self.df["major"].apply(replace_line_break)
        self.df["ethnicity"] = self.df["ethnicity"].apply(replace_line_break)

    def get_categorial_features(self, categorical_encoding):
        """
        convert list of features into categorical values
        """
        for feature in categorical_encoding:
            self.df = self.categorial_feature(self.df, feature)

    def get_one_hot_encoding(self, one_hot_encoding, add_feature_name=True):
        """
        convert list of features into one hot encoded values
        """
        for feature in one_hot_encoding:

            self.df[feature].fillna("", inplace=True)
            self.df[feature] = self.df[feature].apply(lambda text: text.split(" | "))

            mlb = MultiLabelBinarizer()

            df_one_hot = pd.DataFrame(
                mlb.fit_transform(self.df.pop(feature)),
                columns=mlb.classes_,
                index=df.index,
            )

            if add_feature_name:
                df_one_hot.columns = [
                    f"{feature} - {value}" for value in list(df_one_hot)
                ]

            self.df = self.df.join(df_one_hot)

    def get_count_encoding(self, count_encoding):
        """
        Count the number of values in the list
        """

        for feature in count_encoding:
            self.df[feature].fillna("", inplace=True)
            self.df[feature] = self.df[feature].apply(
                lambda text: text.replace("Other", f"Other - {feature}")
            )
            self.df[feature] = self.df[feature].apply(lambda text: text.split(" | "))
            self.df[feature] = self.df[feature].apply(
                lambda values: len(values) if values[0] != "" else 0
            )

    def drop_features(self, exclude_features):
        """
        drop columns from dataframe
        """
        print(exclude_features)
        self.df = self.df.drop(exclude_features, axis=1)

    def categorial_feature(self, df, feature, drop=True):
        """
        Convert categorical feature column, save the mapping into a json

        Args:
            df (TYPE): DESCRIPTION.
            feature (TYPE): DESCRIPTION.
            drop (TYPE, optional): DESCRIPTION. Defaults to True.

        Returns:
            df (TYPE): DESCRIPTION.

        """

        df[feature] = df[feature].astype("category")
        df["{}_cat".format(feature)] = df[feature].cat.codes.astype(int)

        category_mapping = self.get_category_mapping(df, feature)
        self.save_mapping(category_mapping, feature)

        if drop:
            df = df.drop([feature], axis=1)
            df = df.rename(columns={"{}_cat".format(feature): feature})
        return df

    @staticmethod
    def get_category_mapping(df, feature):
        return pd.Series(
            df[feature].values, index=df["{}_cat".format(feature)]
        ).to_dict()

    def save_mapping(self, category_mapping: dict, name: str) -> None:
        """
        Save a json file with the mapping between the category number
        and the category value

        Args:
            category_mapping (dict): DESCRIPTION.
            name (str): DESCRIPTION.

        Returns:
            None: DESCRIPTION.

        """

        with open(
            os.path.join(self.path_save, "mapping", f"{name}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(category_mapping, f, ensure_ascii=False, indent=4)

    def text_to_score(self, essay_column):
        """
        TODO: Load from the ratings
        """

        def score_essay(df):
            return np.random.uniform(0, 10, df.shape[0])

        score = score_essay(self.df)
        self.df["Essay score"] = score
        self.drop_features([essay_column])

    def save_csv(self, df, name=None):
        """
        save the processed data back to csv
        """
        if not name:
            name = file_name.split(".")[0] + "_processed.csv"

        df.to_csv(os.path.join(self.path_save, name), sep=";", encoding="utf-8-sig")
        print(df.shape)
        print(os.path.join(self.path_save, name))

    def fill_missing_values(self, fill_na, approach=None):
        """
        Fill in missing nan values (currently with median)

        Args:
            fill_na (TYPE): DESCRIPTION.

        Returns:
            None.

        """
        for feature in fill_na:
            
            plt.title(feature)
            self.df[feature].hist(figsize=(4,4))
            plt.show()
            
            if approach == "min":
                value = self.df[feature].min()
            elif isinstance(approach, float):
                value = self.df[feature].quantile(approach)

            else:
                value = self.df[feature].median()
                
            print(feature, value)
            self.df[feature].fillna(value, inplace=True)

    def rename_features(self, rename_list):
        """
        Rename column to the defined new name

        Args:
            rename_features_list (TYPE): DESCRIPTION.

        Returns:
            None.

        """
        rename_dict = {feature[0]: feature[1] for feature in rename_list}
        self.df.rename(
            columns=rename_dict,
            inplace=True,
        )

    def convert_rank_to_bins(self, rank_list):
        """
        Convert the college rank into bins

        Args:
            rank_list (TYPE): DESCRIPTION.

        Returns:
            None.

        """
        for feature in rank_list:
            self.df[feature] = self.df[feature].apply(get_rank_bins)

    def main(
        self,
        categorical_encoding,
        one_hot_encoding,
        count_encoding,
        exclude_features,
        rank_list,
        essay_column,
    ):
        self.convert_rank_to_bins(rank_list)
        self.get_categorial_features(categorical_encoding)
        self.get_one_hot_encoding(one_hot_encoding)
        self.get_count_encoding(count_encoding)
        # self.text_to_score(essay_column)
        self.drop_features(exclude_features)
        return self.df


if __name__ == "__main__":

    rename_list = [
        ("ethnicity", "Ethnicity"),
        ("major", "Major"),
        ("gender", "Gender"),
        ("current.grade", "Grade"),
        ("fav.subjects", "Favorite subject"),
        ("leadership", "Number of leadership experiences"),
        ("extracurriculars", "Number of extracurricular activities"),
        ("state", "State of residence"),
        ("college.rank", "College rank"),
        ("age", "Age"),
    ]

    categorical_encoding = [
    ]
    one_hot_encoding = [
        "Major",
        "Grade",
        "Favorite subject",
        "College rank",
    ]

    count_encoding = [
        "Number of leadership experiences",
        "Number of extracurricular activities",
    ]

    exclude_features = [
        "source.of.application",
        "scholarship.rules",
        "birth.date",
        "college.text",
        "essay",
        "Ethnicity",
        "Gender",
        "State of residence",
        "Age"
    ]

    fill_na = [
        "ACT",
        "SAT",
    ]

    rank_list = ["College rank"]

    essay_column = "essay"
    index_name = "Entry ID"

    path_rating = os.path.join(path_base, r"dataset", "ratings", "post_processed_v2")
    path_save = os.path.join(path_base, r"dataset", "training")
    path_data = path_save

    file_name = "All-applications-clean_full_data.csv"
    save_name = f"{file_name}_postprocessed.csv"

    dataset = DataPreprocessing(path_data, file_name, path_save)
    df = dataset.load_real_data(index_name)

    dataset.rename_features(rename_list)
    dataset.fill_missing_values(fill_na, approach=0.05)

    print(list(df))

    df_processed = dataset.main(
        categorical_encoding,
        one_hot_encoding,
        count_encoding,
        exclude_features,
        rank_list,
        essay_column,
    )

    df_ratings = pd.read_csv(
        os.path.join(path_rating, "ratings.csv"),
        sep=";",
        encoding="utf-8-sig",
        index_col=0,
    )
    
    
    df_final = df_processed.merge(
        df_ratings, how="outer", left_index=True, right_index=True
    )

    df_final["Essay grade"] = df_final[
        [col for col in list(df_final) if "essay.player.rating" in col]
    ].mean(axis=1)
    
    
    # for _, row in df_final.iterrows():
    #     print(row[['FirstName', 'LastName', 'Email', 'name']].values)
        
        
    df_final.dropna(subset=['FirstName'], inplace=True)
    df_final = df_final.drop(['name', 'Token'], axis=1)
    
    dataset.save_csv(df_final, name=save_name)

    
