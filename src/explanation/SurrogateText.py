# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 11:41:28 2020

@author: mauro
"""

import numpy as np


class SurrogateText(object):
    """"""

    def __init__(self, text: str, model: object, X: np.array, feature_names: list):
        """


        Args:
            text (TYPE): DESCRIPTION.
            model (TYPE): DESCRIPTION.
            X (TYPE): DESCRIPTION.
            feature_names (TYPE): DESCRIPTION.

        Returns:
            None.

        """

        self.text = text
        self.model = model
        self.X = X
        self.feature_names = feature_names

        self.children_left = self.model.tree_.children_left
        self.children_right = self.model.tree_.children_right
        self.feature = self.model.tree_.feature
        self.threshold = self.model.tree_.threshold
        self.values = self.model.tree_.value.reshape(self.model.tree_.value.shape[0], 1)

    def get_text(self):
        """


        Returns:
            TYPE: DESCRIPTION.

        """

        paths = self.get_paths()

        texts = []
        for key in paths:
            string = self.get_rule(paths[key])
            score = self.values[key][0]
            texts.append(self.text.format(score, string))

        return "\n".join([text + "." for text in texts])

    def get_paths(self):
        """


        Returns:
            None.

        """
        # Leaves
        leave_id = self.model.apply(self.X)

        paths = {}
        for leaf in np.unique(leave_id):
            path_leaf = []
            self.find_path(0, path_leaf, leaf)
            paths[leaf] = np.unique(np.sort(path_leaf))

        return paths

    def find_path(self, node_numb, path, x):
        """


        Args:
            node_numb (TYPE): DESCRIPTION.
            path (TYPE): DESCRIPTION.
            x (TYPE): DESCRIPTION.

        Returns:
            bool: DESCRIPTION.

        """
        path.append(node_numb)
        if node_numb == x:
            return True
        left = False
        right = False

        if self.children_left[node_numb] != -1:
            left = self.find_path(self.children_left[node_numb], path, x)

        if self.children_right[node_numb] != -1:
            right = self.find_path(self.children_right[node_numb], path, x)

        if left or right:
            return True

        path.remove(node_numb)
        return False

    def get_rule(self, path):
        """


        Args:
            path (TYPE): DESCRIPTION.

        Returns:
            TYPE: DESCRIPTION.

        """

        mask = ""
        for index, node in enumerate(path):
            # check if we are not in the leaf
            if index != len(path) - 1:
                # if under the threshold
                if self.children_left[node] == path[index + 1]:
                    mask += "'{}' is smaller or equal to {:.2f}\t".format(
                        self.feature_names[self.feature[node]], self.threshold[node]
                    )
                else:
                    mask += "'{}' is larger than {:.2f}\t".format(
                        self.feature_names[self.feature[node]], self.threshold[node]
                    )
        values = [criterion for criterion in mask.split("\t") if criterion]

        return " and ".join(values)
