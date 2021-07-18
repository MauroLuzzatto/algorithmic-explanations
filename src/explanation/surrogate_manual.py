# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 23:08:55 2021

@author: maurol
"""


import os
from typing import Dict

import graphviz
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


# TRUE False

f = """
digraph Tree {
node [shape=box, style="rounded", color="black", fontname=helvetica] ;
edge [fontname=helvetica] ;
0 [label="Number of leadership experiences < 1.5"] ;
1 [label="Essay grade < 5.25"] ;
0 -> 1 [labeldistance=2.5, labelangle=45, headlabel="True"] ;
3 [label="GPA < 3.46"] ;
1 -> 3 [headlabel="True"] ;
7 [label="Average rating:
 3.4"] ;
3 -> 7 [headlabel="True     "] ;
8 [label="Average rating:
 4.31"] ;
3 -> 8 [headlabel="False     "] ;
4 [label="Grade is not College/University Grad Student"] ;
1 -> 4 [headlabel="False     "] ;
13 [label="Average rating:
 5.07"] ;
4 -> 13 [headlabel="True     "];
14 [label="Average rating:
 6.56"] ;
4 -> 14 [headlabel="False   "];
2 [label="Essay grade < 4.75"] ;
0 -> 2 [labeldistance=2.5, labelangle=-45, headlabel="False"] ;
5 [label="Number of extracurricular activities < 3.5"] ;
2 -> 5 [headlabel="True     "] ;
11 [label="Average rating:
 4.66"] ;
5 -> 11 [headlabel="True     "] ;
12 [label="Average rating:
 5.68"] ;
5 -> 12 [headlabel="False     "] ;
6 [label="GPA < 3.65"] ;
2 -> 6 [headlabel="False"] ;
9 [label="Average rating:
 5.58"] ;
6 -> 9 [headlabel="True     "] ;
10 [label="Average rating:
 6.87"] ;
6 -> 10 [headlabel="              False"] ;
}
"""

def run():

    plot_name = 'surrogate_sample_True_sparse_False.png'
    path_plot = r"C:\Users\maurol\OneDrive\Dokumente\Python_Scripts\algorithmic-explanations\reports\all\plot"
    
    
    name, extension = os.path.splitext(plot_name)
    
    graphviz.Source(
        f,
        filename=os.path.join(path_plot, name),
        format=extension.replace('.', ''),
    ).view()
    
    with open(
        os.path.join(path_plot, "{}.dot".format(plot_name)), "w"
    ) as file:
        file.write(f)