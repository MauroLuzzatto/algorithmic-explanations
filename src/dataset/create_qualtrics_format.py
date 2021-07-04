# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 09:59:12 2021

@author: maurol
"""
import os
from shutil import copyfile

import pandas as pd

from src.model.config import path_base
from src.model.DataConfig import DataConfig
from src.model.utils import create_folder


def rename_columns(df_qualtrics, mapping):

    for from_col, to_col in mapping:
        df_qualtrics.rename(
            columns={
                from_col: f'Text{to_col}', 
                from_col + '_plot': f"File{to_col}_current",
                },
            inplace=True
        )
        
        # df_qualtrics[f'Text{to_col}'] = df_qualtrics[from_col]
        # df_qualtrics[f"File{to_col}_current"] = df_qualtrics[from_col + '_plot']
    
    return df_qualtrics

def rename_images(df_qualtrics, mapping):
    
    path_output = create_folder(os.path.join(path_base, 'qualtrics', 'plot'))
    
    for field, to_col in mapping:
        
        column_from = f"File{to_col}_current"
        column_to = f"File{to_col}"
        
        print(column_from, column_to, field)

        for index, row in df_qualtrics.iterrows():
            
            path_image = os.path.join(path_reports, f'{field}',  'plot')
            
            if isinstance(row[column_from], str):
                from_image = os.path.join(path_image, row[column_from])
                to_image = os.path.join(path_output, row[column_to])
            
                copyfile(
                    from_image, 
                    to_image
                )
                print('--'*10)                
                print(path_image.split('reports')[-1])
                print(f'from: {row[column_from]}\n  to: {row[column_to]}')
                

path_reports = os.path.join(path_base, "reports")
path_load = os.path.join(path_base, "dataset", "training")
path_config = os.path.join(path_base, "src", "resources")


mapping = [
    ('academic', 'A'),
    ('demographic', 'B'),
    ('all', 'C'),
]

data = DataConfig(path_config)
data_config = data.load_config()
df_dataset = pd.read_csv(os.path.join(path_load, data_config['dataset']), sep=";", index_col=0)


df_qualtrics = pd.read_excel(os.path.join(path_base, 'AlgorithmicExplanation.xlsx'))
df_qualtrics = df_qualtrics.iloc[:df_dataset.shape[0]]
column_list = list(df_qualtrics)

df_qualtrics = df_qualtrics.drop(['Email'], axis=1)
df_qualtrics = df_qualtrics.drop(['TextA', "TextB", "TextC"], axis=1)
# , "FileA", "FileB", "FileC"


df_qualtrics['Entry ID'] = df_dataset.index
df_qualtrics = pd.merge(df_qualtrics,  df_dataset[['name', 'email']], left_on='Entry ID', right_index=True)

df_qualtrics['FirstName'] = df_qualtrics['name'].apply(lambda text: text.split()[0].capitalize())
df_qualtrics['LastName'] = df_qualtrics['name'].apply(lambda text: ' '.join(text.split()[1:]).capitalize())

df_qualtrics.rename(
    columns={
        'email': 'Email', 
        },
    inplace=True
)


for field in ['all', 'academic', 'demographic']:

    path = os.path.join(path_reports, f'{field}',  r'results\explanations.csv')

    df = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8",
        index_col=0,
        usecols = ['Entry ID', 'method', 'explanation', 'plot']
    )
    
    df[field] = df['method'] + '\n' + df['explanation']
    df[field + '_plot'] = df['plot']
    
    df = df[~df.index.duplicated(keep='first')]
    
    df_qualtrics = pd.merge(
        df_qualtrics,  
        df[[field, field + '_plot']], 
        left_index=True, 
        right_index=True,
        how='left'
    )



df_qualtrics = rename_columns(df_qualtrics, mapping)

rename_images(df_qualtrics, mapping)

for _, to_col in mapping:
    # replace linebreak with html break <br>
    df_qualtrics[f'Text{to_col}'] = df_qualtrics[f'Text{to_col}'].str.replace(r'\n', '<br>')
    df_qualtrics[f'Text{to_col}'] = df_qualtrics[f'Text{to_col}'].str.replace(r'\\n', '<br>')


df_output = df_qualtrics[column_list]
df_output.to_excel(
    os.path.join(path_base, 'qualtrics', 'AlgorithmicExplanation_with_explanations.xlsx'),
    index=False
)
