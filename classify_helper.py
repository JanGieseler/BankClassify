import logging
import yaml
from pathlib import Path

def assign_target_account(df):
    with open(Path('../data/target_account_regex.yaml')) as file:
        target_dict = yaml.full_load(file)
    
    for k, v in target_dict.items():
        
        indecies = df['desc'].str.match(k)
        if indecies.sum()>0:
            logging.debug(f'assigning {indecies.sum()} entries to {v}')
        df.loc[indecies, 'target account'] = v
    return df


def assign_label(df):
    """
    assign labels based on regular expression
    
    """
    with open(Path('../data/labeling_regex.yaml')) as file:
        regex_dict = yaml.full_load(file)
    
    for k, v in regex_dict.items():
        
        indecies = df['desc'].str.lower().str.match(k.lower())
        if indecies.sum()>0:
            logging.debug(f'labeling {indecies.sum()} entries as \'{v}\'')
        df.loc[indecies, 'class'] = v
    return df