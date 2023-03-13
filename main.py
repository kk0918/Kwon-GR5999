#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 22:29:36 2023

@author: billykwon
"""
from utils import *
import pandas as pd 
import numpy as np
import time
import os
from readability import Readability


if __name__ == '__main__':
    rotten_tomatoes_dataset_path = os.getcwd() + '/data/rotten_tomatoes_movies.csv'
    film_scripts_dir = os.getcwd() + '/data/movie_scripts/'
   
    out_path = os.getcwd() + "/pickles/"
    # Create pickles folder if it doesn't exist
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
    # Df from RT CSVs 
    rotten_tomatoes_df = read_csv(rotten_tomatoes_dataset_path)
    
    """
        Set variables here to define whether we want to read or write new pickles
        These should be set to False unless adding new preprocessing steps 
    """
    WRITE_NEW_FILM_DF_PICKLE = False
    WRITE_NEW_PRE_PREPROCESSING_FILM_DF = False
    NUM_OF_PROCESSES = 8
    
    if(WRITE_NEW_FILM_DF_PICKLE):
        movie_df = write_new_film_df_pickle(film_scripts_dir, out_path)
  
    movie_df = read_pickle(out_path, 'movie_scripts_df')
    
    if(WRITE_NEW_PRE_PREPROCESSING_FILM_DF):
        movie_df['pre_preprocessing_dc_score'] = movie_df.movie_scripts.apply(calculate_dc_score)
        write_pickle(movie_df, out_path, 'movie_scripts_dc_before_preprocessing_df')
        
    movie_dc_before_preprocessing_df = read_pickle(out_path, 'movie_scripts_dc_before_preprocessing_df')
    
    processed_script_df = preprocess_film_scripts_df(movie_dc_before_preprocessing_df)
    processed_rt_df = preprocess_rt_df(rotten_tomatoes_df)
    
    merged_rt_and_scripts_df = processed_script_df.merge(processed_rt_df, how='left', left_on=['movie_titles_stripped'], right_on=['movie_titles_stripped'])
    print(merged_rt_and_scripts_df.runtime.isnull().sum(axis = 0))
    
    find_non_matching = merged_rt_and_scripts_df[merged_rt_and_scripts_df.runtime.isna()]["movie_titles_stripped"]