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
from joblib import Parallel, delayed
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
    WRITE_NEW_NOT_PREPROCESSED_FILM_DF = False
    WRITE_NEW_PREPROCESSED_FILM_DF = False
    WRITE_NEW_DALE_CHALL_DF = False
    NUM_OF_PROCESSES = 4
    
    if WRITE_NEW_FILM_DF_PICKLE:
        movie_df = write_new_film_df_pickle(film_scripts_dir, out_path)
  
    movie_df = read_pickle(out_path, 'movie_scripts_df')
    
    if WRITE_NEW_NOT_PREPROCESSED_FILM_DF:
        movie_df['pre_preprocessing_dc_score'] = movie_df.movie_scripts.apply(calculate_dc_score)
        write_pickle(movie_df, out_path, 'movie_scripts_dc_before_preprocessing_df')
        
    movie_dc_before_preprocessing_df = read_pickle(out_path, 'movie_scripts_dc_before_preprocessing_df')
    
    """ 
        Preprocess the text of the film scripts
    """


    if WRITE_NEW_PREPROCESSED_FILM_DF:
        processed_script_df = preprocess_film_scripts_df(movie_dc_before_preprocessing_df)
        
        if WRITE_NEW_DALE_CHALL_DF:
            processed_script_df['preprocessed_dc_score'] = processed_script_df.film_scripts_processed.apply(calculate_dc_score)
            
        write_pickle(processed_script_df, out_path, 'movie_scripts_dc_after_preprocessing_df')
        
    preprocessed_df = read_pickle(out_path, 'movie_scripts_dc_after_preprocessing_df')
    
    # Just preprocessing film titles
    processed_rt_df = preprocess_rt_df(rotten_tomatoes_df)
    
    """ 
        Use Shapiro wilk Test to determine if it is normally distributed to determine whether to cut by mean
    """
    normally_distrubted = shapiro_wilk_test(preprocessed_df, "preprocessed_dc_score")
    
    """
        Merge film_scripts and RT scores and get count of unmatched 
    """
    merged_rt_and_scripts_df = preprocessed_df.merge(processed_rt_df, how='left', left_on=['movie_titles_stripped'], right_on=['movie_titles_stripped'])
    print(merged_rt_and_scripts_df.runtime.isnull().sum(axis = 0))
    
    """
        Get Genre Counts - note movies fall under several genres 
    """
    genre_count = count_movies_per_genre(merged_rt_and_scripts_df)
    
    
    
    # Print unmatched with rt df film scripts
    find_non_matching = merged_rt_and_scripts_df[merged_rt_and_scripts_df.runtime.isna()]["movie_titles_stripped"]
    
    #sys.exit()
    
    # Just seeing how many film scripts have the greatest percentage of uppercase letters
    #processed_script_df["percent_upper"] = processed_script_df.film_scripts_processed.apply(percentage)
    #find_too_many_upper = processed_script_df[processed_script_df.percent_upper > 20][["movie_titles_stripped","percent_upper"]]
    
    # TF-IDF
    # Apply the function to each row of the DataFrame using parallel processing
    """
        Binary Classification for DC score
        Change with appropriate halfway point POST preprocessing 
    """
    preprocessed_df['binary_dc'] = preprocessed_df['pre_preprocessing_dc_score'].apply(lambda x: 1 if x > 8 else 0)
 
    """ 
      Prep Binary RF Classification Model
    """
    # Add the results to the DataFrame
    #processed_script_df['tfidf_matrix']  =  processed_script_df.movie_scripts.apply(process_tfidf)
    # test_df = processed_script_df.head(5)
    # Split the dataset into a binary classification of "low" and "high" dc scores
    #parallelize_write_tfidf_pickles(processed_script_df.head(100), "movie_scripts", "tfidf_matrix", out_path, process_tfidf, NUM_OF_PROCESSES, 8)
    #tfidf_df = merge_pickle_dfs('tfidf_', 8, out_path)
    print("HERE") 

    #my_vec = merge_pickle_dfs('my_vec_', split_size, out_path)
    my_vec = count_vec_fun(
        preprocessed_df.movie_scripts, "vec", out_path, "tf-idf", 1, 1)
    print("HERE 2")
    # Save into pickle 
    #write_pickle(processed_script_df, out_path, 'movie_scripts_tfidf_df')
    fi_fun = model_test_train_fun(my_vec, preprocessed_df.binary_dc, 0.2, out_path, "vec")
    #build_binary_rf(tfidf_df, "tfidf_matrix", "binary_dc")
    
    """
        Tune RF Model
    """
    
    
    """
        Lime
    """