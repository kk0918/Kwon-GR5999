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
    #WRITE_NEW_DALE_CHALL_DF = False
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
        processed_script_df = preprocess_film(movie_dc_before_preprocessing_df, 'movie_scripts_dc_after_preprocessing_df')

    preprocessed_df = read_pickle(out_path, 'movie_scripts_dc_after_preprocessing_df')
        
    # Just preprocessing film titles
    processed_rt_df = preprocess_rt_df(rotten_tomatoes_df)
    
    
    """
        Medians and describe
    """
    print("Describe pre_preprocessing_dc_score")
    preprocessed_df["pre_preprocessing_dc_score"].describe()
    print("Before Preprocessed DC MEDIAN: " + str(preprocessed_df["pre_preprocessing_dc_score"].median()))
         
    print("Describe preprocessed_dc_score")
    preprocessed_df["preprocessed_dc_score"].describe()
    print("After Preprocessed DC MEDIAN: " + str(preprocessed_df["preprocessed_dc_score"].median()))
    
    
    """
        Plot images
    """
    plot_hist_dc_scores(preprocessed_df, "pre_preprocessing_dc_score", "Distribution of Raw Film Script Dale-Chall Scores")
    plot_hist_dc_scores(preprocessed_df, "preprocessed_dc_score", "Distribution of Preprocessed Film Script Dale-Chall Scores")

    
    """ 
       Use Shapiro wilk Test to determine if it is normally distributed to determine whether to cut by mean
    """
    normally_distributed = shapiro_wilk_test(preprocessed_df, "preprocessed_dc_score")
   
    median_score_dc_score = preprocessed_df["preprocessed_dc_score"].median()
    print("MEDIAN OF MERGED: " + str(median_score_dc_score))
    print("GREATER THAN MEDIAN: " + str(len(preprocessed_df[preprocessed_df['preprocessed_dc_score'] >= median_score_dc_score])))
    print("LESS THAN MEDIAN: " + str(len(preprocessed_df[preprocessed_df['preprocessed_dc_score'] < median_score_dc_score])))
 

    """
       Binary Classification for DC score
       Change with appropriate halfway point POST preprocessing 
    """
    
    preprocessed_df['binary_dc'] = preprocessed_df['preprocessed_dc_score'].apply(lambda x: 1 if x >= median_score_dc_score else 0)
   
      
    """
       Tune RF Model
    """
  
    my_vec = count_vec_fun(
        preprocessed_df.film_scripts_processed, "vec", out_path, "tf-idf", 1, 1)
      
    print("TUNING RF MODEL")
    tuned_rf_model = tune_rf_model(my_vec, preprocessed_df.binary_dc, 0.2)
    
    
    """
        Feature importance
    """
    # Get Feature importance a
    top_feature_df = get_feature_importance(tuned_rf_model, my_vec)
    # Visualize the feature importances
    plot_feature_importance(top_feature_df)
    
    fi_fun = model_test_train_fun(tuned_rf_model, my_vec, preprocessed_df.binary_dc, 0.2, out_path, "vec")


    """
        Top 20 features
    """
    top_feature_labels = top_feature_df.feature


    # Create new DF with additional features 
    df_with_top_20_words = my_vec[top_feature_labels]
    df_with_only_features = preprocessed_df[["avg_sentence_len", "punctuation_count", "ttr", ]]
    df_top_20_with_text = pd.concat([df_with_only_features, df_with_top_20_words], axis=1)

    df_target = preprocessed_df[["binary_dc"]]

    """
        Create Model
    """
    
    print("TUNING RF MODEL with Top 20")
    top_20_rf_model = tune_rf_model(df_top_20_with_text, df_target.binary_dc, 0.2)
    top_20_run = model_test_train_fun(top_20_rf_model, df_top_20_with_text, df_target.binary_dc, 0.2, out_path, "vec")

    
    
    """
        Merge film_scripts and RT scores and get count of unmatched 
    """
    merged_rt_and_scripts_df = preprocessed_df.merge(processed_rt_df, how='left', left_on=['movie_titles_stripped'], right_on=['movie_titles_stripped'])
    print(merged_rt_and_scripts_df.runtime.isnull().sum(axis = 0))
    # Print unmatched with rt df film scripts
    find_non_matching = merged_rt_and_scripts_df[merged_rt_and_scripts_df.runtime.isna()]["movie_titles_stripped"]
    
    # DROP all with same movie name - this is because for film scripts the only source we have is the name
    merged_rt_and_scripts_df.drop_duplicates(subset=['movie_titles_stripped'], keep=False, inplace=True)
    # DROP all rows with no runtime because this indicates they did not match
    merged_rt_and_scripts_df = merged_rt_and_scripts_df.dropna(subset=['runtime'])
    print(merged_rt_and_scripts_df.runtime.isnull().sum(axis = 0))
    """
        Get Genre Counts - note movies fall under several genres 
    """
    dc_and_genre_count = count_dc_scores_per_genre(merged_rt_and_scripts_df)
    
    # Plots
    plot_genre_dc_scores(dc_and_genre_count)
    
        
    # Just seeing how many film scripts have the greatest percentage of uppercase letters
    #processed_script_df["percent_upper"] = processed_script_df.film_scripts_processed.apply(percentage)
    #find_too_many_upper = processed_script_df[processed_script_df.percent_upper > 20][["movie_titles_stripped","percent_upper"]]
    


    
    """
        Lime
    """
    #use_lime(tuned_rf_model, my_vec, preprocessed_df.binary_dc, 0.2)