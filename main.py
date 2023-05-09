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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold

if __name__ == '__main__':
    rotten_tomatoes_dataset_path = os.getcwd() + '/data/rotten_tomatoes_movies.csv'
    film_scripts_dir = os.getcwd() + '/data/movie_scripts/'
    gre_dataset_path = os.getcwd() + '/data/GRE_Master_Wordlist.csv'
   
    out_path = os.getcwd() + "/pickles/"
    # Create pickles folder if it doesn't exist
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
    # Df from RT CSVs 
    rotten_tomatoes_df = read_csv(rotten_tomatoes_dataset_path)
    
    # Read GRE CSV
    gre_words = read_gre_csv(gre_dataset_path)
    
    """
        Set variables here to define whether we want to read or write new pickles
        These should be set to False unless adding new preprocessing steps 
    """
    WRITE_NEW_FILM_DF_PICKLE = False
    WRITE_NEW_NOT_PREPROCESSED_FILM_DF = False
    WRITE_NEW_PREPROCESSED_FILM_DF = False
    TUNE_RF_MODEL = False
    TUNE_RF_MODEL_WEIGHT = False
    TUNE_BC_MODEL = False
    TUNE_ADDTL_WITH_TOP_20 = False
    CALC_BEFORE_PREPROCESSING = False
    TESTING_SPLIT = 0.2
    RANDOM_STATE = 42
    NUM_OF_PROCESSES = 4
    
    if WRITE_NEW_FILM_DF_PICKLE:
        movie_df = write_new_film_df_pickle(film_scripts_dir, out_path)
  
    movie_df = read_pickle(out_path, 'movie_scripts_df')
    
    if WRITE_NEW_NOT_PREPROCESSED_FILM_DF:
        movie_df['pre_preprocessing_dc_score'] = movie_df.movie_scripts.apply(calculate_dc_score)
        movie_df = process_film_scripts(movie_df, 'movie_scripts_dc_before_preprocessing_df', out_path, gre_words, False)
        
    movie_dc_before_preprocessing_df = read_pickle(out_path, 'movie_scripts_dc_before_preprocessing_df')
    
    """ 
        Preprocess the text of the film scripts
    """

    if WRITE_NEW_PREPROCESSED_FILM_DF:
        processed_script_df = process_film_scripts(movie_dc_before_preprocessing_df, 'movie_scripts_dc_after_preprocessing_df', out_path, gre_words, True)

    preprocessed_df = read_pickle(out_path, 'movie_scripts_dc_after_preprocessing_df')

    # Just preprocessing film titles
    processed_rt_df = preprocess_rt_df(rotten_tomatoes_df)

 
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
    
    # Make Fresh or Rotten no in between 
    merged_rt_and_scripts_df['binary_fresh_rotten'] = merged_rt_and_scripts_df['tomatometer_status'].apply(lambda x: 1 if x == "Certified-Fresh" or x == "Fresh" else 0)
  
    print(merged_rt_and_scripts_df['binary_fresh_rotten'].value_counts())
    #'Certified-Fresh', 'Rotten', 'Fresh'
    
    """
        Calculate stats on text before preprocessing to compare before and after
    """
    if CALC_BEFORE_PREPROCESSING:
        pre_df = calc_stats_before_preprocessing(merged_rt_and_scripts_df, gre_words, out_path, "movie_scripts_no_preprocessing_stats_df")
        
    pre_df = read_pickle(out_path, 'movie_scripts_no_preprocessing_stats_df')
    
   
    """
        Dale-Chall Medians and describe
    """
    print("Describe pre_preprocessing_dc_score")
    print(pre_df["pre_preprocessing_dc_score"].describe())
    print("Before Preprocessed DC MEDIAN: " + str(pre_df["pre_preprocessing_dc_score"].median()))
         
    print("Describe preprocessed_dc_score")
    print(merged_rt_and_scripts_df["preprocessed_dc_score"].describe())
    print("After Preprocessed DC MEDIAN: " + str(merged_rt_and_scripts_df["preprocessed_dc_score"].median()))
    
    """
       avg_sentence_len Medians and describe
    """
    print("Describe avg_sentence_len")
    print(pre_df["avg_sentence_len_pre"].describe())
    print("Before Preprocessed avg_sentence_len_pre MEDIAN: " + str(pre_df["avg_sentence_len_pre"].median()))
         
    print("Describe preprocessed avg_sentence_len")
    print(merged_rt_and_scripts_df["avg_sentence_len"].describe())
    print("After Preprocessed avg_sentence_len MEDIAN: " + str(merged_rt_and_scripts_df["avg_sentence_len"].median()))
    
    """
       avg_word_len Medians and describe
    """
    print("Describe avg_word_len_pre")
    print(pre_df["avg_word_len_pre"].describe())
    print("Before Preprocessed avg_word_len MEDIAN: " + str(pre_df["avg_word_len_pre"].median()))
         
    print("Describe preprocessed avg_word_len")
    print(merged_rt_and_scripts_df["avg_word_len"].describe())
    print("After Preprocessed avg_word_len MEDIAN: " + str(merged_rt_and_scripts_df["avg_word_len"].median()))
    
    """
       ttr Medians and describe
    """
    print("Describe ttr_pre")
    print(pre_df["ttr_pre"].describe())
    print("Before Preprocessed ttr MEDIAN: " + str(pre_df["ttr_pre"].median()))
         
    print("Describe preprocessed ttr")
    print(merged_rt_and_scripts_df["ttr"].describe())
    print("After Preprocessed ttr MEDIAN: " + str(merged_rt_and_scripts_df["ttr"].median()))
    
    """
       gre_words Medians and describe
    """
    print("Describe gre_words_pre")
    print(pre_df["gre_words_pre"].describe())
    print("Before Preprocessed gre_words_pre MEDIAN: " + str(pre_df["gre_words_pre"].median()))
         
    print("Describe preprocessed ")
    print(merged_rt_and_scripts_df["gre_words"].describe())
    print("After Preprocessed gre_words MEDIAN: " + str(merged_rt_and_scripts_df["gre_words"].median()))
    
    
    """
        Plot images
    """
    plot_hist_scores(merged_rt_and_scripts_df, "pre_preprocessing_dc_score", "Dale-Chall Scores", "Distribution of Raw Film Script Dale-Chall Scores")
    plot_hist_scores(merged_rt_and_scripts_df, "preprocessed_dc_score","Dale-Chall Scores", "Distribution of Preprocessed Film Script Dale-Chall Scores")
 
    # avg_sentence_len
    plot_hist_scores(pre_df, "avg_sentence_len_pre", "Avg Sentence Length", "Distribution of Raw Film Script Avg Sentence Length")
    plot_hist_scores(merged_rt_and_scripts_df, "avg_sentence_len","Avg Sentence Length", "Distribution of Preprocessed Film Script Avg Sentence Length")
 
    # avg_word_len
    plot_hist_scores(pre_df, "avg_word_len_pre", "Avg Word Length", "Distribution of Raw Fillm Script Avg Word Length")
    plot_hist_scores(merged_rt_and_scripts_df, "avg_word_len","Avg Word Length", "Distribution of Preprocessed Film Script Avg Word Length")
 
    # ttr
    plot_hist_scores(pre_df, "ttr_pre", "Type Token Ratio", "Distribution of Raw Film Script Type Token Ratio")
    plot_hist_scores(merged_rt_and_scripts_df, "ttr","Type Token Ratio", "Distribution of Preprocessed Film Script Type Token Ratio")
 
    # gre_words
    plot_hist_scores(pre_df, "gre_words_pre", "GRE Word Percentage", "Distribution of Raw Film Script GRE Word Percentage")
    plot_hist_scores(merged_rt_and_scripts_df, "gre_words", "GRE Word Percentage", "Distribution of Preprocessed Film Script GRE Word Percentage")
 
    """ 
       Use Shapiro wilk Test to determine if it is normally distributed 
    """
    normally_distributed = shapiro_wilk_test(merged_rt_and_scripts_df, "preprocessed_dc_score")
   
    """
        Target variable = RT fresh or rotten
        Without lexical features, only movie features 
        
        content_rating, genres, runtime
    """
    
    # CONTENT_RATING one-hot
    # ['content_rating_G', 'content_rating_NC17', 'content_rating_NR','content_rating_PG-13', 'content_rating_R'],
    rating_onehot = pd.get_dummies(merged_rt_and_scripts_df['content_rating'], prefix='content_rating')
    rating_onehot = rating_onehot.drop('content_rating_PG', axis=1)
    merged_rt_and_scripts_df = pd.concat([merged_rt_and_scripts_df, rating_onehot], axis=1)

    # GENRES
    all_genres = get_all_genres(merged_rt_and_scripts_df)
    for genre in all_genres:
        merged_rt_and_scripts_df[f'genre_{genre}'] = merged_rt_and_scripts_df['genres'].apply(lambda x: genre in x).astype(int)

 
    df_with_features_and_target_no_lexical = merged_rt_and_scripts_df.loc[:, ['binary_fresh_rotten', 'runtime' ] + list(merged_rt_and_scripts_df.columns[-24:])]

    # Check size after drop na
    df_with_features_and_target_no_lexical = df_with_features_and_target_no_lexical.dropna()
    
    df_with_no_lexical_features = df_with_features_and_target_no_lexical.loc[:, 'runtime':]
    
    
    X_train_initial, X_test_initial, y_train_initial, y_test_initial = train_test_split(
        df_with_no_lexical_features, df_with_features_and_target_no_lexical.binary_fresh_rotten, test_size=TESTING_SPLIT, random_state=RANDOM_STATE)
   
    if TUNE_RF_MODEL:
        tuned_rf_no_lex_model = tune_rf_model(df_with_no_lexical_features, df_with_features_and_target_no_lexical.binary_fresh_rotten,
                                       X_train_initial, y_train_initial, True)
        write_pickle(tuned_rf_no_lex_model, out_path, 'top_rf_no_lexical_model')
        
    tuned_rf_no_lex_model = read_pickle(out_path, 'top_rf_no_lexical_model')
        
    fi_no_lexical_fun = model_test_train_fun(tuned_rf_no_lex_model, df_with_no_lexical_features, df_with_features_and_target_no_lexical.binary_fresh_rotten, out_path, "vec",
                                      X_test_initial, y_test_initial)
    
    # Do it with class_weight='balanced_subsample'    
    if TUNE_RF_MODEL_WEIGHT:        
        tuned_rf_no_lex_model_weight = tune_rf_model(df_with_no_lexical_features, df_with_features_and_target_no_lexical.binary_fresh_rotten,
                                       X_train_initial, y_train_initial, True, 'balanced_subsample')
        write_pickle(tuned_rf_no_lex_model_weight, out_path, 'top_rf_no_lexical_model_weight')
        
    tuned_rf_no_lex_model_weight = read_pickle(out_path, 'top_rf_no_lexical_model_weight')
    print("CLASS WEIGHT \n")
    fi_no_lexical_fun_weight = model_test_train_fun(tuned_rf_no_lex_model_weight, df_with_no_lexical_features, df_with_features_and_target_no_lexical.binary_fresh_rotten, out_path, "vec",
                                          X_test_initial, y_test_initial)
    
    # TRY BAG-CART
    if TUNE_BC_MODEL:
        tuned_bc_no_lex_model = tune_bag_cart_model(df_with_no_lexical_features, df_with_features_and_target_no_lexical.binary_fresh_rotten,
                                   X_train_initial, y_train_initial, True)
        write_pickle(tuned_bc_no_lex_model, out_path, 'top_bc_no_lexical_model')
                     
    tuned_bc_no_lex_model = read_pickle(out_path, 'top_bc_no_lexical_model')
    
    print("BAG CART \n")
    bc_fi_no_lexical_fun = model_test_train_fun(tuned_bc_no_lex_model, df_with_no_lexical_features, df_with_features_and_target_no_lexical.binary_fresh_rotten, out_path, "vec",
                                      X_test_initial, y_test_initial)
    

    # Feature importance 
    importances_initial = tuned_rf_no_lex_model.feature_importances_

    for feature, importance in zip(df_with_no_lexical_features.columns, importances_initial):
        print(f"{feature}: {importance}")
        
    # Plot feature importance    
    top_feature_no_lex_df = get_feature_importance(tuned_rf_no_lex_model, df_with_no_lexical_features.columns)
    # Visualize the feature importances
    plot_feature_importance(top_feature_no_lex_df, 'Top 10 Feature Importances with RT Film Features')
    
    
    """
        Target variable = RT fresh or rotten
        With lexical features
   
    """
    print("\n USING LEXICAL FEATURES \n")
    df_with_features_and_target = merged_rt_and_scripts_df.loc[:, ['binary_fresh_rotten', 'runtime', 'preprocessed_dc_score', "gre_words", "avg_sentence_len", "avg_word_len", "ttr" ] + list(merged_rt_and_scripts_df.columns[-24:])]
    # Check size after drop na
    df_with_features_and_target = df_with_features_and_target.dropna()
    
    df_with_only_features = df_with_features_and_target.loc[:, 'runtime':]
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        df_with_only_features, df_with_features_and_target.binary_fresh_rotten, test_size=TESTING_SPLIT, random_state=RANDOM_STATE)
   
    if TUNE_RF_MODEL:
        tuned_rf_model = tune_rf_model(df_with_only_features, df_with_features_and_target.binary_fresh_rotten,
                                       X_train, y_train, True)
        write_pickle(tuned_rf_model, out_path, 'top_rf_fresh_model')
        
    tuned_rf_model = read_pickle(out_path, 'top_rf_fresh_model')
        
    fi_fun = model_test_train_fun(tuned_rf_model, df_with_only_features, df_with_features_and_target.binary_fresh_rotten, out_path, "vec",
                                      X_test, y_test)
    
    # Do it with class weight    
    if TUNE_RF_MODEL_WEIGHT:
        tuned_rf_model_weight = tune_rf_model(df_with_only_features, df_with_features_and_target.binary_fresh_rotten,
                                       X_train, y_train, True, 'balanced_subsample')
        write_pickle(tuned_rf_model_weight, out_path, 'top_rf_model_weight')
        
    tuned_rf_lex_model_weight = read_pickle(out_path, 'top_rf_model_weight')
    print("LEXICAL CLASS WEIGHT \n")
    fi_fun_weight = model_test_train_fun(tuned_rf_lex_model_weight, df_with_only_features, df_with_features_and_target.binary_fresh_rotten, out_path, "vec",
                                      X_test, y_test)
    
    # TRY BAG-CART
    if TUNE_BC_MODEL:
        tuned_bc_lex_model = tune_bag_cart_model(df_with_only_features, df_with_features_and_target.binary_fresh_rotten,
                                   X_train, y_train, True)
        write_pickle(tuned_bc_lex_model, out_path, 'top_bc_lexical_model')
                     
    tuned_bc_lex_model = read_pickle(out_path, 'top_bc_lexical_model')
    
    print("LEXICAL BAG CART \n")
    bc_fi_lexical_fun = model_test_train_fun(tuned_bc_lex_model, df_with_only_features, df_with_features_and_target.binary_fresh_rotten, out_path, "vec",
                                      X_test, y_test)
    
    # Get Feature importance 
    importances = tuned_rf_model.feature_importances_

    for feature, importance in zip(df_with_only_features.columns, importances):
        print(f"{feature}: {importance}")
        
    # Plot feature importance    
    top_feature_lex_df = get_feature_importance(tuned_rf_model, df_with_only_features.columns)
    plot_feature_importance(top_feature_lex_df, 'Top 10 Feature Importances with Lexical Features')
    
    """
        Get Genre Counts - note movies fall under several genres 
    """
    dc_and_genre_count = count_dc_scores_per_genre(merged_rt_and_scripts_df)
     
    # Plots
    plot_genre_dc_scores(dc_and_genre_count)
           
    
    """
        Plots - boxplos
    """    
    # Dale-chall
    plot_box_plots(merged_rt_and_scripts_df, "preprocessed_dc_score", "Dale-Chall Score")
    
    # Runtime
    plot_box_plots(merged_rt_and_scripts_df, "runtime", "Runtime")
    
    # TTR
    plot_box_plots(merged_rt_and_scripts_df, "ttr", "Type-Token Ratio")
    
    # gre_words
    plot_box_plots(merged_rt_and_scripts_df, "gre_words", "GRE Word Percentage")
    
    # avg_sentence_len
    plot_box_plots(merged_rt_and_scripts_df, "avg_sentence_len", "Avg Sentence Length")
    
    # avg_word_len
    plot_box_plots(merged_rt_and_scripts_df, "avg_word_len", "Avg Word Length")
    
    # Bar chart for content_rating and fresh/rotten count
    fig, ax = plt.subplots(figsize=(6, 6))
    
    grouped_content = merged_rt_and_scripts_df.groupby(['content_rating', 'binary_fresh_rotten'])
    content_counts = grouped_content.size().unstack()
    
    success_film_df = merged_rt_and_scripts_df['binary_fresh_rotten'] == 1
    rotten_film_df = merged_rt_and_scripts_df['binary_fresh_rotten'] == 0
    # Set the x-axis and y-axis labels
    ax.set_xlabel('Content Rating')
    ax.set_ylabel('Number of Movies')
    
    # Set title
    ax.set_title('Number of Fresh and Rotten Movies by Content Rating')
    unique_contents = merged_rt_and_scripts_df['content_rating'].unique()

    x_position = range(len(unique_contents))
    ax.set_xticks(x_position)
    ax.set_xticklabels(unique_contents)
    
    # Create a grouped bar chart
    bar_width = 0.4
    
    for i in range(len(unique_contents)):
        rating = unique_contents[i]
        ax.bar(i - bar_width/2, content_counts[1][rating], width=bar_width, label='Fresh', color="green")
        ax.bar(i + bar_width/2, content_counts[0][rating], width=bar_width, label='Rotten', color="red")
    ax.legend(['Fresh', 'Rotten'])

    # Show the chart
    plt.show()

    # Extra data on most common words
    find_most_common_words(merged_rt_and_scripts_df)
    
    
    
    
    