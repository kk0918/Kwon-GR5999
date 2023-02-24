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


if __name__ == '__main__':
    rotten_tomatoes_dataset_path = os.getcwd() + '/data/rotten_tomatoes_movies.csv'
    film_scripts_dir = os.getcwd() + '/data/movie_scripts/'
   
    out_path = os.getcwd() + "/pickles/"
    # Create pickles folder if it doesn't exist
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
    # Df from RT CSVs 
    rotten_tomatoes_df = read_csv(rotten_tomatoes_dataset_path)
    # Get film scripts and put into dictionary
    movie_dict = {}
    i = 0;
    for filename in os.listdir(film_scripts_dir):
             name, file_extension = os.path.splitext(filename)
             try:
                 f = open(film_scripts_dir + filename, "r", encoding="utf-8")
                 text = f.read()
                 movie_dict[name] = text
             except Exception as e:
                 print(e)
                 i+=1
                 pass
             
    print("Error count" + str(i))
