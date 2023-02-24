#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 22:29:36 2023

@author: billykwon
"""
import pandas as pd 
import numpy as np
import time
import os


if __name__ == '__main__':
    rotten_tomatoes_dataset_path = os.getcwd() + './data/rotten_tomatoes_movies.csv'
    film_scripts_dir = os.getcwd() + './data/movie_scripts'
   
    out_path = os.getcwd() + "/pickles/"
    # Create pickles folder if it doesn't exist
    if not os.path.exists(out_path):
        os.makedirs(out_path)