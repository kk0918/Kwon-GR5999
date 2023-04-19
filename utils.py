# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 20:31:11 2022

@author:Billy K
"""
import nltk
import pandas as pd
from plotnine import *


def process_film_scripts(movie_dc_before_preprocessing_df, pickle_name, out_path, gre_words, PREPROCESS_FLAG=True):
    
    processed_script_df = movie_dc_before_preprocessing_df.copy()
    if PREPROCESS_FLAG:
        """
            Run all the film script preprocessing steps
        """
        processed_script_df = preprocess_film_scripts_df(movie_dc_before_preprocessing_df)
            
    
        processed_script_df['preprocessed_dc_score'] = processed_script_df.film_scripts_processed.apply(calculate_dc_score)
        
    """
        Add new feaures such as average sentence length
    """
    # Average sentence length
    processed_script_df["avg_sentence_len"] = processed_script_df.film_scripts_processed.apply(calc_avg_sentence_length)
    # Average word len
    processed_script_df["avg_word_len"] = processed_script_df.film_scripts_processed.apply(calc_avg_word_length)
    # Type token ratio
    processed_script_df["ttr"] = processed_script_df.film_scripts_processed.apply(calc_ttr)
    # Gre word count 
    processed_script_df["gre_words"] = processed_script_df.film_scripts_processed.apply(lambda x: calc_gre_words(x, gre_words))


    write_pickle(processed_script_df, out_path, pickle_name)
    
    return processed_script_df 

def calc_stats_before_preprocessing(df_in, gre_words, out_path, pickle_name):
    film_df = df_in.copy()
    # Average sentence length
    film_df["avg_sentence_len_pre"] = film_df.movie_scripts.apply(calc_avg_sentence_length)
    # Average word len
    film_df["avg_word_len_pre"] = film_df.movie_scripts.apply(calc_avg_word_length)
    # Type token ratio
    film_df["ttr_pre"] = film_df.movie_scripts.apply(calc_ttr)
    # Gre word count 
    film_df["gre_words_pre"] = film_df.movie_scripts.apply(lambda x: calc_gre_words(x, gre_words))

    write_pickle(film_df, out_path, pickle_name)

    return film_df
    
# Not using anymore... remove?
def get_feature_importance(tuned_rf_model, col_names):
    import numpy as np
    import matplotlib.pyplot as plt
    # Get feature importances and sort them in descending order
    feature_importances = tuned_rf_model.feature_importances_
    #feature_names = my_vec.columns
    sorted_idx = np.argsort(feature_importances)[::-1]
    
    # Create top 10 most important features DF 
    top_features_num = 10
    top_feature_df = pd.DataFrame(columns=['feature', 'feature_importance'])
    print("Top 10 features:")
    for i in range(10):
        top_feature_df = top_feature_df.append({'feature': col_names[sorted_idx[i]], 'feature_importance': feature_importances[sorted_idx[i]]}, ignore_index=True)
        print(f"{i+1}. {col_names[sorted_idx[i]]}: {feature_importances[sorted_idx[i]]:.4f}")
    
    return top_feature_df

def build_binary_rf(df, feature_col, target_col):
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from scipy.sparse import vstack
    # Concatenate all the TF-IDF matrices into a single matrix
    X = df[feature_col]
    y = df[target_col]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

    # Random Forest Classifer
    """ TODO: tune
    param_grid = {'n_estimators': np.arange(1,50,2),
                  'max_depth': np.arange(1,10,2)}    
    """
    
    random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=25)
    
    # Random Forest does not require scaled data 
    random_forest_classifier.fit(X_train, y_train)
    
    # Evaluate model on test data
    score = rf.score(X_test, y_test)
    print("Test accuracy: {:.3f}".format(score))


def process_tfidf(text):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk.corpus import stopwords
 
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Apply frequency-based feature selection
    # Double check max_features
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, max_features=1000)
    tfidf = tfidf_vectorizer.fit_transform(tokens)
    return  pd.DataFrame(tfidf).toarray()


def read_gre_csv(file_path):
    import csv
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
     
        vocab_list = []
        for row in reader:
            word = row[0].split()[0]
            vocab_list.append(word)
    return vocab_list

# Preprocess film scripts
def preprocess_film_scripts_df(df_in):
    import pandas as pd
    processed_df = df_in.copy()
    # Remove "Script_" from Title
    processed_df["movie_titles_stripped"] = processed_df.movie_titles.apply(clean_film_titles)
    
    # Manual Fixes
    processed_df.loc[processed_df["movie_titles_stripped"] == "terminator 2 judgement day", "movie_titles_stripped"] = 'terminator 2 judgment day'
    processed_df.loc[processed_df["movie_titles_stripped"] == "star wars phantom menace", "movie_titles_stripped"] = 'star wars episode i phantom menace'
    processed_df.loc[processed_df["movie_titles_stripped"] == "star wars revenge of sith", "movie_titles_stripped"] = 'star wars episode iii revenge of sith'
    processed_df.loc[processed_df["movie_titles_stripped"] == "star wars return of jedi", "movie_titles_stripped"] = 'star wars episode vi return of jedi'
    processed_df.loc[processed_df["movie_titles_stripped"] == "star wars empire strikes back", "movie_titles_stripped"] = 'star wars episode v empire strikes back'
    processed_df.loc[processed_df["movie_titles_stripped"] == "star wars a new hope", "movie_titles_stripped"] = 'star wars episode iv a new hope'
    processed_df.loc[processed_df["movie_titles_stripped"] == "star wars attack of clones", "movie_titles_stripped"] = 'star wars episode ii attack of clones'
    processed_df.loc[processed_df["movie_titles_stripped"] == "star wars force awakens", "movie_titles_stripped"] = 'star wars episode vii force awakens'
    processed_df.loc[processed_df["movie_titles_stripped"] == "walk to remember a", "movie_titles_stripped"] = 'a walk to remember'
    processed_df.loc[processed_df["movie_titles_stripped"] == "perfect world a", "movie_titles_stripped"] = 'a perfect world'
    processed_df.loc[processed_df["movie_titles_stripped"] == "hellraiser 3 hell on earth", "movie_titles_stripped"] = 'hellraiser iii hell on earth'
    processed_df.loc[processed_df["movie_titles_stripped"] == "american werewolf in london", "movie_titles_stripped"] = 'an american werewolf in london'
    processed_df.loc[processed_df["movie_titles_stripped"] == "halloween curse of michael myers", "movie_titles_stripped"] = 'halloween curse of michael myers halloween 6'
    processed_df.loc[processed_df["movie_titles_stripped"] == "jurassic park lost world", "movie_titles_stripped"] = 'lost world jurassic park'
    processed_df.loc[processed_df["movie_titles_stripped"] == "harold and kumar go to white castle", "movie_titles_stripped"] = 'harold kumar go to white castle'
    processed_df.loc[processed_df["movie_titles_stripped"] == "alien 3", "movie_titles_stripped"] = 'alien3'
    processed_df.loc[processed_df["movie_titles_stripped"] == "men in black 3", "movie_titles_stripped"] = 'men in black iii'
    processed_df.loc[processed_df["movie_titles_stripped"] == "ringu", "movie_titles_stripped"] = 'ringu ring'
    processed_df.loc[processed_df["movie_titles_stripped"] == "south park", "movie_titles_stripped"] = 'south park bigger longer uncut'
    processed_df.loc[processed_df["movie_titles_stripped"] == "sandlot kids", "movie_titles_stripped"] = 'sandlot'
    processed_df.loc[processed_df["movie_titles_stripped"] == "mission impossible ii", "movie_titles_stripped"] = 'mission impossible 2'
    processed_df.loc[processed_df["movie_titles_stripped"] == "gremlins 2", "movie_titles_stripped"] = 'gremlins 2 new batch'
    processed_df.loc[processed_df["movie_titles_stripped"] == "precious", "movie_titles_stripped"] = 'precious based on novel push by sapphire'
    
    
    # remove words in parentheses from film scripts since these are not spoken
    processed_df["film_scripts_processed"] = processed_df.movie_scripts.apply(remove_parens)
    # Remove text following INT. or EXT. up to first punctuation mark or uppcase followed by lower case or lower case
    processed_df["film_scripts_processed"] = processed_df.film_scripts_processed.apply(remove_int_ext)
    # remove all capital letters followed by a colon
    processed_df["film_scripts_processed"] = processed_df.film_scripts_processed.apply(remove_names)
    # Add space after punctuations in case words are stuck. this helps sent_tokenize in the next step
    processed_df["film_scripts_processed"] = processed_df.film_scripts_processed.apply(add_space_after_punctuations)
    # Remove stuck words
    processed_df["film_scripts_processed"] = processed_df.film_scripts_processed.apply(split_stuck_words)
    
    return processed_df

# Preprocess rotten tomatoes
def preprocess_rt_df(df_in):
    import pandas as pd
    processed_df = df_in.copy()
    # Remove "Script_" from Title
    processed_df["movie_titles_stripped"] = processed_df.movie_title.apply(clean_film_titles)
    return processed_df


def remove_names(sent_in):
    import re
    # Remove any text in all caps followed by a colon
    sent_clean = re.sub(r'[A-Z]+:', ' ', sent_in)
    # Remove apostrophes from words in all caps
    sent_clean = re.sub(r"\b[A-Z]+'?[A-Z]*\b", lambda match: match.group().replace("'", ""), sent_clean)
    # Remove ALL Caps of length at least 2 and leave the rest
    sent_clean = re.sub( r'[A-Z]{2,}(?![a-z])', ' ', sent_clean)
    return sent_clean

def calc_avg_sentence_length(col_in):
    sentences = nltk.sent_tokenize(col_in)
    word_count = [len(sentence.split()) for sentence in sentences]
    avg_sen_length = sum(word_count) / len(word_count)
    return avg_sen_length


def calc_avg_word_length(col_in):
    import re
    remove_punct = re.sub(r'[^a-zA-Z]', ' ', col_in)
    words = remove_punct.split()  
    total_len = sum(len(word) for word in words) 
    avg = total_len / len(words) 
    #print(f"Average word length: {avg:.2f}") 
    return avg

def calc_punctuation_frequency(col_in):
    import re
    import string
    # Punctuation includes !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    #punctuation_count = sum(col_in.count(char) for char in string.punctuation)
    punct_pattern = re.compile(f"[{re.escape(string.punctuation)}]+")
    punct_count = len(punct_pattern.findall(col_in))
    #print(f"Punctuation Count: {punct_count}")
    return punct_count

def calc_ttr(col_in):
    import re
    remove_punct = re.sub(r'[^a-zA-Z]', ' ', col_in)
    #words = nltk.word_tokenize(col_in)
    words = col_in.split()
    unique = set(words)
    num_unique = len(unique)
    total_words = len(words)
    
    ttr = num_unique / total_words 
    return ttr

def calc_gre_words(col_in, gre_words):
    import collections
    import re
    remove_punct = re.sub(r'[^a-zA-Z]', ' ', col_in)
    words = remove_punct.split()  
    gre_word_counts = collections.Counter(word.lower() for word in words if word.lower() in gre_words)
    return sum(gre_word_counts.values()) / len(words) 


def get_all_genres(df):
    genres_col = df['genres']
    genres_list = genres_col.str.split(',')
    genres_list_exploded = genres_list.explode().str.strip()
    unique_genres = set(genres_list_exploded)
    return unique_genres

def count_movies_per_genre(df_in):
    genre_df = df_in.copy()
    # Split the genres column by comma, strip whitespace, and create a new dataframe with binary indicator variables
    genre_df = genre_df['genres'].str.replace(',\s+', ',', regex=True).str.get_dummies(sep=',')
   
    # Compute the count of movies for each genre
    genre_counts = genre_df.sum().sort_values(ascending=False)

    return genre_counts

def count_dc_scores_per_genre(df_in):
    genre_df = df_in.copy()
    genre_df['genres'] = genre_df['genres'].str.split(', ')
    df_exploded = genre_df.explode('genres')
    
    genre_stats = df_exploded.groupby('genres').agg({'genres': 'count', 
                                                     'preprocessed_dc_score': ['mean', 'std'], 
                                                     'ttr': ['mean', 'std'],
                                                     'avg_sentence_len': ['mean', 'std'],
                                                     'avg_word_len': ['mean', 'std']})
    genre_stats.columns = genre_stats.columns.map('_'.join)
    genre_stats = genre_stats.rename(columns={'genres_count': 'count', 'preprocessed_dc_score_mean': 'mean_dc_score', 
                                              'preprocessed_dc_score_std': 'std_dc_score',
                                              'ttr_mean': 'type_token_ratio_mean',
                                              'ttr_std': 'type_token_ratio_std'})
    return genre_stats
    
def shapiro_wilk_test(df, column_name):
    import scipy.stats as stats
    import pandas as pd
    stat, p = stats.shapiro(df[column_name])
    
    normally_distributed = False

    if p > 0.05:
        print("Data is normally distributed")
        normally_distributed = True
    else:
        print("Data is not normally distributed")
        
    return normally_distributed

def plot_box_plots(df_in, x_col, title):
    import matplotlib.pyplot as plt
    labels=["Rotten", "Fresh"]
    groups = [0,1]
    fig, ax = plt.subplots(figsize=(11, 6))
    
    print(f'PLOTTING BOXPLOT {title}')
    
    i = 0
    for group in groups:
      mask = df_in['binary_fresh_rotten'] == group
      y = df_in.loc[mask, x_col] 
      
      print(f'GROUP {labels[i]}')
      print(y.describe())
      
      plt.subplot(1, 2, i+1)
      plt.boxplot(y, labels=[labels[i]])
      plt.title(labels[i])
      plt.xlabel("Cluster")
      plt.ylabel(title)
      i+=1
    
    
    
    plt.suptitle(f'Fresh vs Rotten and {title}')
    plt.show()

def plot_hist_scores(df, col_in, x_label, title):
    hist_plot = (
        ggplot(df, aes(x=col_in)) + 
        geom_histogram(color="black", bins=20, alpha=0.7, fill="#0072B2") +
        labs(title=title, x=x_label, y="Frequency"))
    print(hist_plot)
    
def plot_genre_dc_scores(df_in):
    genre_df = df_in.copy()
    genre_df = genre_df.reset_index().rename(columns={'index': 'genres'})

    genre_df = genre_df.sort_values(by="mean_dc_score")
    
    bar_plot = (
        ggplot(genre_df, aes(x="reorder(genres, mean_dc_score)", y="mean_dc_score")) + 
        geom_col(fill='#0000CC') + 
        coord_flip() +
        labs(title='Mean Dale-Chall Scores by Genre', x='Genre', y='Mean Score') + 
        theme_bw())
    
    print(bar_plot)
    
def plot_feature_importance(df_in, title):
    feature_df = df_in.copy()
    
    feature_plot = (
        ggplot(feature_df, aes(x='reorder(feature, feature_importance)', y='feature_importance',  fill='feature_importance')) + 
        geom_bar(stat='identity') + ggtitle(title) + 
        scale_fill_gradient(low="#0072b2", high="#02ed02") +
        coord_flip() +
        xlab('Feature') + 
        ylab('Importance')) 
    
    print(feature_plot)

def remove_parens(sent_in):
    import re
    sent_clean =  re.sub("[\(\[].*?[\)\]]", " ", sent_in)
    return sent_clean

def add_space_after_punctuations(sent_in):
    import re
    sent_clean = re.sub(r'([\.\?!])(?=[A-Z])', r'\1 ', sent_in)
    return sent_clean

# Split stuck words such as "systemcontrolled" or "Jedibecome" using wordninja
def split_stuck_words(text):
    import wordninja
    from nltk import tokenize
    
    sentences_out = []
    
    arr_sentences = tokenize.sent_tokenize(text)
    for sent in arr_sentences:
        my_sent_split = wordninja.split(sent)
            

        fixed_sentence = ""
        # Check if last char is punctuation so we can add to wordninja split
        if sent and sent[-1] and sent[-1][-1].strip() in ",.;:!?":
            last_word = sent[-1]
            last_char = last_word[-1]
            fixed_sentence = " ".join(my_sent_split) + last_char
        else:
            fixed_sentence = " ".join(my_sent_split)
        
            
        sentences_out.append(fixed_sentence.strip())

    return " ".join(sentences_out)

def remove_int_ext(text):
    import re
    #int_or_ext_pattern = '(INT\.|EXT\..*?)(?=\s*-?\s*[A-Z][a-z])'
    #int_or_ext_pattern = r'(INT\.|EXT\.)(?=\s*-?\s*[A-Z][a-z])|\b[A-Z][a-z]*([A-Z](?![a-z]))?\b'
    int_or_ext_pattern =  '(INT\.|EXT\.).*?(?=\s*[A-Z][a-z])|\b([A-Z][a-z]*[A-Z])\b'
    replaced_text = re.sub(int_or_ext_pattern, r'\2', text, flags=re.DOTALL)
    return replaced_text


# Custom helper functions for film scripts
def clean_film_titles(sent_in):
    import re
    # Remove "Script_" from Title
    sent_clean =  sent_in.replace("Script_", "")
    # Remove all '_' 
    sent_clean = sent_clean.replace("_", "")
    # Remove non alphanumberic characters and lowercase 
    sent_clean = re.sub("[^A-Za-z0-9]+", " ", sent_clean.lower()) 
    # If movie title ends with "the", move it to the beginning
    sent_clean = sent_clean.replace("the", "")
    # Remove all double spaces
    sent_clean = re.sub(' +', ' ', sent_clean)
    # Remove words between parens for movie titles
    sent_clean =  re.sub("[\(\[].*?[\)\]]", "", sent_clean)
    # Strip the sentences
    sent_clean = sent_clean.strip()
    return sent_clean

def calculate_dc_score(text_in):
    from readability import Readability
    r = Readability(text_in)
    dc_score = r.dale_chall().score
    return dc_score

def percentage(data):
    """Returns how many % of the 'data' returns 'True' for the given isupper func."""
    return sum(map(str.isupper,data)) / len(data)*100

def write_new_film_df_pickle(film_scripts_dir, out_path): 
    import os
    import pandas as pd
    # Get film scripts and put into dictionary
    movie_titles = []
    movie_scripts = []

    err_count = 0
    for filename in os.listdir(film_scripts_dir):
             name, file_extension = os.path.splitext(filename)
             try:
                 f = open(film_scripts_dir + filename, "r", encoding="utf-8")
                 text = f.read()

                 # Add to lists to create DF
                 movie_titles.append(name)
                 movie_scripts.append(text)
                 
             except Exception as e:
                 print(e)
                 err_count+=1
                 pass
    movie_df = pd.DataFrame(list(zip(movie_titles, movie_scripts)), columns=['movie_titles', 'movie_scripts'])
    write_pickle(movie_df, out_path, 'movie_scripts_df')

    print("Error count: " + str(err_count))

    return movie_df

# Custom helper functions not available from utils
def read_csv(file_path):
    import pandas as pd
    df = pd.read_table(file_path, sep=",")
    return df

# add direct split pickles function
def split_pickles(data, out_path, file_prefix, num_of_processes=8):
    import numpy as np
    import time
    import os
    data_split = np.array_split(data, num_of_processes)

    for i in range(num_of_processes):
        print(f'Writing to pickles on data_split {i}')
        start = time.time()
        write_pickle(data_split[i], out_path, f'{file_prefix}{i}')
        end = time.time()
        print("-------------------------------------------")
        print("PPID %s Completed in %s" % (os.getpid(), round(end-start, 2)))
    return

"""
   Write out sentiment pickles, defined by num_of_processes which I default set to 8
   results in /pickles folder with names such as sentiment_0.pk
"""
def parallelize_write_tfidf_pickles(data, col_in, col_out, out_path, func, num_of_processes=4, split_size=10):
    from multiprocessing import Pool
    import numpy as np
    import time
    import os
    data_split = np.array_split(data, split_size)

    for i in range(split_size):
        print(f'Running tfidf on data_split {i}')
        start = time.time()
        pool = Pool(num_of_processes)
        data_split[i][col_out] = pool.map(func, data_split[i][col_in])
        pool.close()
        pool.join()
        write_pickle(data_split[i], out_path, f'tfidf_{i}')
        end = time.time()
        print("-------------------------------------------")
        print("PPID %s Completed in %s" % (os.getpid(), round(end-start, 2)))
    return


"""
    Merge the resultant sentiment pickles. They were split up into 8 pickles 
    for smaller pickle sizes
"""
def merge_pickle_dfs(file_prefix, num_of_sentiment_dfs, file_path):
    import pandas as pd 
    sentiments = []
    for i in range(num_of_sentiment_dfs):
        sentiments.append(read_pickle(file_path, f'{file_prefix}{i}'))
    result = pd.concat(sentiments)
    return result

"""
    ***************** Preprocessing Helpers *****************
"""
# RT score DF
def preprocess_rt_scores_df(df_in, out_path, name_in):
    import pandas as pd
    processed_df = df_in.copy()
    # lowercase headers
    processed_df.columns = [header.lower() for header in processed_df.columns]
    processed_df['movie'] = processed_df['movie_title'].str.lower()
    # Only get year of review
    processed_df['release_year'] = pd.DatetimeIndex(processed_df['original_release_date']).year
    write_pickle(processed_df, out_path, name_in)
    return processed_df

# Box office DF
def preprocess_box_office_df(df_in, out_path, name_in):
    processed_df = df_in.copy()
    # lowercase headers
    processed_df.columns = [header.lower() for header in processed_df.columns]
    processed_df['movie'] = processed_df['movie'].str.lower()
    
    # Remove dollar signs and change to int type for domestic gross
    processed_df['domestic gross'] = processed_df['domestic gross'].str.replace(',', '').str.replace('$', '').astype(int)

    write_pickle(processed_df, out_path, name_in)
    return processed_df

# Sentiment df
def preprocess_sentiment_df(df_in, out_path, num_split):    
    import numpy as np
    import pandas as pd

    processed_df = df_in.copy()
    # lowercase headers
    processed_df.columns = [header.lower() for header in processed_df.columns]
    
    # Remove words between quotes
    processed_df["cleaned_review"] = processed_df.review.apply(remove_words_between_quotes)
    # Remove title 
    processed_df["cleaned_review"] = processed_df.apply(lambda x: remove_title(x.cleaned_review, x.movie), axis=1)
    # Clean text
    processed_df["cleaned_review"] = processed_df.cleaned_review.apply(clean_text_without_lower)
    # Remove stopwords
    processed_df["cleaned_review"] = processed_df.cleaned_review.apply(rem_sw)
    # Lowercase title
    processed_df["movie"] = processed_df.movie.str.lower()
    
    # Only get year of review
    processed_df['date_year'] = pd.DatetimeIndex(processed_df['date']).year
    # Drop columns not needed for analysis
    processed_df = processed_df.drop('publish', axis=1)

    # Split data into chunks
    data_split = np.array_split(processed_df, num_split)

    # Write pickles
    for i in range(num_split):
        write_pickle(data_split[i], out_path, f'preprocessed_{i}')
    
    return processed_df


"""
    Remove title of movie in review itself
    Example:
        Movie name: Hearts and Bones
        Before:
            Review: Hearts and Bones' dull visuals and undernourished....
        After:
            Review: dull visuals and undernourished....
"""
def remove_title(str_in, title):
    import re
    title_lower = title.lower()
    sent_clean = re.sub(r"\b{}\b".format(title_lower), "", str_in, flags=re.IGNORECASE)
    return sent_clean

"""
    Remove words in between quotes
    Words in between quotes are almost always a reference to the shortened
    version of the title or a name.
    Example: 
        Movie name: THE LAST BLACK MAN IN SAN FRANCISCO
        Before:
            Review: "Last" doesn't rely much on conventional narrative..
        After:
            Review: doesn't rely much on conventional narrative..
"""
def remove_words_between_quotes(str_in):
    import re
    # This removes the first and last quotes if the entire review is in quotes
    sent_stripped_first_and_last_quotes = re.sub(r'^"|"$', '', str_in)
    sent_clean = re.sub('".*?"', "", sent_stripped_first_and_last_quotes)
    return sent_clean


def clean_text_without_lower(str_in):
    import re
    sent_a_clean = re.sub("[^A-Za-z]+", " ", str_in) 
    return sent_a_clean

"""
    ***************** Utils file from lecture *****************
"""

def count_fun(var_in):
    tmp = var_in.split()
    return len(tmp)

def count_fun_unique(var_in):
    tmp = set(var_in.split())
    return len(tmp)

def clean_text(str_in):
    import re
    sent_a_clean = re.sub("[^A-Za-z]+", " ", str_in.lower()) 
    return sent_a_clean

def open_file(file_in):
    f = open(file_in, "r", encoding="utf-8")
    text = f.read()
    text = clean_text(text)
    f.close()
    return text

def file_reader(path_in):
    import os
    import pandas as pd
    the_data_t = pd.DataFrame()
    for root, dirs, files in os.walk(path_in, topdown=False):
       for name in files:
           try:
               label = root.split("/")[-1:][0]
               file_path = root + "/" + name
               text = open_file(file_path)
               if len(text) > 0:
                   the_data_t = the_data_t.append(
                       {"label": label, "body": text}, ignore_index=True)
           except:
               print (file_path)
               pass
    return the_data_t

def word_freq(df_in, col_in):
    import collections
    wrd_freq = dict()
    for topic in df_in.label.unique():
        tmp = df_in[df_in.label == topic]
        tmp_concat = tmp[col_in].str.cat(sep=" ")
        wrd_freq[topic] = collections.Counter(tmp_concat.split())
    return wrd_freq

def rem_sw(df_in):
    from nltk.corpus import stopwords
    sw = stopwords.words('english')
    sw.append("xp") #append a keyword to sw
    tmp = [word for word in df_in.split() if word not in sw]
    tmp = ' '.join(tmp)
    return tmp

def read_pickle(path_o, name_in):
    import pickle
    tmp_data = pickle.load(open(path_o + name_in + ".pk", "rb"))
    return tmp_data

def write_pickle(obj_in, path_o, name_in):
    import pickle
    pickle.dump(obj_in, open(path_o + name_in + ".pk", "wb"))
    return 0

def my_stem(var_in):
    #stemming
    from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()
    # example_sentence = "i was hiking down the trail towards by favorite fishing spot to catch lots of fishes"
    ex_stem = [ps.stem(word) for word in var_in.split()]
    ex_stem = ' '.join(ex_stem)
    return ex_stem

def sent_fun(str_in):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    senti = SentimentIntensityAnalyzer()
    ex = senti.polarity_scores(str_in)["compound"]
    return ex

def count_vec_fun(df_col_in, name_in, out_path_in, sw_in, min_in=1, max_in=1):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd
    from nltk.corpus import stopwords
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    # Ignore numbers from this
    pattern = r'\b[A-Za-z]{2,}\b'
    df_col_in = df_col_in.apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))
   
    if sw_in == "tf-idf":
        # Apply frequency-based feature selection
        cv = TfidfVectorizer(token_pattern=pattern, max_df=0.8, min_df=2, ngram_range=(min_in, max_in))
    else:
        cv = CountVectorizer(ngram_range=(min_in, max_in))
    xform_data = pd.DataFrame(cv.fit_transform(df_col_in).toarray()) #be careful
    #takes up memory when force from sparse to dense
    xform_data.columns = cv.get_feature_names_out()
    return xform_data, cv

def extract_embeddings_pre(df_in, num_vec_in, path_in, filename):
    #from gensim.models import Word2Vec
    import pandas as pd
    #from gensim.models import KeyedVectors
    import pickle
    #import gensim
    import gensim.downloader as api
    my_model = api.load(filename)
    
    #my_model = KeyedVectors.load_word2vec_format(
    #    filename, binary=True) 
    #my_model = Word2Vec(df_in.str.split(),
    #                    min_count=1, vector_size=300)
    #word_dict = my_model.wv.key_to_index
    #my_model.most_similar("calculus")
    #my_model.similarity("trout", "fish")
    def get_score(var):
        import numpy as np
        tmp_arr = list()
        try:
            for word in var:
                tmp_arr.append(list(my_model.get_vector(word)))
        except:
            tmp_arr.append(np.zeros(num_vec_in).tolist())
            #print(word)
            pass
        return np.mean(np.array(tmp_arr), axis=0)
    tmp_out = df_in.str.split().apply(get_score)
    tmp_data = tmp_out.apply(pd.Series).fillna(0)
    pickle.dump(my_model, open(path_in + "embeddings.pk", "wb"))
    pickle.dump(tmp_data, open(path_in + "embeddings_df.pk", "wb" ))
    return tmp_data

def extract_embeddings_domain(df_in, num_vec_in, path_in):
    #domain specific, train out own model specific to our domains
    from gensim.models import Word2Vec
    import pandas as pd
    import numpy as np
    import pickle
    model = Word2Vec(
        df_in.str.split(), min_count=1,
        vector_size=num_vec_in, workers=3, window=5, sg=0)
    wrd_dict = model.wv.key_to_index
    def get_score(var):
        try:
            tmp_arr = list()
            for word in var:
                tmp_arr.append(list(model.wv[word]))
        except:
            pass
        return np.mean(np.array(tmp_arr), axis=0)
    tmp_out = df_in.str.split().apply(get_score)
    tmp_data = tmp_out.apply(pd.Series).fillna(0)
    #model.wv.save_word2vec_format(path_in + "embeddings_domain.pk")
    pickle.dump(model, open(path_in + "embeddings_domain.pk", "wb"))
    pickle.dump(tmp_data, open(path_in + "embeddings_df_domain.pk", "wb" ))
    
    return tmp_data, wrd_dict

def pca_fun(df_in, exp_var_in, path_o, name_in):
    #pca
    from sklearn.decomposition import PCA
    import pandas as pd
    dim_red = PCA(n_components=exp_var_in)
    red_data = pd.DataFrame(dim_red.fit_transform(df_in))
    exp_var = sum(dim_red.explained_variance_ratio_)
    print ("Explained variance:", exp_var)
    write_pickle(dim_red, path_o, name_in)
    return red_data

def sparse_pca_fun(df_in, target_component, path_o, name_in):
    #pca for large sparse dataset
    from sklearn.decomposition import TruncatedSVD
    import pandas as pd
    dim_red = TruncatedSVD(n_components=target_component, random_state=42)
    red_data = dim_red.fit_transform(df_in)
    exp_var = sum(dim_red.explained_variance_ratio_)
    print ("Explained variance:", exp_var)
    write_pickle(dim_red, path_o, name_in)
    return red_data

def tune_rf_model(df_in, label_in, X_train, y_train, large_param_grid):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.model_selection import GridSearchCV
    import numpy as np
    import pandas as pd 
    """
        TUNE
    """
    print("TUNING")
    # Random Forest Classifer
    
    if large_param_grid:
        param_grid = {'n_estimators': np.arange(20,100,2),
               'max_depth': np.arange(2,20,1)}
    else:
        param_grid = {'n_estimators': np.arange(20,70,10),
                  'max_depth': np.arange(2,8,1)}    
    
    random_forest_classifier = GridSearchCV(RandomForestClassifier(random_state=25), param_grid=param_grid, cv=10)
    
    
    # Random Forest does not require scaled data 
    random_forest_classifier.fit(X_train, y_train)
    
    print("best mean cross-validation score: {:.3f}".format(random_forest_classifier.best_score_))
    print("best parameters: {}".format(random_forest_classifier.best_params_))
    
    # Return the best model
    best_rf_model = random_forest_classifier.best_estimator_
    print("FINISHED TUNING")
    
    return best_rf_model
    
    #return random_forest_classifier
    
def model_test_train_fun(rf_in, df_in, label_in, path_in, xform_in,
                         X_test, y_test):
    #TRAIN AN ALGO USING my_vec
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import confusion_matrix
    import numpy as np
    import pandas as pd 
    
    y_pred = rf_in.predict(X_test)
    y_pred_proba = pd.DataFrame(rf_in.predict_proba(X_test))
    y_pred_proba.columns = rf_in.classes_
    
    metrics = pd.DataFrame(precision_recall_fscore_support(
        y_test, y_pred, average='weighted'))
    metrics.index = ["precision", "recall", "fscore", "none"]
    print (metrics)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    the_feats = read_pickle(path_in, xform_in)
    try:
        #feature importance
        fi = pd.DataFrame(rf_in.feature_importances_)
        fi["feat_imp"] = the_feats.get_feature_names()
        fi.columns = ["feat_imp", "feature"]
        perc_propensity = len(fi[fi.feat_imp > 0]) / len(fi)
        print ("percent features that have propensity:", perc_propensity)
    except:
        print ("can't get features")
        pass
    
    return fi

def use_lime(rf_in, df_in, label_in, X_train, X_test, y_train, y_test):
    from sklearn.model_selection import train_test_split
    import lime
    import lime.lime_tabular
    import numpy as np
    import pandas as pd
    print("LIME")

    class_names = ['0', '1']
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns.values, class_names=class_names)

    exp = explainer.explain_instance(X_test.values[0], rf_in.predict_proba, num_features=len(X_train.columns))
    
    # Print the top features contributing to the predicted class
    print('Explanation for class %s' % class_names[y_test.values[0]])
    for i in range(len(exp.as_list())):
        print(exp.as_list()[i])

def grid_fun(df_in, label_in, test_size_in, path_in, xform_in, grid_d, cv_in):
    #TRAIN AN ALGO USING my_vec
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_fscore_support
    import pandas as pd 
    from sklearn.model_selection import GridSearchCV
    my_model = RandomForestClassifier(random_state=123)
    my_grid_model = GridSearchCV(my_model, param_grid=grid_d, cv=cv_in)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df_in, label_in, test_size=test_size_in, random_state=42)
    
    my_grid_model.fit(X_train, y_train)
    
    print ("Best perf", my_grid_model.best_score_)
    print ("Best perf", my_grid_model.best_params_)
    
    my_model = RandomForestClassifier(
        **my_grid_model.best_params_, random_state=123)
    
    #lets see how balanced the data is
    agg_cnts = pd.DataFrame(y_train).groupby('label')['label'].count()
    print (agg_cnts)
    
    my_model.fit(X_train, y_train)
    write_pickle(my_model, path_in, "rf")
    
    y_pred = my_model.predict(X_test)
    y_pred_proba = pd.DataFrame(my_model.predict_proba(X_test))
    y_pred_proba.columns = my_model.classes_
    
    metrics = pd.DataFrame(precision_recall_fscore_support(
        y_test, y_pred, average='weighted'))
    metrics.index = ["precision", "recall", "fscore", "none"]
    print (metrics)
    
    the_feats = read_pickle(path_in, xform_in)
    try:
        #feature importance
        fi = pd.DataFrame(my_model.feature_importances_)
        fi["feat_imp"] = the_feats.get_feature_names()
        fi.columns = ["feat_imp", "feature"]
        perc_propensity = len(fi[fi.feat_imp > 0]) / len(fi)
        print ("percent features that have propensity:", perc_propensity)
    except:
        print ("can't get features")
        pass
    return fi