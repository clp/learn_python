# -*- coding: utf-8 -*-
# Using ~/anaconda3/bin/python: Python 3.6.0 :: Anaconda 4.3.0 (64-bit)
# Using Python 3.4.5 :: Anaconda 4.3.0 (64-bit), since Tue2017_0710

"""nltk_ex25.py

   Analyze text files from stackoverflow.com data using NLTK.

   NLTK is the Natural Language Toolkit, a suite of 
   Python s/w, data sets, and documents to support
   the development of Natural Language Processing.

   The input file has one question and multiple answers for that
   question.

   A goal is to find answers in stackoverflow that might be
   good, but 'hidden' because they have low scores.
   Look for such answers based on the terms found in high-score
   answers, that are also found in low-score answers.
   Save the o/p to a file for further evaluation & processing.

   Usage:
     pydoc  nltk_ex25

     See fga_find_good_answers.py for an example that calls
     the functions in this module.

     Set the value of num_hi_score_terms in the calling program.
     It determines how many hi-score terms are used to search
     text in low-score answers.  A bigger number will cause the
     program to run longer, and might find more 'hidden' answers
     that could be valuable.

   Options:
     TBD

------
"""

# Ref: https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import pprint
from six.moves import range
import os
import csv

import config as cf
#TBD import logging


# ----------------------------------------------------------
# TBD Hard-coded i/p file to use for temporary debugging.
#
# a_fname = 'pid_231767.csv'  # Based on o/p from fga*.py?
# ----------------------------------------------------------


log_msg = cf.log_file + ' - Start logging for ' + os.path.basename(__file__)
cf.logger.info(log_msg)

# Process the words of each input line.

from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords

#TBD nltk.download()
    # Note: Did this download once & it took 90 minutes;
    # it downloaded text data sets, including stop words.
    # Are there any issues btwn py2 & py3 for nltk?
    # S/w installed at ~/nltk_data/.
    # You can use the web site & only d/l some parts if you don't need all.
    # Add code to check for current files on local disk before d/l.


def convert_text_to_words( raw_q_a ):
    """ Convert a raw stackoverflow question or answer
    to a string of words.
    The input is a single string (a raw ques or ans entry), and
    the output is a single string (a preprocessed ques or ans).
    """
    # 1. Remove HTML
    q_a_text = BeautifulSoup(raw_q_a, "lxml").get_text()

    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", q_a_text)

    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()

    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))

    # Add more noise terms to stopwords.
    stops.add('th')

    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]

    # 6. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join( meaningful_words ))


def clean_raw_data(a_fname, progress_msg_factor, qagroup_df, tmpdir ):
    """ For all answers: Clean and parse the training set bodies.")
    """
    # Get the number of bodies based on that column's size
    num_bodies = qagroup_df["Body"].size
    cf.logger.info("Number of bodies: " + str(num_bodies))

    clean_ans_bodies_l = []
    qagroup_df["CleanBody"] = ""

    # Build a list that holds the cleaned text from each answer's body field.
    # Use it to find terms that match terms found in hi-score Answers.
    for i in range( 0, num_bodies ):
        clean_body = convert_text_to_words( qagroup_df["Body"][i] )
        clean_ans_bodies_l.append(clean_body)
        #
        # Add new column to Answers df.
        qagroup_df.loc[i,"CleanBody"] = clean_body
        # Print a progress message; default is for every 10% of i/p data handled.
        if( (i+1) % progress_msg_factor == 0 ):
            clean_q_a = convert_text_to_words( qagroup_df["Body"][i] )
            #D cf.logger.debug("Body %d of %d" % ( i+1, num_bodies ))
            #D cf.logger.debug('  Original text: ' + qagroup_df['Body'][i])
            cf.logger.debug('  Cleaned text:  ' + clean_q_a)

    #TBR cf.pdb.set_trace()
    # Write cleaned bodies to a file, one body per line, for visual review.
    outfile = tmpdir + a_fname + '.out'
    if os.path.exists(outfile):
        os.rename(outfile, outfile + '.bak')
    with open(outfile, 'w') as f:
        f.write('\n'.join(clean_ans_bodies_l))
    return clean_ans_bodies_l


def make_bag_of_words(clean_ans_bodies_l):
    """Collect and count words (or phrases = ngrams) in the text.

    To use ngrams instead of single words in the analysis,
    specify 'ngram_range = (3,5)' (for example) in the args for
    CountVectorizer.

    Using this arg & pattern will include 1-letter
    words: 'token_pattern = r'\b\w+\b'.
    """
    cf.logger.debug("Creating the bag of words for word counts ...")
    from sklearn.feature_extraction.text import CountVectorizer

    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool

    # TBD, Fails w/ MemErr with max_features at 5000; ok at 100-200.

    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 ngram_range = (3,5), \
                                 # token_pattern = r'\b\w+\b', \
                                 max_features = 200)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.

    train_data_features = vectorizer.fit_transform(clean_ans_bodies_l)

    # Numpy arrays are easy to work with, so convert the result to an
    # array

    train_data_features = train_data_features.toarray()

    # Note, w/ q9999 data, got (727, 1550):
    cf.logger.info('Bag shape: rows (num of records), cols (num of features):')
    cf.logger.info(train_data_features.shape)

    # Take a look at the words in the vocabulary
    vocab = vectorizer.get_feature_names()

    # Print counts of each word in vocab.
    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)

    return (vocab, dist)


def sort_save_vocab(suffix, vocab, dist, a_fname, tmpdir):
    """
    Sort and save vocabulary data to a list and to a file
    with a specified suffix.

    For each item in the bag of words, print the vocabulary word and
    the number of times it appears in the training set
    """
    count_tag_l = []
    word_freq_d = {}
    for tag, count in zip(vocab, dist):
        count_tag_l.append((count, tag))
        #D word_freq_d[tag] = count

    # Sort the list of tuples by count.
    words_sorted_by_count_l = sorted(count_tag_l, key=lambda x: x[0])

    # Write sorted vocab to a file.
    outfile = tmpdir + a_fname + suffix
    if os.path.exists(outfile):
        os.rename(outfile, outfile + '.bak')
    with open(outfile, 'w') as f:
        for count, word in words_sorted_by_count_l:
            print(count, word, file=f)
    return words_sorted_by_count_l 


def sort_answers_by_score(numlines, qagroup_df):
    """
    Build a sorted dataframe of answers.
    """
    score_df = qagroup_df.sort_values(['Score'])
    score_df = score_df[['Id', 'Score']]

    # Compute the number of records to use for computation and display.
    rec_selection_ratio = 0.10  # Default 0.01?
    num_selected_recs = int(numlines * rec_selection_ratio)
    if num_selected_recs < 6:
        num_selected_recs = 5
    log_msg = "  rec_selection_ratio,  number of selected recs: " + str(rec_selection_ratio) + ", " + str(num_selected_recs)
    cf.logger.info(log_msg)
    cf.logger.info('Lowest scoring Answers:')
    cf.logger.info(score_df.head())
    #D print(score_df.head(num_selected_recs), '\n')
    cf.logger.info('Highest scoring Answers:')
    cf.logger.info(score_df.tail())
    #D print(score_df.tail(num_selected_recs), '\n')

    return score_df, num_selected_recs


def find_freq_words(top, score_top_n_df, num_selected_recs, progress_msg_factor, qagroup_df):
    """
    Build a list of the most frequent terms found in answers.
    """
    top_n_bodies = []
    score_df_l = []
    # Convert dataframe to list of Id's, to get the body of each Id.
    if top: # Get the tail of the list, highest-score items.
        score_df_l = score_top_n_df['Id'].tail(num_selected_recs).tolist()
    else:  # Get the head of the list, lowest-score items.
        score_df_l = score_top_n_df['Id'].head(num_selected_recs).tolist()
    df8 = qagroup_df.set_index('Id')
    progress_count = 0
    for i in score_df_l:
        progress_count += 1
        top_n_bodies.append( convert_text_to_words( df8["Body"][i] ))
        # Print a progress message for every 10% of i/p data handled.
        if( (progress_count+1) % progress_msg_factor == 0 ):
            clean_q_a = convert_text_to_words( df8["Body"][i] )
            #D cf.logger.debug("Body for Id %d " % ( i))
            #D cf.logger.debug('  Original text:\n' + df8['Body'][i][:70])
            cf.logger.debug('  Partial slice of cleaned text:\n' + clean_q_a[:70])
    return top_n_bodies 


def search_for_terms(words_sorted_by_count_main_l, clean_ans_bodies_l, num_hi_score_terms, qagroup_df):
    """
    Read each answer and save any terms that it has in common
    with (high frequency) text from the high-score answers.

    Save those answers for further investigation.
    """

    clean_ans_bodies_df = pd.DataFrame(clean_ans_bodies_l)
    cf.logger.debug("clean_ans_bodies_l, top:") 
    cf.logger.debug(clean_ans_bodies_l[:1]) 
    cf.logger.debug("clean_ans_bodies_l, bottom:")
    cf.logger.debug(clean_ans_bodies_l[-1:]) 

    #D Use short list cab_l for testing:
    #D cab_l = ['step isprimenumber call', 'foo step isprimenumber call bar', 'yield abc def ghi generators long first string', 'generator', 'foo next function bar', 'short list foo bar baz']

    #D cab_df = pd.DataFrame(cab_l)

    #TBF
    # p.1. Separate the terms from each other so it is clear what was found
    #  and what was not found in the A's being checked.
    #  Should tmp_df not be a df?  Would a list of lists be better?

    qagroup_df["HiScoreTerms"] = ""
    qagroup_df["hstCount"] = 0

    #D print("\nTerms from hi-score Answers.")
    cf.logger.info("Terms from hi-score Answers.")
    #TBD, Maybe collect these "count,w" data into a single list,
    #   & write it in one step to the log file.
    for count,w in words_sorted_by_count_main_l[-num_hi_score_terms:]:
        #D print("count, w: ", count, " , ", w)
        log_msg = "count, w: " + str(count) + " , " + w
        cf.logger.info(log_msg)
        tmp2_sr = clean_ans_bodies_df[0].str.contains(w)
        for index,row in tmp2_sr.iteritems():
            if row:
                #TBD, Change string 'w' to list or tuple for better storage in df?
                #OK qagroup_df.loc[index, "HiScoreTerms"] = qagroup_df.loc[index, "HiScoreTerms"] + w + ' , '
                qagroup_df.loc[index, "HiScoreTerms"] += ( w + ' , ')
                #TBD, increment hst counter each time one is found.
                #TBD, How to handle multiple instances of hst? count each occurrence?
                #TBD, use df func to simply count number of HiScoreTerms in each row?
                qagroup_df.loc[index, "hstCount"] += 1


    #D print()
    #D cf.logger.debug("DBG, qagroup_df.head():")
    #D cf.logger.debug(qagroup_df.head())

    # Save full df to a file.
    outfile = "tmpdir/all_ans.csv"
    # TBD.1 This overwrites the preceding file each time this func is called.
    #   Must change the code to save data from each time this func is called.
    qagroup_df.to_csv(outfile)

    # Save possible valuable answers to a separate file for review.
    # Replace empty strings in HiScoreTerms cells with NaN,
    # to drop low value answers easily w/ dropna().
    qagroup_df['HiScoreTerms'].replace('', np.nan, inplace=True)
    #
    # Save a new df with only rows that have data in the HiScoreTerms column.
    #   TBD, Remove this statement to fix bug that deletes many Q&A of value.
    #   Consider other workarounds if this is a problem.
    # qagroup_df = qagroup_df.dropna(subset=['HiScoreTerms'])

    #D # Print partial data about interesting answers to check.
    #D print("\nCheck these low score Answers for useful data: ")
    #D for index,row in qagroup_df.iterrows():
    #D     if np.isnan(row['ParentId']):  # Found a question.
    #D         print('Id, Title: ', row['Id'], row['Title'])
    #D print(qagroup_df[['Id', 'Score', 'CreationDate']])
    #D print(qagroup_df[['Id', 'HiScoreTerms']])

    # Also write summary data to log.
    cf.logger.info("Check low score Answers for useful data: ")
    cf.logger.info(qagroup_df[['Id', 'Score', 'hstCount', 'CreationDate', 'Title', 'HiScoreTerms']])
    return qagroup_df

