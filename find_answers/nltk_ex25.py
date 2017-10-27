# -*- coding: utf-8 -*-
# Using ~/anaconda3/bin/python: Python 3.6.0 :: Anaconda 4.3.0 (64-bit)
# Using Python 3.4.5 :: Anaconda 4.3.0 (64-bit), since Tue2017_0710

"""nltk_ex25.py

Analyze data using the Natural Language Toolkit, NLTK.

NLTK is a suite of Python s/w, data sets, and documents
to support the development of Natural Language Processing, NLP.

This module contains several functions that use NLTK software
to process text and produce output that might help to find
some good but 'hidden' records in the input.  A 'hidden' record
has a low score and might not be prominently displayed in a
list of records based on scores.

One approach is to collect terms found in high-score records,
and look for low-score records that contain the same terms.
Count the number of matching terms, and assume that a high count
indicates a low-score record that might have useful content.

Return the o/p to the caller for further evaluation & processing.

This module was built to work with the code in
fga_find_good_answers.py, which uses data from stackoverflow.com.
See that program for more details.


Usage:
    import nltk_ex25
    pydoc nltk_ex25

    See fga_find_good_answers.py for an example that uses
    the functions in this module.

    Set the value of num_hi_score_terms in the calling program.
    It determines how many hi-score terms are used to search
    for matching terms in low-score answers.  A bigger number
    will cause the program to run longer, and might find more
    'hidden' answers that could be valuable.

----------------------------------------------------------

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

from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords


def main():
    """Initialization:
    Download NLTK stopwords data if not found locally.
    """

    log_msg = cf.log_file + ' - Start logging for ' + os.path.basename(__file__)
    cf.logger.info(log_msg)

    # Check and download stopwords if not found locally.
    nltk.download('stopwords')


def convert_text_to_words(raw_q_a):
    """Convert and filter the i/p text string to a string of words.

    Convert a raw stackoverflow question or answer
    to a string of meaningful words for detailed analysis.

    The input is a single string of text.
    That content is processed in various ways, eg, remove HTML,
    remove non-letters, convert to lower-case, and remove
    stop words that clutter the output.
    
    Return a single string of meaningful words.
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
    #ORG meaningful_words = [w for w in words if not w in stops]
    meaningful_words = [w for w in words if w not in stops]

    # 6. Join the words back into one string of words, each word
    # separated by a space, and return the result.

    return(" ".join(meaningful_words))


def clean_raw_data(qagroup_df):
    """Clean and parse the training set text.

    The input is a data frame of one question and its related
    answers.

    Create a new column in the data frame to hold the
    cleaned data.

    Process the text in one cell of a row of the data frame
    into a string of meaningful words.
    Store that string in the new cell for that row.
    Loop over all the rows in the data frame.

    One use of this function converts text from the Body column into
    the words placed in the CleanBody cell of an Answer record
    in the data frame.

    Return a list of strings of clean answer bodies, for all the
    answers to one question.
    """

    # Get the number of bodies based on that column's size
    num_bodies = qagroup_df["Body"].size
    cf.logger.info("clean_raw(): Number of bodies: " + str(num_bodies))

    clean_ans_bodies_l = []
    qagroup_df["CleanBody"] = ""

    # Build a list that holds the cleaned text from each answer's body field.
    # Use it to find terms that match terms found in hi-score Answers.
    progress_msg_factor = int(round(num_bodies / 10))
    if progress_msg_factor <= 10:
        progress_msg_factor = 10
    for i in range(0, num_bodies):
        clean_body_s = convert_text_to_words(qagroup_df["Body"][i])
        clean_ans_bodies_l.append(clean_body_s)
        #
        # Add new column to Answers df.
        qagroup_df.loc[i, "CleanBody"] = clean_body_s
        # Print a progress message; default is for every 10% of i/p data handled.
        if((i+1) % progress_msg_factor == 0):
            #D cf.logger.debug("Body %d of %d" % (i+1, num_bodies))
            #D cf.logger.debug('  Original text: ' + qagroup_df['Body'][i])
            cf.logger.debug('  clean*(): Partial slice of cleaned text:\n' + clean_body_s[:70])

    return clean_ans_bodies_l


def make_bag_of_words(clean_ans_bodies_l):
    """Collect and count ngrams (words or phrases) in the text.

    To use phrases instead of single words in the analysis, specify
    a range, eg, 'ngram_range = (3,5)' in the arguments for
    CountVectorizer.

    To include 1-letter words, use this argument & pattern:
    'token_pattern = r'\b\w+\b'.

    Return a tuple of two structures: a list of ngrams found
    in the i/p; and an array with the count of each ngram in that list.
    """

    cf.logger.debug("Creating the bag of words for word counts ...")
    from sklearn.feature_extraction.text import CountVectorizer

    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool

    # TBD, Fails w/ MemErr with max_features at 5000; ok at 200.

    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 ngram_range=(3, 5),
                                 # token_pattern=r'\b\w+\b',
                                 max_features=200)

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
    vocab_l = vectorizer.get_feature_names()

    # Sum up the counts of each vocabulary term in vocab_l.
    dist_a = np.sum(train_data_features, axis=0)

    return (vocab_l, dist_a)


def sort_vocab(vocab_l, dist_a):
    """Sort the i/p vocabulary data by count.

    The two input structures are a list of terms and an array
    of counts that correspond to the terms.

    Combine these two structures into a list of tuples;
    then sort that list by the count element.

    Return a list of tuples sorted by count: count, term.
    """

    count_term_l = []
    for term, count in zip(vocab_l, dist_a):
        count_term_l.append((count, term))

    # Sort the list of tuples by count.
    words_sorted_by_count_l = sorted(count_term_l, key=lambda x: x[0])

    return words_sorted_by_count_l


def sort_answers_by_score(qagroup_df):
    """Build a dataframe of Id's and their Scores, sorted by score.

    Use one Q&A group dataframe as input.

    Return the sorted dataframe with only Id's and Scores.
    """

    ids_and_scores_df = qagroup_df.sort_values(['Score'])
    ids_and_scores_df = ids_and_scores_df[['Id', 'Score']]

    cf.logger.info('Lowest scoring Answers:')
    cf.logger.info(ids_and_scores_df.head())
    cf.logger.info('Highest scoring Answers:')
    cf.logger.info(ids_and_scores_df.tail())

    return ids_and_scores_df


def find_words_based_on_score(top, ids_sorted_by_score_l, num_selected_recs, progress_msg_factor, qagroup_df):
    """Build a list of strings; each string has terms from a record's body text.

    The inputs include the dataframe of one Q&A group, and the
    list of Id's for the members of the Q&A group sorted by Score.

    Select items from the Q&A group based on their score, either the
    highest or lowest score items, depending on how the caller
    sets the 'top' var.

    Clean the raw body text of each selected item, store it in a string
    of words, and append that string to a list.

    Return that list of items.
    """
    selected_bodies_l = []
    # Convert dataframe to list of Id's, to get the body of each Id.
    #
    #TBD, Wed2017_1025_13:59 , num_selected_recs=7603 for i/p a6* file; Check its use here:
        # Should ids_sorted_by_score_l be 10% of i/p length, or just a small number-10-20?
    if top:  # Get the tail of the list, highest-score items.
        ids_sorted_by_score_l = ids_sorted_by_score_l[-num_selected_recs:]
    else:  # Get the head of the list, lowest-score items.
        ids_sorted_by_score_l = ids_sorted_by_score_l[0:num_selected_recs]
    tmp_df = qagroup_df.set_index('Id')
    progress_count = 0
    for i in ids_sorted_by_score_l:
        progress_count += 1
        clean_q_a = convert_text_to_words(tmp_df["Body"][i])
        selected_bodies_l.append(clean_q_a)
        # Print a progress message (default: for every 10% of i/p data handled).
        if((progress_count+1) % progress_msg_factor == 0):
            #D cf.logger.debug("Body for Id %d " % (i))
            #D cf.logger.debug('  Original text:\n' + tmp_df['Body'][i][:70])
            cf.logger.debug('  find_words*(): Partial slice of cleaned text:\n' + clean_q_a[:70])
    return selected_bodies_l


def search_for_terms(words_sorted_by_count_orig_l, clean_ans_bodies_l, num_hi_score_terms, qagroup_df):
    """TBD summary.

    Important Variables.

    clean_ans_bodies_df: A dataframe with text from bodies that were cleaned.
        TBD, what other fields are in it?

    HiScoreTerms: A column added to the qagroup_df, holding terms that have high scores,
    extracted from TBD, based on TBD.

    HSTCount: A column added to the qagroup_df, holding the total count
    of the number of high score terms for a record.

    qagroup_df: A dataframe with one Q&A group, one question with its related answers.

    words_sorted_by_count_orig_l: The original list of all words found TBD,
    that is sorted by the count of each word.


    Actions.

    TBD:
    Read each answer and save any terms that it has in common
    with (high frequency) text from the high-score answers.


    TBD:
    Start with a list of words from the
    (answers?) of a Q&A group.
    Use the subset of words that appear most often.
    Compare each word of that subset with each word in the body of each
    (answer?).  If there is a match,
    save the word and increment the total HSTCount
    for that row in the output dataframe.
    Repeat for each item in the frequent words list,
    and for each item in the Q&A group,
    by using two nested loops.

    Return the updated dataframe further investigation.
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

    qagroup_df["HiScoreTerms"] = ""
    qagroup_df["HSTCount"] = 0

    #D print("\nTerms from hi-score Answers.")
    cf.logger.info("Terms from hi-score Answers.")
    #TBD, Maybe collect these "count,w" data into a single list,
    #   & write it in one step to the log file.
    for count, w in words_sorted_by_count_orig_l[-num_hi_score_terms:]:
        #D print("count, w: ", count, " , ", w)
        log_msg = "count, w: " + str(count) + " , " + w
        cf.logger.info(log_msg)
        tmp2_sr = clean_ans_bodies_df[0].str.contains(w)
        for index, row in tmp2_sr.iteritems():
            if row:
                #TBD, Change string 'w' to list or tuple for better storage in df?
                #OK qagroup_df.loc[index, "HiScoreTerms"] = qagroup_df.loc[index, "HiScoreTerms"] + w + ' , '
                qagroup_df.loc[index, "HiScoreTerms"] += (w + ' , ')
                #TBD, increment hst counter each time one is found.
                #TBD, How to handle multiple instances of hst? count each occurrence?
                #TBD, use df func to simply count number of HiScoreTerms in each row?
                qagroup_df.loc[index, "HSTCount"] += 1

    #D print()
    #D cf.logger.debug("DBG, qagroup_df.head():")
    #D cf.logger.debug(qagroup_df.head())

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
    cf.logger.info(qagroup_df[['Id', 'Score', 'HSTCount', 'CreationDate', 'Title', 'HiScoreTerms']])
    return qagroup_df

if __name__ == '__main__':
    main()
