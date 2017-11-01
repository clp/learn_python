# -*- coding: utf-8 -*-
# Using ~/anaconda3/bin/python: Python 3.6.0 :: Anaconda 4.3.0 (64-bit)
# Using Python 3.4.5 :: Anaconda 4.3.0 (64-bit), since Tue2017_0710

"""nltk_ex25.py

Analyze text data using the Natural Language Toolkit, NLTK.

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

These functions were built to work with the calling code in
fga_find_good_answers.py (which uses data from stackoverflow.com).
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


def convert_text_to_words(raw_q_a_s):
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
    q_a_text = BeautifulSoup(raw_q_a_s, "lxml").get_text()

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
    # separated by a space, and return the resulting string.

    return(" ".join(meaningful_words))


def clean_raw_data(qagroup_from_pop_top_ques_df):
    """Clean and parse the training set text.

    The input is a dataframe of one question and its related
    answers.  The question is both pop (popular) and top: it has
    several answers; and some answers are by owners with high
    reputation scores.

    Create a new column in the dataframe to hold the
    cleaned data.

    Process the text from one cell of a row of the dataframe
    into a string of meaningful words.
    Store that new string in the new cell for that row.
    Repeat for all the rows in the dataframe by using a loop.
    This updated dataframe is then used elsewhere.

    One use of this function reads text from the Body column of a row
    in the dataframe; converts it into words; and places them in the
    CleanBody cell of that row.

    Return a list of strings of cleaned text, one string for each
    row in the dataframe.
    """

    # Get the number of bodies based on that column's size
    num_bodies = qagroup_from_pop_top_ques_df["Body"].size
    cf.logger.info("clean_raw(): Number of bodies: " + str(num_bodies))

    clean_q_a_bodies_l = []
    qagroup_from_pop_top_ques_df["CleanBody"] = ""

    # Build a list that holds the cleaned text from each answer's body field.
    # Use it to find terms that match terms found in hi-score Answers.
    progress_msg_factor = int(round(num_bodies / 10))
    if progress_msg_factor <= 10:
        progress_msg_factor = 10
    for i in range(0, num_bodies):
        clean_body_s = convert_text_to_words(qagroup_from_pop_top_ques_df["Body"][i])
        clean_q_a_bodies_l.append(clean_body_s)
        #
        # Add new column to Answers df.
        qagroup_from_pop_top_ques_df.loc[i, "CleanBody"] = clean_body_s
        # Print a progress message; default is for every 10% of i/p data handled.
        if((i+1) % progress_msg_factor == 0):
            #D cf.logger.debug("Body %d of %d" % (i+1, num_bodies))
            #D cf.logger.debug('  Original text: ' + qagroup_from_pop_top_ques_df['Body'][i])
            cf.logger.debug('  clean*(): Partial slice of cleaned text:\n' + clean_body_s[:70])

    return clean_q_a_bodies_l


def make_bag_of_words(clean_q_a_bodies_l):
    """Extract numerical features from input text data.
    Later steps can use the numeric form with machine learning
    algorithms to analyze the data further.

    The input data is a list of strings of cleaned text,
    for one Q&A group.  Each string contains the words of the
    Body field of the question or of one of its answers.

    Consider the entire list to be a corpus or collection
    of documents in the form of a matrix.

    Each string in the list is a row of the matrix;
    and each ngram (word or phrase) in each row occupies one column
    of the matrix.

    Use CountVectorizer() from scikit-learn to convert
    that matrix into numerical feature vectors.

    Return a tuple of two structures.
    vocab_l: one list of all ngrams in the vocabulary;
    dist_a: one array with the total count of the occurrences
    of each ngram in the vocabulary list.


    # Some Details.

    To use phrases instead of single words in making the bag of words,
    specify a range, eg, 'ngram_range = (3,5)' in the arguments for
    CountVectorizer().

    To include 1-letter words in the bag, use this argument & pattern:
    'token_pattern = r'\b\w+\b'.

    Ref: http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction.

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

    train_data_features = vectorizer.fit_transform(clean_q_a_bodies_l)

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


def sort_q_a_by_score(qagroup_from_pop_top_ques_df):
    """Build a dataframe of Id's and their Scores, sorted by score.

    Use one Q&A group dataframe as input.

    Return the sorted dataframe with only Id's and Scores.
    """

    ids_and_scores_df = qagroup_from_pop_top_ques_df.sort_values(['Score'])
    ids_and_scores_df = ids_and_scores_df[['Id', 'Score']]

    cf.logger.info('Lowest scoring Answers:')
    cf.logger.info(ids_and_scores_df.head())
    cf.logger.info('Highest scoring Answers:')
    cf.logger.info(ids_and_scores_df.tail())

    return ids_and_scores_df


def find_hi_score_terms_in_bodies(words_sorted_by_count_orig_l, clean_q_a_bodies_l, num_hi_score_terms, qagroup_from_pop_top_ques_df):
    """Save terms that a record has in common with frequently-seen text.

    Important Variables.

    clean_q_a_bodies_df: A dataframe with cleaned text from Q&A bodies.

    HiScoreTerms: A column added to qagroup_from_pop_top_ques_df, holding terms
    that might indicate that this is a useful record.

    HSTCount: A column added to qagroup_from_pop_top_ques_df, holding the total count
    of the number of high score terms in a record.

    qagroup_from_pop_top_ques_df: A dataframe with one Q&A group
    (ie, one question with its related answers), which is selected from all
    such groups based on being 'pop' (popular, questions with several
    answers) and 'top' (having one or more answers by high-reputation
    owners).

    words_sorted_by_count_orig_l: The original list of all terms found
    in the bodies of records, which is sorted by the count of each term.


    Actions.

    Start with a list of terms (words or phrases) from the bodies
    of a Q&A group, sorted by count.
    Select a subset of terms that appear most often.
    Compare each term of that subset with each term in the body
    of one row in the Q&A group dataframe.  If there is a match,
    save the term and increment the total HSTCount
    for that row in the output dataframe.
    Use nested loops to check all frequently-seen terms against
    all rows of the Q&A group dataframe.

    Return the updated Q&A group dataframe for further investigation.
    """

    clean_q_a_bodies_df = pd.DataFrame(clean_q_a_bodies_l)
    cf.logger.debug("clean_q_a_bodies_l, top:")
    cf.logger.debug(clean_q_a_bodies_l[:1])
    cf.logger.debug("clean_q_a_bodies_l, bottom:")
    cf.logger.debug(clean_q_a_bodies_l[-1:])

    #TBF
    # p.1. Separate the terms from each other so it is clear what was found
    #  and what was not found in the A's being checked.

    qagroup_from_pop_top_ques_df["HiScoreTerms"] = ""
    qagroup_from_pop_top_ques_df["HSTCount"] = 0

    cf.logger.info("Terms from hi-score Answers.")
    #TBD, Maybe collect these "count,w" data into a single list,
    #   & write it in one step to the log file.
    for count, w in words_sorted_by_count_orig_l[-num_hi_score_terms:]:
        log_msg = "count, w: " + str(count) + " , " + w
        cf.logger.info(log_msg)
        tmp2_sr = clean_q_a_bodies_df[0].str.contains(w)
        for index, row in tmp2_sr.iteritems():
            if row:
                #TBD, Maybe Change string 'w' to list or tuple for better storage in df?
                #OK qagroup_from_pop_top_ques_df.loc[index, "HiScoreTerms"] = qagroup_from_pop_top_ques_df.loc[index, "HiScoreTerms"] + w + ' , '
                qagroup_from_pop_top_ques_df.loc[index, "HiScoreTerms"] += (w + ' , ')
                #TBD, increment hst counter each time one is found.
                #TBD, How to handle multiple instances of hst? count each occurrence?
                #TBD, use df func to simply count number of HiScoreTerms in each row?
                qagroup_from_pop_top_ques_df.loc[index, "HSTCount"] += 1

    #D cf.logger.debug("DBG, qagroup_from_pop_top_ques_df.head():")
    #D cf.logger.debug(qagroup_from_pop_top_ques_df.head())

    # Save possible valuable answers to a separate file for review.
    # Replace empty strings in HiScoreTerms cells with NaN,
    # to drop low value answers easily w/ dropna().
    qagroup_from_pop_top_ques_df['HiScoreTerms'].replace('', np.nan, inplace=True)
    #
    # Save a new df with only rows that have data in the HiScoreTerms column.
    #   TBD, Remove this statement to fix bug that deletes many Q&A of value.
    #   Consider other workarounds if this is a problem.
    # qagroup_from_pop_top_ques_df = qagroup_from_pop_top_ques_df.dropna(subset=['HiScoreTerms'])

    #D # Print partial data about interesting answers to check.
    #D print("\nCheck these low score Answers for useful data: ")
    #D for index,row in qagroup_from_pop_top_ques_df.iterrows():
    #D     if np.isnan(row['ParentId']):  # Found a question.
    #D         print('Id, Title: ', row['Id'], row['Title'])
    #D print(qagroup_from_pop_top_ques_df[['Id', 'Score', 'CreationDate']])
    #D print(qagroup_from_pop_top_ques_df[['Id', 'HiScoreTerms']])

    # Also write summary data to log.
    cf.logger.info("Check low score Answers for useful data: ")
    cf.logger.info(qagroup_from_pop_top_ques_df[['Id', 'Score', 'HSTCount', 'CreationDate', 'Title', 'HiScoreTerms']])
    return qagroup_from_pop_top_ques_df

if __name__ == '__main__':
    main()
