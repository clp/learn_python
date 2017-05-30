# -*- coding: utf-8 -*-
# Using ~/anaconda3/bin/python: Python 3.6.0 :: Anaconda 4.3.0 (64-bit)

# Time-stamp: <Tue 2017 May 30 08:39:06 AMAM clpoda>

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
     python nltk_ex25.py
    
     Set the value of num_hi_score_terms at the bottom of the file;
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

import logging


# ----------------------------------------------------------
# TBD Hard-coded i/p file to use for temporary debugging.
#
# a_fname = 'pid_231767.csv'  # Based on o/p from fga*.py?
# ----------------------------------------------------------



# Configure basic logging.
# Set logging level to DEBUG, INFO, WARNING, ERROR, CRITICAL
# Set level to ERROR for normal use; set to INFO or DEBUG for development.
#
# logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')
# logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')

# Redirect STDERR or specify a log file:
log_file = "nltk_ex25.log"
log_level = logging.INFO
#TBD log_level = logging.DEBUG
logging.basicConfig(filename=log_file, level=log_level, format=' %(asctime)s - %(levelname)s - %(message)s')
# Sample log cmd: logging.debug('msg text var=' + str(var))
logger = logging.getLogger(__name__)
log_msg = '\n' + log_file + ' - Start logging.\n'
logger.info(log_msg)

a_infile  = ""

def main():
    init()
    global tmpdir, all_ans_df, all_ques_df
    
    a_fname, a_infile, q_infile, datadir, tmpdir, outdir = config_data()
    
    logger.info("Step 1. Read data from file into dataframes.")
    all_ans_df, all_ques_df, progress_msg_factor, numlines = read_data(a_infile, q_infile)
    
    logger.info("Step 2. Process the words of each input line.")
    clean_ans_bodies_l = clean_raw_data(a_fname, progress_msg_factor)
    
    logger.info("Step 3. Build a bag of words and their counts.")
    (vocab, dist) = make_bag_of_words(clean_ans_bodies_l)
    words_sorted_by_count_l = sort_save_vocab('.vocab', vocab, dist, a_fname)
    # Save the original list for later searching.
    words_sorted_by_count_main_l = words_sorted_by_count_l
    
    logger.info('Step 4. Sort Answers by Score.')
    score_df, num_selected_recs = sort_answers_by_score(numlines)
    
    logger.info('Step 5. Find most freq words for top-scoring Answers.')
    score_top_n_df = score_df[['Id']]
    
    #OK.tbr  print("DBG.103", score_top_n_df.tail(num_selected_recs), '\n')
    #TBD, Maybe convert df to string so logger can print title & data w/ one cmd:
    #TBD log_msg = "score_top_n_df.tail():" + CONVERT_DF_TO_STRING(score_top_n_df.tail())
    #TBD logger.debug(log_msg)
    logger.debug("score_top_n_df.tail():")
    logger.debug(score_top_n_df.tail(20)) 
    # Use top_n Answers & count their words.
    logger.info("For top ans: Cleaning and parsing the training set bodies...")
    
    top = True
    top_n_bodies = find_freq_words(top, score_top_n_df, num_selected_recs, progress_msg_factor)
    logger.info('make_bag_of_words(top_n_bodies)')
    (vocab, dist) = make_bag_of_words(top_n_bodies)
    sort_save_vocab('.vocab.hiscore', vocab, dist, a_fname)
    
    logger.info("Step 6. Find most frequent words for bottom-scoring Answers.")
    # Keep these data to compare w/ words for top-scoring Answers; s/b some diff.
    # If they are identical, there may be a logic problem in the code.
    score_bot_n_df = score_df[['Id']]
    logger.debug("score_bot_n_df.head():")
    logger.debug(score_bot_n_df.head(20))
    #TBR print(score_bot_n_df.head(num_selected_recs), '\n')
    top = False
    bot_n_bodies = find_freq_words(top, score_top_n_df, num_selected_recs, progress_msg_factor)
    logger.info('make_bag_of_words(bot_n_bodies)')
    (vocab, dist) = make_bag_of_words(bot_n_bodies)
    sort_save_vocab('.vocab.loscore', vocab, dist, a_fname)
    
    logger.info("Step 7. Search lo-score A's for hi-score text.")
    search_for_terms(words_sorted_by_count_main_l, clean_ans_bodies_l)
    
    
def init():
    """Initialize some settings for the program.
    """
    # Initialize settings for pandas.
    pd.set_option('display.width', 0)  # 0=no limit, use for debugging; try 120.
    
    # Don't show commas in large numbers.
    # Show OwnerUserId w/o '.0' suffix.
    pd.options.display.float_format = '{:.0f}'.format
    
    
def config_data():
    """Configure path and file names for i/o data.
    """
    datadir = '/data/datasets/'
    tmpdir = 'tmpdir/'  # Relative to current dir
    outdir = 'outdir/'  # Relative to current dir
    #D q_fname = 'Questions.csv'
    #D a_fname = 'Answers.csv'
    
    # Smaller data sets, used for debugging.
    #D a_fname = 'a6_999999.csv'  
    #D a_fname = 'a5_99998.csv'  # Bag has 7903 rows.
    #D q_fname = 'q3_992.csv'
    #D a_fname = 'a3_986.csv'
    #D a_fname = 'q_with_a.csv'  # O/p from fga*.py
    a_fname = 'pid_231767.csv'  # TBD Temp hard-coded i/p file
    #D a_fname = 'q_with_a.0211_1308.csv'  # 2729 lines; O/p from fga*.py
    #D a_fname = 'q_with_a.40_owners_a5_9998.csv'  # 800 lines; O/p from fga*.py
    q_fname = 'q2.csv'
    #D a_fname = 'a2.csv'
    
    # Choose tmpdir or datadir:
    #D a_infile = tmpdir  + a_fname
    #D a_infile = datadir + a_fname
    a_infile = outdir + a_fname
    
    # Choose tmpdir or datadir:
    #D q_infile = tmpdir  + q_fname
    q_infile = datadir + q_fname
    
    print('\nInput files, q & a:\n'  + q_infile + '\n' + a_infile)
    logger.info('Input files, q & a:\n'  + q_infile + '\n' + a_infile)
    
    return a_fname, a_infile, q_infile, datadir, tmpdir, outdir
    
    
def read_data(ans_file, ques_file):
    """Read the csv i/p files and store into a pandas data frame.
    Compute a factor that dictates how progress will be indicated
    during read operations.
    """
    ans_df = pd.read_csv(ans_file, encoding='latin-1', warn_bad_lines=False, error_bad_lines=False)
    #TBD question file is not used now. Use empty df for ques_df now.
    #D ques_df = pd.read_csv(ques_file, encoding='latin-1', warn_bad_lines=False, error_bad_lines=False)
    ques_df = pd.DataFrame()
    
    numlines = len(ans_df)
    print('\nNumber of answer records in i/p data frame, ans_df: ' + str(numlines))
    logger.info('Number of answer records in i/p data frame, ans_df: ' + str(numlines))
    progress_msg_factor = int(round(numlines/10))
    print()
    return ans_df, ques_df, progress_msg_factor, numlines


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
    
    
def clean_raw_data(a_fname, progress_msg_factor ):
    """ For all answers: Clean and parse the training set bodies.")
    """
    # Get the number of bodies based on that column's size
    num_bodies = all_ans_df["Body"].size
    logger.info("Number of bodies: " + str(num_bodies))
    
    clean_ans_bodies_l = []
    all_ans_df["CleanBody"] = ""
    
    # Build a list that holds the cleaned text from each answer's body field.
    # Use it to find terms that match terms found in hi-score Answers.
    for i in range( 0, num_bodies ):
        clean_body = convert_text_to_words( all_ans_df["Body"][i] )
        clean_ans_bodies_l.append(clean_body)
        #
        # Add new column to Answers df.
        all_ans_df.loc[i,"CleanBody"] = clean_body
        # Print a progress message; default is for every 10% of i/p data handled.
        if( (i+1) % progress_msg_factor == 0 ):
            clean_q_a = convert_text_to_words( all_ans_df["Body"][i] )
            #D logger.debug("Body %d of %d" % ( i+1, num_bodies ))
            #D logger.debug('  Original text: ' + all_ans_df['Body'][i])
            #D logger.debug('  Cleaned text:  ' + clean_q_a)
    
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
    
    This arg includes 1-letter words: 'token_pattern = r'\b\w+\b'.
    """
    logger.debug("Creating the bag of words for word counts ...")
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
    logger.info('Bag shape: rows (num of records), cols (num of features):')
    logger.info(train_data_features.shape)
    
    # Take a look at the words in the vocabulary
    vocab = vectorizer.get_feature_names()
    
    # Print counts of each word in vocab.
    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)
    
    return (vocab, dist)
    
    
def sort_save_vocab(suffix, vocab, dist, a_fname):
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
    
    
def sort_answers_by_score(numlines):
    """
    Build a sorted dataframe of answers.
    """
    score_df = all_ans_df.sort_values(['Score'])
    score_df = score_df[['Id', 'Score']]
    
    # Compute the number of records to use for computation and display.
    rec_selection_ratio = 0.10  # Default 0.01?
    num_selected_recs = int(numlines * rec_selection_ratio)
    if num_selected_recs < 6:
        num_selected_recs = 5
    log_msg = "  rec_selection_ratio,  number of selected recs: " + str(rec_selection_ratio) + ", " + str(num_selected_recs)
    logger.info(log_msg)
    logger.info('Lowest scoring Answers:')
    logger.info(score_df.head())
    #D print(score_df.head(num_selected_recs), '\n')
    logger.info('Highest scoring Answers:')
    logger.info(score_df.tail())
    #D print(score_df.tail(num_selected_recs), '\n')
    
    return score_df, num_selected_recs
    
    
def find_freq_words(top, score_top_n_df, num_selected_recs, progress_msg_factor):
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
    df8 = all_ans_df.set_index('Id')
    progress_count = 0
    for i in score_df_l:
        progress_count += 1
        top_n_bodies.append( convert_text_to_words( df8["Body"][i] ))
        # Print a progress message for every 10% of i/p data handled.
        if( (progress_count+1) % progress_msg_factor == 0 ):
            clean_q_a = convert_text_to_words( df8["Body"][i] )
            #D logger.debug("Body for Id %d " % ( i))
            #D logger.debug('  Original text:\n' + df8['Body'][i][:70])
            #D logger.debug('  Cleaned text:\n' + clean_q_a[:70])
    return top_n_bodies 
    
    
def search_for_terms(words_sorted_by_count_main_l, clean_ans_bodies_l):
    """
    Read each answer and save any terms that it has in common
    with (high frequency) text from the high-score answers.
    
    Save those answers for further investigation.
    """
    
    clean_ans_bodies_df = pd.DataFrame(clean_ans_bodies_l)
    logger.debug("clean_ans_bodies_l, top:") 
    logger.debug(clean_ans_bodies_l[:1]) 
    logger.debug("clean_ans_bodies_l, bottom:")
    logger.debug(clean_ans_bodies_l[-1:]) 
    
    #D Use short list cab_l for testing:
    #D cab_l = ['step isprimenumber call', 'foo step isprimenumber call bar', 'yield abc def ghi generators long first string', 'generator', 'foo next function bar', 'short list foo bar baz']
    
    #D cab_df = pd.DataFrame(cab_l)
    
    #TBF
    # p.1. Separate the terms from each other so it is clear what was found
    #  and what was not found in the A's being checked.
    #  Should tmp_df not be a df?  Would a list of lists be better?
    
    all_ans_df["HiScoreTerms"] = ""
    
    print("\nTerms from hi-score Answers.")
    logger.info("Terms from hi-score Answers.")
    #TBD, Maybe collect these "count,w" data into a single list,
    #   & write it in one step to the log file.
    for count,w in words_sorted_by_count_main_l[-num_hi_score_terms:]:
        print("count, w: ", count, " , ", w)
        log_msg = "count, w: " + str(count) + " , " + w
        logger.info(log_msg)
        tmp2_sr = clean_ans_bodies_df[0].str.contains(w)
        for index,row in tmp2_sr.iteritems():
            if row:
                #TBD, Change string 'w' to list or tuple for better storage in df?
                #OK all_ans_df.loc[index, "HiScoreTerms"] = all_ans_df.loc[index, "HiScoreTerms"] + w + ' , '
                all_ans_df.loc[index, "HiScoreTerms"] += ( w + ' , ')
    
    #D print()
    #D logger.debug("DBG, all_ans_df.head():")
    #D logger.debug(all_ans_df.head())
    
    # Save full df to a file.
    outfile = "tmpdir/all_ans.csv"
    all_ans_df.to_csv(outfile)
    
    # Save possible valuable answers to a separate file for review.
    # Replace empty strings in HiScoreTerms cells with NaN,
    # to drop low value answers easily w/ dropna().
    all_ans_df['HiScoreTerms'].replace('', np.nan, inplace=True)
    # Save a new df with only rows that have data in the HiScoreTerms column.
    ans_with_hst_df = all_ans_df.dropna(subset=['HiScoreTerms'])
    outfile = "tmpdir/ans_with_hst.csv"
    ans_with_hst_df.to_csv(outfile)
    
    # Print partial data about interesting answers to check.
    print("\nCheck these low score Answers for useful data: ")
    for index,row in ans_with_hst_df.iterrows():
        if np.isnan(row['ParentId']):  # Found a question.
            print('Id, Title: ', row['Id'], row['Title'])
    print(ans_with_hst_df[['Id', 'Score', 'CreationDate']])
    print(ans_with_hst_df[['Id', 'HiScoreTerms']])
    
    # Also write summary data to log.
    logger.info("Check low score Answers for useful data: ")
    logger.info(ans_with_hst_df[['Id', 'Score', 'CreationDate', 'Title', 'HiScoreTerms']])
    #TBD.0 return


if __name__ == '__main__':
    # Set initial values of some important variables.
    num_hi_score_terms = 22  # Use 3 for testing; 11 or more for use.
    print("num_hi_score_terms: ", num_hi_score_terms)
    main()
    log_msg = log_file + " - Finish program & logging.\n\n"
    logger.warning(log_msg)

'bye'
