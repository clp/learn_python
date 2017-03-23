# -*- coding: utf-8 -*-

# nltk_ex22.py  clpoda  2017_0115 . 2017_0126 . 2017_0220
# Time-stamp: <Wed 2017 Mar 22 09:48:27 PMPM clpoda>
# Stand-alone program to test nltk.
#
# Ref: https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words


import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import pprint
from six.moves import range
import os
import csv

pd.set_option('display.width', 120)

# Show OwnerUserId w/o trailing '.0'; don't show commas in large numbers.
pd.options.display.float_format = '{:.0f}'.format


datadir = '/data/datasets/'
tmpdir = 'indir/'  # Relative to current dir
outdir = 'outdir/'  # Relative to current dir
#D q_fname = 'Questions.csv'
#D a_fname = 'Answers.csv'
#D a_fname = 'a6_999999.csv'  
#D a_fname = 'a5_99998.csv'  # Bag has 7903 rows.
q_fname = 'q3_992.csv'
a_fname = 'a3_986.csv'
#D a_fname = 'q_with_a.csv'  # O/p from fga*.py
#D a_fname = 'q_with_a.0211_1308.csv'  # 2729 lines; O/p from fga*.py
#D a_fname = 'q_with_a.40_owners_a5_9998.csv'  # 800 lines; O/p from fga*.py
#D q_fname = 'q2.csv'
#D a_fname = 'a2.csv'

#
# Choose tmpdir or datadir:
#D a_infile = tmpdir  + a_fname
a_infile = datadir + a_fname

# Choose tmpdir or datadir:
#D q_infile = tmpdir  + q_fname
q_infile = datadir + q_fname

print('=== Input files, q & a:\n'  + q_infile + '\n' + a_infile + '\n')


# Step 1. Read data from file into dataframe.

# Build data frames.
df_all_ans = pd.read_csv(a_infile, encoding='latin-1', warn_bad_lines=False, error_bad_lines=False)
#D questions are not yet used.
#D df_all_ques = pd.read_csv(q_infile, encoding='latin-1', warn_bad_lines=False, error_bad_lines=False)

print('=== df_all_ans.head():\n', df_all_ans.head())

numlines = len(df_all_ans)
progress_msg_factor = int(round(numlines/10))
print('\n=== Number of records in i/p data frame, df_all_ans: ' + str(numlines) + '\n')


# Step 2. Process the words of each input line.

from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords

#TBD nltk.download()
    # Done once & took 90 minutes; download text data sets, including stop words
    # Are there any issues btwn py2 & py3 for nltk?
    # S/w installed at ~/nltk_data/.
    # You can use web site & only d/l some parts if you don't need all.


def convert_text_to_words( raw_q_a ):
    # Convert a raw stackoverflow question or answer
    # to a string of words.
    # The input is a single string (a raw ques or ans entry), and
    # the output is a single string (a preprocessed ques or ans).
    #
    # 1. Remove HTML
    q_a_text = BeautifulSoup(raw_q_a, "lxml").get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", q_a_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join( meaningful_words ))


# Get the number of bodies based on that column's size
num_bodies = df_all_ans["Body"].size
#D print("=== Number of bodies: " + str(num_bodies))

print("=== For all ans: Cleaning and parsing the training set bodies...")
clean_a_bodies_l = []
for i in range( 0, num_bodies ):
    clean_a_bodies_l.append( convert_text_to_words( df_all_ans["Body"][i] ))
    # Print a progress message; default is for every 10% of i/p data handled.
    if( (i+1) % progress_msg_factor == 0 ):
        clean_q_a = convert_text_to_words( df_all_ans["Body"][i] )
        #D print("\n=== Body %d of %d" % ( i+1, num_bodies ))
        #D print('  Original text: ' + df_all_ans['Body'][i])
        #D print('  Cleaned text:  ' + clean_q_a)

# Write cleaned bodies to a file, one body per line, for visual review.
outfile = tmpdir + a_fname + '.out'
if os.path.exists(outfile):
    os.rename(outfile, outfile + '.bak')
    print('\nWARN: renamed o/p file w/ .bak; save it manually if needed: ' + outfile)
with open(outfile, 'w') as f:
    f.write('\n'.join(clean_a_bodies_l))


# Step 3. Build a bag of words and their counts.

# This code uses ngrams instead of single words.
def make_bag_of_words(clean_a_bodies_l):
    print("\nCreating the bag of words for word counts ...\n")
    from sklearn.feature_extraction.text import CountVectorizer

    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool
    #
    # Fails w/ MemErr with max_features at 5000; ok at 100-200.
    # Include ngram_range to use ngrams; otherwise, use single words only.
    # This: token_pattern = r'\b\w+\b', includes 1-letter words.
    #
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
    #
    train_data_features = vectorizer.fit_transform(clean_a_bodies_l)

    # Numpy arrays are easy to work with, so convert the result to an
    # array
    #
    train_data_features = train_data_features.toarray()

    #DBG Wed2017_0118_19:38 , w/ q9999 data, got (727, 1550):
    print('Bag shape: rows (num of records), cols (num of features):')
    print(train_data_features.shape, '\n')

    # Take a look at the words in the vocabulary
    vocab = vectorizer.get_feature_names()

    # Print counts of each word in vocab.
    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)

    return (vocab, dist)

print('=== make_bag_of_words(clean_a_bodies_l')
(vocab, dist) = make_bag_of_words(clean_a_bodies_l)


# Sort and save vocabulary data to file w/ a specified suffix.

def sort_save_vocab(suffix):
    # For each, print the vocabulary word and the number of times it
    # appears in the training set
    count_tag_l = []
    word_freq_d = {}
    for tag, count in zip(vocab, dist):
        count_tag_l.append((count, tag))
        #D word_freq_d[tag] = count

    # Sort the list of tuples by count.
    words_sorted_by_count = sorted(count_tag_l, key=lambda x: x[0])

    # Write sorted vocab to a file.
    outfile = tmpdir + a_fname + suffix
    if os.path.exists(outfile):
        os.rename(outfile, outfile + '.bak')
        print('\nWARN: renamed o/p file to *.bak; save it manually if needed:'+ outfile)
    with open(outfile, 'w') as f:
        for count, word in words_sorted_by_count:
            print(count, word, file=f)

sort_save_vocab('.vocab')


# Step 4. Sort data by score.

print('\n=== Sort data by Score for each record.\n')
df_score = df_all_ans.sort_values(['Score'])
df_score = df_score[['Id', 'Score']]

# Compute the number of records to use for computation and display.
rec_selection_ratio = 0.10  # Default 0.01?
num_selected_recs = int(numlines * rec_selection_ratio)
if num_selected_recs < 6:
    num_selected_recs = 5
print('  rec_selection_ratio,  number of selected recs: ', rec_selection_ratio, num_selected_recs, '\n')
print('Lowest scoring records:')
print(df_score.head(), '\n') #D print(df_score.head(num_selected_recs), '\n')
print('Highest scoring records:')
print(df_score.tail(), '\n') #D print(df_score.tail(num_selected_recs), '\n')


# Step 5. Find most frequent words for top-scoring records.

print('\n=== Step 5. Find most freq words for top-scoring records.')
df_score_top_n = df_score[['Id']]
#D print(df_score_top_n.tail(20), '\n') #D print(df_score_top_n.tail(num_selected_recs), '\n')

# Use top_n records & count their words.
print("\nFor top ans: Cleaning and parsing the training set bodies...")
#D print("Number of bodies: " + str(num_bodies))

def find_freq_words():
    top_n_bodies = []
    df_score_l = []
    # Convert dataframe to list of Id's, to get the body of each Id.
    if top: # Get the tail of the list, highest-score items.
        df_score_l = df_score_top_n['Id'].tail(num_selected_recs).tolist()
    else:  # Get the head of the list, lowest-score items.
        df_score_l = df_score_top_n['Id'].head(num_selected_recs).tolist()
    df8 = df_all_ans.set_index('Id')
    progress_count = 0
    for i in df_score_l:
        progress_count += 1
        top_n_bodies.append( convert_text_to_words( df8["Body"][i] ))
        # Print a progress message for every 10% of i/p data handled.
        if( (progress_count+1) % progress_msg_factor == 0 ):
            clean_q_a = convert_text_to_words( df8["Body"][i] )
            print("\nBody for Id %d " % ( i))
            print('  Original text:\n' + df8['Body'][i][:70])
            print('  Cleaned text:\n' + clean_q_a[:70])
    return top_n_bodies 

top = True
top_n_bodies = find_freq_words()

print('=== make_bag_of_words(top_n_bodies')
(vocab, dist) = make_bag_of_words(top_n_bodies)

sort_save_vocab('.vocab.hiscore')


# Step 6. Find most frequent words for bottom-scoring records.

print('\n=== Step 6. Find most freq words for bottom-scoring records.')
df_score_bot_n = df_score[['Id']]
#D print(df_score_bot_n.head(20), '\n') #D print(df_score_bot_n.head(num_selected_recs), '\n')

top = False
bot_n_bodies = find_freq_words()

print('=== make_bag_of_words(bot_n_bodies')
(vocab, dist) = make_bag_of_words(bot_n_bodies)

sort_save_vocab('.vocab.loscore')


'bye'
