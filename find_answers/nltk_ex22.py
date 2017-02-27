# -*- coding: utf-8 -*-

# nltk_ex22.py  clpoda  2017_0115 . 2017_0126 . 2017_0220
#   VM-ds2:/home/ann/p/learn_python/find_answers/
#   Time-stamp: <Fri 2017 Feb 24 04:14:09 PMPM clpoda>
#
# Ref: https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words
#
# Thu2017_0126_15:39 :
# Copied from antxso/anques.py and modified to process Answers.csv
# Mon2017_0220_18:26
# Stand-alone program to test nltk.
# Copied from ~/p/antxso/antxso/aa_ananswer.py to ~/p/learn_python/find_answers/nltk_ex22.py.

# #####################################################
# Preliminary steps.
# TBD, Maybe not needed.
# Mon2017_0220_18:29 , Build csv file w/ each question with its answers.
# python fga_find_good_answers.py
# Use the o/p data in outdir/qa_with_keyword.csv (or q_with_a.csv).
# #####################################################


import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import pprint
from six.moves import range
import os
import csv

pd.set_option('display.width', 120)

#TBD, To show OwnerUserId w/o '.0'; see b.10.
#TBD, Do we ever need floats to be shown as floats? Maybe restrict this to ouid field.
#TBD  Some calculations of score stats will be floats.
pd.options.display.float_format = '{:.0f}'.format  # Don't show commas in large numbers


#TBD Make the tmp & out dirs w/ this program, if they don't exist?
#TBD Make the test data files w/ this program, if they don't exist?
datadir = '/data/datasets/'  # Don't make this dir-permits problem.
#
# TBD.1, Mon2017_0220_18:51 , Rename: indir, outdir, tmpdir? See b.5.
#TBR tmpdir = 'tmp/'  # Relative to /home/ann/p/antxso/
tmpdir = 'indir/'  # Relative to /home/ann/p/antxso/
outdir = 'outdir/'  # Relative to /home/ann/p/antxso/
q_fname = 'Questions.csv'
a_fname = 'Answers.csv'
a_fname = 'a6_999999.csv'  # Bag has TBD rows.
a_fname = 'a5_99998.csv'  # Bag has 7903 rows.
q_fname = 'q3_992.csv'
a_fname = 'a3_986.csv'
a_fname = 'q_with_a.csv'  # O/p from fga*.py
a_fname = 'q_with_a.0211_1308.csv'  # O/p from fga*.py
a_fname = 'q_with_a.40_owners_a5_9998.csv'  # O/p from fga*.py
#D q_fname = 'q2.csv'
#D a_fname = 'a2.csv'

# Other test data files.
#D a_fname = 'a4.csv'
#D a_fname = 'a5.csv'
#D q_fname = 'q27_117.csv'
#D q_fname = 'q29_992.csv'  # Has 992 HTML question lines
#D q_fname = 'q30_99993.csv' # Has 99993 HTML question lines, abt 5666 recs
#
# Choose one:
#D a_infile = tmpdir  + a_fname
a_infile = datadir + a_fname

# Choose one:
#D q_infile = tmpdir  + q_fname
q_infile = datadir + q_fname

print('Input files, q & a:\n'  + q_infile + '\n' + a_infile)
print()


# Build data frames.
df_all_ans = pd.read_csv(a_infile, encoding='latin-1', warn_bad_lines=False, error_bad_lines=False)
#TBD.not.used  df_all_ques = pd.read_csv(q_infile, encoding='latin-1', warn_bad_lines=False, error_bad_lines=False)

print('df_all_ans.head(): ' )
print(df_all_ans.head())

numlines = len(df_all_ans)
print('Number of records in i/p data frame, df_all_ans: ' + str(numlines))
progress_msg_factor = int(round(numlines/10))
print()




## print('\n=== Find range of scores for top-scoring OwnerUserId.')
#TBD, Tue2017_0124_12:27
# Get list of top-scoring ouid's
# For each ouid:
#   Find every record they wrote.
#   Save the score of each record.
#   Run describe() on that list of scores.
#   Write the range of scores.
#   Write top and bottom N scores.

## # Convert dataframe to list of Id's, to get the title of each Id.
## df_score_l = df_score_top_n['Id'].tail(num_selected_recs).tolist()
## df8 = df_all_ans.set_index('Id')
## for i in df_score_l:
##     top_n_titles.append( qa_to_words( df8["Title"][i] ))
##     # Print a progress message for every 10% of i/p data handled.
##     if( (i+1)%progress_msg_factor == 0 ):
##         clean_qa = qa_to_words( df8["Title"][i] )
##         print("\nTitle for Id %d " % ( i))
##         print('  Original text: ' + df8['Title'][i])
##         print('  Cleaned text:  ' + clean_qa)

'''
# TBD, answers have no title field! Use body field.
Steps
Mon2017_0116_16:33
Read each Title from df_all_ans, ie, column 5.
For each entry:
    For each word:
        Process each word from current line..
        Term is any white-space separated string; including first & last strings.
        Shift to lower case?
        Remove all punctuation symbols?
            No: "... 'in' ..." is different from "in".
            So keep punc w/ words?
            Maybe remove trailing punc (,;:) and unmatched quotes?
Build dict of word:count.
Print most frequent words.

'''


# Process the words of each input line.
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords # Import the stop word list

#TBD print stopwords.words("english")
#TBD.done.once.90.minutes  nltk.download()  # Download text data sets, including stop words
    #TBD, are there any issues btwn py2 & py3 for nltk?
    # S/w installed at ~/nltk/ by default.


def qa_to_words( raw_qa ):
    # Function to convert a raw stackoverflow question or answer
    # to a string of words.
    # The input is a single string (a raw q or a entry), and
    # the output is a single string (a preprocessed q or a).
    #
    # 1. Remove HTML
    #ORG.OK qa_text = BeautifulSoup(raw_qa).get_text()
    qa_text = BeautifulSoup(raw_qa, "lxml").get_text()
    #
    # 2. Remove non-letters
    #TBD.skip letters_only = qa_text
    letters_only = re.sub("[^a-zA-Z]", " ", qa_text)
    #
    # 3. Convert to lower case, split into individual words
    #TBD Keep camel case terms?
    #TBD words = letters_only.lower().split()
    #OK? words = letters_only.split()
    words = letters_only.lower().split()
        # Sun2017_0122_18:51 , see b.13; more stopwords were removed by adding lower().
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

# '''
print()
#D clean_qa = qa_to_words( df_all_ans["Body"][0:5].all())
clean_qa = qa_to_words( df_all_ans["Body"].all())
print('The answer body keywords: ')
#D print('  Cleaned text:  ' + clean_qa)
print(clean_qa)
print()
# '''


'''
# Get the number of bodies based on the dataframe column size
num_titles = df_all_ans["Body"].size
'''
num_bodies = df_all_ans["Body"].size
#D print("Number of bodies: " + str(num_bodies))

# '''
print()
print("For all ans: Cleaning and parsing the training set bodies...")
clean_q_bodies = []
for i in range( 0, num_bodies ):
    clean_q_bodies.append( qa_to_words( df_all_ans["Body"][i] ))
    # Print a progress message for every 10% of i/p data handled.
    if( (i+1)%progress_msg_factor == 0 ):
        clean_qa = qa_to_words( df_all_ans["Body"][i] )
        #D print("\nBody %d of %d" % ( i+1, num_bodies ))
        #D print('  Original text: ' + df_all_ans['Body'][i])
        #D print('  Cleaned text:  ' + clean_qa)


# Write cleaned bodies to a file, one body per line, for visual review.
outfile = tmpdir + a_fname + '.out'
if os.path.exists(outfile):
    os.rename(outfile, outfile + '.bak')
    print('\nWARN: renamed o/p file w/ .bak; save it manually if needed: ' + outfile)
with open(outfile, 'w') as f:
    f.write('\n'.join(clean_q_bodies))
# '''


'''
# TBD Time-stamp: Tue2017_0221_18:04  Use single words.
def make_bag_of_words(clean_q_bodies):
    print("\nCreating the bag of words for word counts ...\n")
    from sklearn.feature_extraction.text import CountVectorizer

    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool
    #
    # Sun2017_0122, Chg max_features from 5000 to 100, for MemErr
    # Sun2017_0219_15:01 , TBD, Is max_features number of words in o/p list?
    #   Seems to be true.
    #
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = 200)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    train_data_features = vectorizer.fit_transform(clean_q_bodies)

    # Numpy arrays are easy to work with, so convert the result to an
    # array
    #
    # TBF, Fails here.
    # Sun2017_0122_19:46 , Prog stops w/ MemErr at this line,
    # when i/p is Questions.csv, after abt 7.5 minutes.
    # Free mem drops steadily to abt 75 MB, then program stops.
    train_data_features = train_data_features.toarray()

    #DBG Wed2017_0118_19:38 , w/ q9999 data, got (727, 1550):
    print('Bag shape: rows (num of records), cols (num of features):')
    print(train_data_features.shape)
    print()

    # Take a look at the words in the vocabulary
    vocab = vectorizer.get_feature_names()
    #D print(vocab)

    # Print counts of each word in vocab.
    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)

    return (vocab, dist)
'''


# TBD Time-stamp: Tue2017_0221_18:04  Use ngrams instead of single words.
def make_bag_of_words(clean_q_bodies):
    print("\nCreating the bag of words for word counts ...\n")
    from sklearn.feature_extraction.text import CountVectorizer

    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool
    #
    # Sun2017_0122, Chg max_features from 5000 to 100, for MemErr
    # Sun2017_0219_15:01 , TBD, Is max_features number of words in o/p list?
    #   Seems to be true.
    #
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 ngram_range = (3,5), \
                                 token_pattern = r'\b\w+\b', \
                                 max_features = 200)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    #
    # TBF, Stuck here.
    # Tue2017_0221_22:42  , Prog uses all VM's memory & much swap & busy disk..
    #   It was still running after 3 hrs, but will probably fail.
    #   Using a6*.csv for i/p.
    train_data_features = vectorizer.fit_transform(clean_q_bodies)

    # Numpy arrays are easy to work with, so convert the result to an
    # array
    #
    # TBF, Fails here.
    # Sun2017_0122_19:46 , Prog stops w/ MemErr at this line,
    # when i/p is Questions.csv, after abt 7.5 minutes.
    # Free mem drops steadily to abt 75 MB, then program stops.
    train_data_features = train_data_features.toarray()

    #DBG Wed2017_0118_19:38 , w/ q9999 data, got (727, 1550):
    print('Bag shape: rows (num of records), cols (num of features):')
    print(train_data_features.shape)
    print()

    # Take a look at the words in the vocabulary
    vocab = vectorizer.get_feature_names()
    #D print(vocab)

    # Print counts of each word in vocab.
    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)

    return (vocab, dist)

print('==== make_bag_of_words(clean_q_bodies')
(vocab, dist) = make_bag_of_words(clean_q_bodies)
# '''

# '''
# For each, print the vocabulary word and the number of times it
# appears in the training set
count_tag = []
word_freq_d = {}
for tag, count in zip(vocab, dist):
    #D print(count, tag)
    count_and_tag = (count, tag)
    # Build list of tuples
    count_tag.append(count_and_tag)
    #D word_freq_d[tag] = count

# Sort the list of tuples by count.
wsbc = words_sorted_by_count = sorted(count_tag, key=lambda x: x[0])

# Write sorted vocab to a file.
outfile = tmpdir + a_fname + '.vocab'
if os.path.exists(outfile):
    os.rename(outfile, outfile + '.bak')
    print('\nWARN: renamed o/p file to *.bak; save it manually if needed:'+ outfile)
with open(outfile, 'w') as f:
    for count, word in words_sorted_by_count:
        print(count, word, file=f)
# '''




#TBD  Tue2017_0124  Remove code temporarily
# Sort data by Score for each record.
print()
print('\n=== Sort data by Score for each record.')
print()
df_score = df_all_ans.sort_values(['Score'])
df_score = df_score[['Id', 'Score']]
#D print(df_score.head())
#D print()
# Compute the number of records to use for computation and display.
rec_selection_ratio = 0.01
num_selected_recs = int(numlines * rec_selection_ratio)
if num_selected_recs < 6:
    num_selected_recs = 5
print('  rec_selection_ratio, number of selected recs: ', rec_selection_ratio, num_selected_recs, '\n')
print('Lowest scoring records:')
#D print(df_score.head(num_selected_recs))
print(df_score.head())
print()
print('Highest scoring records:')
#D print(df_score.tail(num_selected_recs))
print(df_score.tail())
print()


print('\n=== Find most freq words for top-scoring records.')
df_score_top_n = df_score[['Id']]
#D print(df_score_top_n.tail(20))
#D print()
#D print(df_score_top_n.tail(num_selected_recs))
#D print()

# Use top_n records & count their words.
print()
print("For top ans: Cleaning and parsing the training set bodies...")
#D print("Number of bodies: " + str(num_bodies))
top_n_bodies = []
df_score_l = []
# Convert dataframe to list of Id's, to get the body of each Id.
df_score_l = df_score_top_n['Id'].tail(num_selected_recs).tolist()
df8 = df_all_ans.set_index('Id')
for i in df_score_l:
    top_n_bodies.append( qa_to_words( df8["Body"][i] ))
    # Print a progress message for every 10% of i/p data handled.
    if( (i+1)%progress_msg_factor == 0 ):
        clean_qa = qa_to_words( df8["Body"][i] )
        print("\nBody for Id %d " % ( i))
        print('  Original text: ' + df8['Body'][i])
        print('  Cleaned text:  ' + clean_qa)


print('==== make_bag_of_words(top_n_bodies')
(vocab, dist) = make_bag_of_words(top_n_bodies)

#TBD Mon2017_0123_14:37 , Move helper code to funcs.

# For each, print the vocabulary word and the number of times it
# appears in the training set
count_tag = []
word_freq_d = {}
for tag, count in zip(vocab, dist):
    #D print(count, tag)
    count_and_tag = (count, tag)
    # Build list of tuples
    count_tag.append(count_and_tag)
    #D word_freq_d[tag] = count

# Sort the list of tuples by count.
wsbc = words_sorted_by_count = sorted(count_tag, key=lambda x: x[0])

# Write sorted vocab to a file, w/ name suffix 'hiscore'.
outfile = tmpdir + a_fname + '.vocab.hiscore'
if os.path.exists(outfile):
    os.rename(outfile, outfile + '.bak')
    print('\nWARN: renamed o/p file to *.bak; save it manually if needed:'+ outfile)
with open(outfile, 'w') as f:
    for count, word in words_sorted_by_count:
        print(count, word, file=f)


print('\n=== Find most freq words for bottom-scoring records.')
df_score_bot_n = df_score[['Id']]
#D print(df_score_bot_n.head(20))
#D print()
#D print(df_score_bot_n.head(num_selected_recs))
#D print()

# Use bot_n records & count their words.
print()
print("For bottom ans: Cleaning and parsing the training set bodies...")
#D print("Number of bodies: " + str(num_bodies))
bot_n_bodies = []
df_score_l = []
# Convert dataframe to list of Id's, to get the body of each Id.
df_score_l = df_score_bot_n['Id'].head(num_selected_recs).tolist()
df8 = df_all_ans.set_index('Id')
for i in df_score_l:
    bot_n_bodies.append( qa_to_words( df8["Body"][i] ))
    # Print a progress message for every 10% of i/p data handled.
    if( (i+1)%progress_msg_factor == 0 ):
        clean_qa = qa_to_words( df8["Body"][i] )
        print("\nBody for Id %d " % ( i))
        print('  Original text: ' + df8['Body'][i])
        print('  Cleaned text:  ' + clean_qa)


print('==== make_bag_of_words(bot_n_bodies')
(vocab, dist) = make_bag_of_words(bot_n_bodies)

#TBD Mon2017_0123_14:37 , Move helper code to funcs.

# For each, print the vocabulary word and the number of times it
# appears in the training set
count_tag = []
word_freq_d = {}
for tag, count in zip(vocab, dist):
    #D print(count, tag)
    count_and_tag = (count, tag)
    # Build list of tuples
    count_tag.append(count_and_tag)
    #D word_freq_d[tag] = count

# Sort the list of tuples by count.
wsbc = words_sorted_by_count = sorted(count_tag, key=lambda x: x[0])

# Write sorted vocab to a file, w/ name suffix 'loscore'.
outfile = tmpdir + a_fname + '.vocab.loscore'
if os.path.exists(outfile):
    os.rename(outfile, outfile + '.bak')
    print('\nWARN: renamed o/p file to *.bak; save it manually if needed:'+ outfile)
with open(outfile, 'w') as f:
    for count, word in words_sorted_by_count:
        print(count, word, file=f)





'''
# Sort data by OwnerUserId for each record.
print('\n=== Sort data by OwnerUserId for each record.')
print()
df_ouid = df_all_ans.sort_values(['OwnerUserId'])
df_ouid = df_ouid[['Id', 'OwnerUserId', 'Score']]
#D print(df_ouid)
print(df_ouid.head())
print()
print(df_ouid.tail())
print()
'''


'''
# Count length of each record's Body field: number of words.
#TBD, Use w/ or w/o stopwords?
print('\n=== Count number of words in each Body.')
#
#TBD Ref: http://stackoverflow.com/questions/40058436/python-count-frequency-of-words-from-a-column-and-store-the-results-into-anothe
#TBD Explain
df_all_ans['Length'] = df_all_ans['Body'].apply(lambda x: len(x.split()))
df_aa_sort_len = df_all_ans.sort_values(['Length'])
df_aa_sort_len = df_aa_sort_len[['Id', 'OwnerUserId', 'Score', 'Length']]
print()
print(df_aa_sort_len.head())
print()
print(df_aa_sort_len.tail())
print()
'''

'bye'
