# -*- coding: utf-8 -*-

# Using ~/anaconda3/bin/python: Python 3.5.2 :: Anaconda 4.2.0 (64-bit), or later

#   Time-stamp: <Sun 2017 Jun 11 07:19:59 PMPM clpoda>
"""fga_find_good_answers.py


   Find answers in stackoverflow that might be good, but 'hidden'
   because they have low scores.
   Look for such answers from contributors who have high scores
   based on their other questions & answers.
   Save the o/p to a file for further evaluation & processing.

   Usage:
     pydoc  fga_find_good_answers
     python fga_find_good_answers.py

     Set the value of num_owners at the bottom of the file;
     default is 10.  It determines how much o/p data will be saved.

------

Input data format of stackoverflow.com python file from kaggle.com.

==> Answers.csv <==
        Id,OwnerUserId,CreationDate,ParentId,Score,Body
        497,50,2008-08-02T16:56:53Z,469,4,
        "<p>open up a terminal (Applications-&gt;Utilities-&gt;Terminal) and type this in:</p>
..."
==> Questions.csv <==
        Id,OwnerUserId,CreationDate,Score,Title,Body
        469,147,2008-08-02T15:11:16Z,21,How can I find the full path to a font
        from its display name on a Mac?,
        "<p>I am using the Photoshop's javascript API to find the fonts in a given PSD.</p>
..."

Output data format of q_with_a.csv o/p file from this program.
    Note: question records have a Title but no ParentId;
    answer records have a ParentId (which is the related
    question's Id) but no Title.

==> q_with_a.csv <==
        Id,ParentId,OwnerUserId,CreationDate,Score,Title,Body
        5313,,680.0,2008-08-07T21:07:24Z,11,"Cross Platform, Language
        Agnostic GUI Markup Language?",
        "<p>I learned Swing back in the day but now
..."
        5319,5313.0,380.0,2008-08-07T21:10:27Z,8,,<p>erm.. HTML?
        (trying to be funny here... while we wait for real answers..)</p>
        5320,5313.0,216.0,2008-08-07T21:11:28Z,1,,"<p>The
        <a href=""http://www.wxwidgets.org/"" rel=""nofollow""
        title=""wxWidgets"">wxWidgets</a> (formerly known as wxWindows)
..."

"""


# ----------------------------------------------------------
# Plan
# Find top-scoring owners.
# Find all answers by top-scoring owners.
# Find the questions for each of those answers.
# Find all answers for each of those questions.
# Build a data frame with each question followed by all its answers.
# Find the subset of Q's and A's that contain a keyword.
# Save the subset of data to a csv file.
#
# Next steps.
#
# Build tools to analyze a larger test set.
# ----------------------------------------------------------


# ----------------------------------------------------------
# Preliminary steps.
#
# This csvcut (part of csvkit) operation  may not be needed; used for debug.
# Build subset of full Answers data set; exclude Body field:
# csvcut -c 1,2,3,4,5 -e latin1 indir/Answers.csv > outdir/a21_all_iocps.csv
# ----------------------------------------------------------

version = '0.0.5'

import argparse
import nltk
import numpy as np
import os
import pandas as pd
import random

import config as cf
import nltk_ex25 as nl



# ----------------------------------------------------------
# Specify question and parent ID's to find.
#DBG pid_l = [469, 535, 231767]
#DBG.many pid_l = [469, 502, 535, 594, 683, 742, 766, 773, 972, 1476, 766, 1734, 1829, 1854, 1983, 2311, 2933, 3061, 3976, 4942, 5102, 5313, 1983, 5909, 5966, 8692, 8948, 10123, 11060, 2933, 1983]
pid_l = [469, 502, 535, 594, 683, 742, 766, 773, 972]

# ----------------------------------------------------------

log_msg = cf.log_file + ' - Start logging.\n'
cf.logger.info(log_msg)


def main():
    init()
    cf.a_fname, a_infile, q_infile, indir, outdir = config_data()
    cf.all_ans_df, cf.all_ques_df, cf.progress_msg_factor, numlines = read_data(a_infile, q_infile)
    popular_ids = find_popular_ques(cf.all_ans_df, cf.a_fname)
    popular_ids_a = popular_ids.index.values
    top_scoring_owners_a = group_data(cf.all_ans_df)
    parent_id_l = find_question_ids(top_scoring_owners_a, cf.all_ans_df)
    #
    # pop_and_top_l:
    # Find parent IDs that are popular (have several answers), and
    # assume that some of those answers come from top owners
    # (owners who have high mean scores).
    # Set the size of the following list slices to get enough o/p to analyze.
    # With slice limits at 40 and 30, got 2 Q, 36 A.
    # With slice limits at 400 and 300, got 26 Q, 308 A.
    pop_and_top_l = list(set(parent_id_l[:500]).intersection(set(popular_ids_a[:500])))
    print('len(pop_and_top_l) : ', len(pop_and_top_l))
    if args['verbose']:
        print('pop_and_top_l, parent id\'s to examine: ', pop_and_top_l[:])
    q_with_a_df, all_ans_with_hst_df = combine_related_q_and_a(pop_and_top_l, cf.all_ques_df, cf.all_ans_df, numlines)
    #
    #TBD Save the df to a file for review & debug; later processing may
    # use the df & the file is not needed.
    outfile = "tmpdir/all_ans_with_hst.csv"
    all_ans_with_hst_df.to_csv(outfile)
    
    #D print('#D fga, End of debug code; exiting.')
    #D exit()


    # Write full data set to a csv file.
    outfields_l = ['Id', 'ParentId', 'OwnerUserId', 'CreationDate', 'Score', 'Title', 'Body']
    outfile = 'outdir/q_with_a.csv'
    q_with_a_df[outfields_l].to_csv(outfile, header=True, index=None, sep=',', mode='w')
    # DBG  write_df_to_file(q_with_a_df, outdir, cf.a_fname)

    if keyword:
        # Write keyword-containing records to a csv file.
        qa_with_keyword_df = select_keyword_recs(keyword, q_with_a_df, outfields_l)
        outfile = 'outdir/qa_with_keyword.csv'
        qa_with_keyword_df[outfields_l].to_csv(outfile, header=True, index=None, sep=',', mode='w')


def init():
    """Initialize some settings for the program.
    """
    # Initialize settings for pandas.
    pd.set_option('display.width', 0)  # 0=no limit, use for debugging

    # Don't show commas in large numbers.
    # Show OwnerUserId w/o '.0' suffix.
    pd.options.display.float_format = '{:.0f}'.format


def config_data():
    """Configure path and file names for i/o data.
    """
    # TBD Make the in & out dirs w/ this program, if they don't exist?
    # TBD Include the test data files w/ this project.
    indir = 'indir/'  # Relative to pwd, holds i/p files.
    outdir = 'outdir/'  # Relative to pwd, holds o/p files.
    #D cf.a_fname = 'Answers.csv'
    #D cf.q_fname = 'Questions.csv'

    # Smaller data sets, used for debugging.
    #D cf.q_fname = 'q6_999994.csv'
    #D cf.a_fname = 'a6_999999.csv'
    #D cf.a_fname = 'a5_99998.csv'
    #D cf.q_fname = 'q30_99993.csv'
    cf.a_fname = 'a3_986.csv'
    cf.q_fname = 'q3_992.csv'
    #D cf.a_fname = 'a2.csv'
    #D cf.q_fname = 'q2.csv'

    a_infile = indir + cf.a_fname
    q_infile = indir + cf.q_fname

    print('Input files, q & a:\n' + q_infile + '\n' + a_infile)
    print()

    return cf.a_fname, a_infile, q_infile, indir, outdir


def read_data(ans_file, ques_file):
    """Read the csv i/p files and store data into a pandas data frame.
    Compute a factor that dictates how progress will be indicated
    during read operations.
    """
    #TBD, Sat2017_0506_15:34  Maybe rm latin-1 encoding here also?
    ans_df = pd.read_csv(ans_file, encoding='latin-1', warn_bad_lines=False, error_bad_lines=False)
    ques_df = pd.read_csv(ques_file, encoding='latin-1', warn_bad_lines=False, error_bad_lines=False)

    numlines = len(ans_df)
    print('Number of answer records in i/p data frame, ans_df: ' + str(numlines))
    cf.progress_msg_factor = int(round(numlines/10))
    print('\n#D  cf.progress_msg_factor : ' , cf.progress_msg_factor)
    print()
    return ans_df, ques_df, cf.progress_msg_factor, numlines


def find_popular_ques(all_ans_df, a_fname):
    """Find the most frequent ParentIds in the answers df.
    """
    popular_ids = pd.value_counts(all_ans_df['ParentId'])
    outfile = "outdir/fpq_popular_ids."+ a_fname+ ".csv"
    popular_ids.to_csv(outfile)
    return popular_ids


def group_data(all_ans_df):
    """Group the contents of the answers df by a specific column.
    Group by OwnerUserId, and sort by mean score for each owner.
    Make a numpy array of owners w/ highest mean scores.
    TBD.1, Find low scores for these hi-score owners;
    then mark the low score records for evaluation.
    Low score is any score below lo_score_limit.
    """
    print('=== owner_grouped_df: Group by owner and sort by mean score for each owner.')
    owner_grouped_df = all_ans_df.groupby('OwnerUserId').mean()
    owner_grouped_df = owner_grouped_df[['Score']].sort_values(['Score'])

    # Copy index column into owner column; Change index column to integer
    owner_grouped_df['OwnerUserId'] = owner_grouped_df.index
    owner_grouped_df.reset_index(drop=True, inplace=True)
    owner_grouped_df.rename(columns={'Score': 'MeanScore'}, inplace=True)

    print()
    print('len(owner_grouped_df): number of unique OwnerUserId values: ' + str(len(owner_grouped_df)))
    print()
    if args['verbose']:
        print('Show owners with ', str(num_owners), ' highest MeanScores.')
        print(owner_grouped_df.tail(num_owners))  # See highest scores at bottom:
        print()

    # Take slice of owners w/ highest mean scores; convert to int.
    owners_a = owner_grouped_df['OwnerUserId'].values
    top_scoring_owners_a = np.vectorize(np.int)(owners_a[-num_owners:])
    # D print('top_scoring_owners_a: ', top_scoring_owners_a )
    # D print()

    owners_df_l = []
    lo_score_limit = args['lo_score_limit']
    for owner in top_scoring_owners_a:
        # Get a pandas series of booleans for filtering:
        answered_by_o2_sr = (all_ans_df.OwnerUserId == owner)
        # Get a pandas df with rows for all answers of one user:
        answers_df = all_ans_df[['Id', 'OwnerUserId', 'Score']][answered_by_o2_sr]

        # Get a pandas series of booleans for filtering:
        lo_score_by_o2_sr = (answers_df.Score < lo_score_limit)
        # Get a pandas df with rows for all low-score answers of one user:
        lo_score_answers_by_o2_df = answers_df[['Id', 'OwnerUserId', 'Score']][lo_score_by_o2_sr]
        owners_df_l.append(lo_score_answers_by_o2_df)

    #TBD These are the answers to examine for useful data, even though
    # they have low scores.
    # View these answers and evaluate them manually; and analyze them
    # with other s/w.
    lo_scores_for_top_users_df = pd.concat(owners_df_l)
    print('lo_score_limit: ', lo_score_limit)
    print('Length of lo_scores_for_top_users_df: ', len(lo_scores_for_top_users_df))
    outfile = 'outdir/lo_scores_for_top_users.csv'
    lo_scores_for_top_users_df.to_csv(outfile, header=True, index=None, sep=',', mode='w')
    if args['verbose']:
        print('lo_scores_for_top_users_df: ')
        print(lo_scores_for_top_users_df)
    print()

    return top_scoring_owners_a


def find_question_ids(top_scoring_owners_a, all_ans_df):
    """Make a list of all answer records by the high-score owners.
    Use that list to build a list of Question Id's (= ParentId)
    to collect for evaluation.
    Return that list of question Id's.
    """
    owners_df_l = []
    for owner in top_scoring_owners_a:
        # Get a pandas series of booleans for filtering:
        answered_by_owner_sr = (all_ans_df.OwnerUserId == owner)
        # Get a pandas df with rows for all answers of one user:
        answers_df = all_ans_df[['Id', 'OwnerUserId', 'ParentId', 'Score']][answered_by_owner_sr]
        owners_df_l.append(answers_df)

    hi_scoring_users_df = pd.concat(owners_df_l)
    # D print('Length of hi_scoring_users_df: ', len(hi_scoring_users_df))
    # D print('hi_scoring_users_df: ')
    # D print(hi_scoring_users_df)

    # Get list of unique ParentId's:
    parent_id_l = list(set(hi_scoring_users_df['ParentId']))
    # D print('parent_id_l: ')
    # D print(parent_id_l)
    # D print()
    return parent_id_l


def combine_related_q_and_a(pop_and_top_l, all_ques_df, all_ans_df, numlines):
    """Get each Q in the list of ParentId's, and the related A's.
    Loop over all the question Id's and store all Q & A data in a df.
    Return that df.
    Update, Fri2017_0609_18:28 : This function now includes calls
    to the routines that perform the natural language processing
    of the text data.  The code will probably be reorganized into
    other functions.
    """
    all_ans_with_hst_df = pd.DataFrame()

    ques_match_df = all_ques_df[all_ques_df['Id'].isin(pop_and_top_l)]
    ans_match_df = all_ans_df[all_ans_df['ParentId'].isin(pop_and_top_l)]
    q_with_a_df = pd.concat([ques_match_df, ans_match_df]).reset_index(drop=True)
        # Full list w/ all Q's at top, A's after.
    #
    print('#D len of ques_match_df: ', len(ques_match_df ))
    print('#D len of ans_match_df: ', len(ans_match_df ))
    print('\n#D ques_match_df.head() & ans_match_df: ')
    print(ques_match_df.head() )
    print('#D')
    print(ans_match_df.head() )
    #
    i = 0
    # Build each Q&A group: one Q w/ all its A.'s
    #TBD, How to do this w/o explicit loop, using df tools?
    #TBD, Combine or replace the ques_match_df & ans*df code above w/ this?
        # Maybe delete those 'intermediate' results?
    for qid in pop_and_top_l:
        i += 1
        #OK if(i % cf.progress_msg_factor == 0):
        if(i % 20 == 0):
            print("#D combine_related_q_and_a:for-qid-loop count i: ", i)
        qm_df = ques_match_df[ques_match_df['Id'] == qid]
        am_df = ans_match_df[ans_match_df['ParentId'] == qid]
        qag_df = pd.concat([qm_df, am_df]).reset_index(drop=True)
        #D print("\n#D qag_df.head(): ")
        #D print(qag_df.head())
        cf.logger.info('qag_df.head(1): ')
        cf.logger.info( qag_df.head(1))
        #
        cf.all_ans_df = qag_df  #TMP to avoid renaming all_ans_df in many places
        cf.logger.info("Step 2. Process the words of each input line.")
        clean_ans_bodies_l = nl.clean_raw_data(cf.a_fname, cf.progress_msg_factor )
        #D print('\n#D, clean_ans_bodies_l[:1]')
        #D print(clean_ans_bodies_l[:1])
        #
        cf.logger.info("Step 3. Build a bag of words and their counts.")
        (vocab, dist) = nl.make_bag_of_words(clean_ans_bodies_l)
        #D print('\n#D, vocab[:1]')
        #D print(vocab[:1])
        words_sorted_by_count_l = nl.sort_save_vocab('.vocab', vocab, dist, cf.a_fname)
        # Save the original list for later searching.
        words_sorted_by_count_main_l = words_sorted_by_count_l
        #
        cf.logger.info('Step 4. Sort Answers by Score.')
        score_df, num_selected_recs = nl.sort_answers_by_score(numlines)
        #
        cf.logger.info('Step 5. Find most freq words for top-scoring Answers.')
        score_top_n_df = score_df[['Id']]
        
        #OK.tbr  print("DBG.103", score_top_n_df.tail(num_selected_recs), '\n')
        #TBD, Maybe convert df to string so logger can print title & data w/ one cmd:
        #TBD log_msg = "score_top_n_df.tail():" + CONVERT_DF_TO_STRING(score_top_n_df.tail())
        #TBD cf.logger.debug(log_msg)
        cf.logger.debug("score_top_n_df.tail():")
        cf.logger.debug(score_top_n_df.tail(20)) 
        # Use top_n Answers & count their words.
        cf.logger.info("For top ans: Cleaning and parsing the training set bodies...")
        # 
        top = True
        top_n_bodies = nl.find_freq_words(top, score_top_n_df, num_selected_recs, cf.progress_msg_factor)
        cf.logger.info('make_bag_of_words(top_n_bodies)')
        (vocab, dist) = nl.make_bag_of_words(top_n_bodies)
        nl.sort_save_vocab('.vocab.hiscore', vocab, dist, cf.a_fname)
        #
        cf.logger.info("Step 6. Find most frequent words for bottom-scoring Answers.")
        # Keep these data to compare w/ words for top-scoring Answers; s/b some diff.
        # If they are identical, there may be a logic problem in the code.
        score_bot_n_df = score_df[['Id']]
        cf.logger.debug("score_bot_n_df.head():")
        cf.logger.debug(score_bot_n_df.head(20))
        #TBR print(score_bot_n_df.head(num_selected_recs), '\n')
        top = False
        bot_n_bodies = nl.find_freq_words(top, score_top_n_df, num_selected_recs, cf.progress_msg_factor)
        cf.logger.info('make_bag_of_words(bot_n_bodies)')
        (vocab, dist) = nl.make_bag_of_words(bot_n_bodies)
        nl.sort_save_vocab('.vocab.loscore', vocab, dist, cf.a_fname)
        #
        cf.logger.info("Step 7. Search lo-score A's for hi-score text.")
        ans_with_hst_df = nl.search_for_terms(words_sorted_by_count_main_l, clean_ans_bodies_l, num_hi_score_terms)
        all_ans_with_hst_df = pd.concat([all_ans_with_hst_df, ans_with_hst_df]).reset_index(drop=True)

        
        
        #
        #TBD.Thu2017_0608_23:58 
        # Analyze qag_df w/ nlp s/w in this loop; 
        # or save each qag_df to a separate df for later processing;
        # or save each qag_df to a disk file for later processing;
        # or move this code to the nlp processing section of the program.
        #
    #
    #D print('\n#D fga, End of debug code; exiting.')
    #D exit()

    return q_with_a_df, all_ans_with_hst_df

# TBD.1 Sat2017_0211_22:55 , should this func be called before combine_related*()?
# Use it to make the final pop_and_top_l?
#
def select_keyword_recs(keyword, q_with_a_df, outfields_l):
    """Find the Q's & A's from the filtered df that contain the keyword,
    in Title or Body.
    Combine the sets into one set of unique Q's w/ their A's.
    Save all the selected data for analysis.
    TBD, In future, search entire collection of Q&A, not just
    the filtered subset.
    """
    # Get a pandas series of booleans to find the current question id.
    # Check Question & Answer, both Title and Body columns.
    qt_sr  = q_with_a_df.Title.str.contains(keyword, regex=False)
    qab_sr = q_with_a_df.Body.str.contains(keyword, regex=False)
    # Combine two series into one w/ boolean OR.
    qa_contains_sr = qab_sr | qt_sr
    qa_df = q_with_a_df[outfields_l][qa_contains_sr]
    return qa_df


def write_df_to_file(in_df, wdir, wfile):
    """Write one column of a pandas data frame to a file w/ suffix '.qanda'.
    """
    # Used for testing and debugging.
    outfile = wdir + wfile + '.qanda'
    if os.path.exists(outfile):
        os.rename(outfile, outfile + '.bak')
        print('\nWARN: renamed o/p file to *.bak; save it manually if needed:' + outfile)
    with open(outfile, 'w') as f:
        print('\nWriting Q and A to outfile: ' + outfile)
        print(in_df['Body'], file=f)


def get_parser():
    """Create parser to specify cmd line options for this program.
    """
    parser = argparse.ArgumentParser(description='find good answers hidden in stackoverflow data')

    parser.add_argument('-L', '--lo_score_limit', help='lowest score for an answer to be included', default=10, type=int)

    #TBD parser.add_argument('-p', '--popular_questions', help='select questions with many answers', action='store_true')
    #TBD parser.add_argument('-t', '--top_users', help='find lo-score answers by hi-scoring owners', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = vars(parser.parse_args())

    # Set the number of top scoring owners to select from the data.
    num_owners = 10  # Default is 10.
    num_owners = 40  # Default is 10.
    num_owners = 200  # Default is 10.
    print("num_owners: ", num_owners)
    #
    keyword = False
    #D keyword = 'beginner'
    #D keyword = 'yield'
    # D keyword = 'begin'
    # D keyword = 'pandas'
    #D keyword = 'Python'  # Both Title & Body of data sets have it; for debug
    print("Keyword: ", keyword)
    #
    num_hi_score_terms = 22  # Use 3 for testing; 11 or more for use.
    print("num_hi_score_terms: ", num_hi_score_terms)

    main()
    log_msg = cf.log_file + " - Finish program & logging.\n\n"
    cf.logger.warning(log_msg)

