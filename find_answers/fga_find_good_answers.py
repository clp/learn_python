# -*- coding: utf-8 -*-
# Using ~/anaconda3/bin/python: Python 3.5.2 :: Anaconda 4.2.0 (64-bit)

#   Time-stamp: <Tue 2017 Feb 07 02:58:24 PMPM clpoda> 
"""fga_find_good_answers.py

   Usage:
     fga_find_good_answers.py 

     Set the value of num_owners below; default is 10.
     It determines how much o/p data will be saved.

   Options:
     TBD: -h  --help  Show this help data.
     TBD: --version   Show version

------

Data format of stackoverflow.com python file from kaggle.com.

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

Data format of q_with_a.csv o/p file from this program.
    Note: question records have a Title but no ParentId;
    answer records have a ParentId but no Title.

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


#----------------------------------------------------------
# Plan
# Loop over top scoring owners of answers.
#   Find next top-scoring owner.
#   Use pandas.describe() or other tool to find stats.
#   Loop over each record by ParentId in the bottom N % (0.1-1-10-25%) of scores
#     Use lowest scores to find answers that might be good & deserve higher score.
#     Get Q.Body of the related question from Questions.csv.
#     Get other answers with that ParentId from Answers.csv.
#     Save the Q & all related A's: append to data frame.
# Print a report to a csv file w/ all Q's & all A's.
#
# Next steps.
#
# Manually review the A's.
#   Mark the answer w/ value tag: Is it worth reading?: Y or N.
#     TBD, Consider using Hi-M-Lo or an integer to add to its score.
# Use these data to learn about the data; then analyze a larger test set.
#----------------------------------------------------------


#----------------------------------------------------------
# Preliminary steps.
#
# This csvcut (part of csvkit) operation  may not be needed; used for debug.
# Build subset of full Answers data set; exclude Body field:
# csvcut -c 1,2,3,4,5 -e latin1 indir/Answers.csv > outdir/a21_all_iocps.csv
#----------------------------------------------------------


import pandas as pd
import numpy as np
import os


def main():
    init()
    a_fname, a_infile, q_infile, indir, outdir = config_data()
    all_ans_df, all_ques_df, progress_msg_factor = read_data(a_infile, q_infile)
    top_scoring_owners_a = group_data(all_ans_df)
    parent_id_l = find_question_ids(top_scoring_owners_a, all_ans_df)
    q_with_a_df = combine_related_q_and_a(parent_id_l, all_ques_df, all_ans_df)
    # Write df to a csv file.
    outfile = 'outdir/q_with_a.csv'
    q_with_a_df[['Id', 'ParentId', 'OwnerUserId', 'CreationDate', 'Score', 'Title', 'Body']].to_csv(outfile, header=True, index=None, sep=',', mode='w')
    #DBG  write_df_to_file(q_with_a_df, outdir, a_fname)


def init():
    """Initialize some settings for pandas.
    """
    pd.set_option('display.width', 80)
    pd.options.display.float_format = '{:.0f}'.format  # Don't show commas in large numbers
        # To show OwnerUserId w/o '.0' suffix; see b.10.


def config_data():
    """Configure path and file names for i/o data.
    """
    #TBD Make the in & out dirs w/ this program, if they don't exist?
    #TBD Include the test data files w/ this project.
    indir = 'indir/'  # Relative to pwd, holds i/p files.
    outdir = 'outdir/'  # Relative to pwd, holds o/p files.
    a_fname = 'Answers.csv'      
    q_fname = 'Questions.csv'
    a_fname = 'a5_99998.csv'  
    q_fname = 'q30_99993.csv'
    #D q_fname = 'q3_992.csv'
    #D a_fname = 'a3_986.csv'   
    #D q_fname = 'q2.csv'
    #D a_fname = 'a2.csv'

    a_infile = indir  + a_fname
    q_infile = indir  + q_fname

    print('Input files, q & a:\n'  + q_infile + '\n' + a_infile)
    print()

    return a_fname, a_infile, q_infile, indir, outdir


def read_data(ans_file, ques_file):
    """Read the csv i/p files and store into a pandas data frame.
    Compute a factor that dictates how progress will be indicated
    during read operations.
    """
    ans_df = pd.read_csv(ans_file, encoding='latin-1', warn_bad_lines=False, error_bad_lines=False)
    ques_df = pd.read_csv(ques_file, encoding='latin-1', warn_bad_lines=False, error_bad_lines=False)

    numlines = len(ans_df)
    print('Number of answer records in i/p data frame, ans_df: ' + str(numlines))
    progress_msg_factor = int(round(numlines/10))
    print()
    return ans_df, ques_df, progress_msg_factor


def group_data(all_ans_df):
    """Group the contents of the answers df by a specific column.
    Group by OwnerUserId, ouid, and sort by mean score for each ouid.
    Make a numpy array of owners w/ highest mean scores.
    TBD.1, Find low of scores for these hi-score owners; 
    then mark the low score records for evaluation.
    Low score is any score below lo_score_limit.
    """
    print('\n=== ouid_grouped_df: Group by owner and sort by mean score for each owner.')
    ouid_grouped_df = all_ans_df.groupby('OwnerUserId').mean()
    ouid_grouped_df = ouid_grouped_df[['Score']].sort_values(['Score'])

    # Copy index column into ouid column; Change index column from ouid to integer
    ouid_grouped_df['OwnerUserId'] = ouid_grouped_df.index
    ouid_grouped_df.reset_index(drop=True, inplace=True)
    ouid_grouped_df.rename(columns = {'Score' : 'MeanScore'}, inplace=True)

    print()
    print('len(ouid_grouped_df): number of unique OwnerUserId values: ' + str(len(ouid_grouped_df)))
    print()
    print('Show owners with ', str(num_owners), ' highest MeanScores.')
    print(ouid_grouped_df.tail(num_owners))  # See highest scores at bottom:
    print()

    # Take slice of owners w/ highest mean scores; convert to int.
    owners_a = ouid_grouped_df['OwnerUserId'].values
    top_scoring_owners_a = np.vectorize(np.int)(owners_a[-num_owners:])
    print('top_scoring_owners_a: ', top_scoring_owners_a )
    print()

    # Make a list of the A scores for each of the hi-score owners.
    print('\n=== ouid_grouped_df: Group by owner .')
    #ORG ouid_grouped_df = all_ans_df.groupby('OwnerUserId')
    o2_grouped_df = all_ans_df.groupby('OwnerUserId')
    #ORG ouid_grouped_df = ouid_grouped_df[['Score']].sort_values(['Score'])

    o2_df_l = []
    lo_score_limit = 10
    for o2 in top_scoring_owners_a:
        # Get a pandas series of booleans for filtering:
        answered_by_o2_b = (all_ans_df.OwnerUserId == o2)
        # Get a pandas df with rows for all answers of one user:
        answers_by_o2_df = all_ans_df[['Id', 'OwnerUserId', 'Score']][answered_by_o2_b]
        # Get a pandas series of booleans for filtering:
        lo_score_by_o2_b = (answers_by_o2_df.Score < lo_score_limit)
        # Get a pandas df with rows for all low-score answers of one user:
        lo_score_answers_by_o2_df = answers_by_o2_df [['Id', 'OwnerUserId', 'Score']][lo_score_by_o2_b]
        o2_df_l.append(lo_score_answers_by_o2_df)

    lo_scores_for_top_users_df = pd.concat(o2_df_l)
    print('Length of lo_scores_for_top_users_df: ', len(lo_scores_for_top_users_df))
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
    ouids_df_l = []
    for ouid in top_scoring_owners_a:
        # Get a pandas series of booleans for filtering:
        answered_by_ouid_b = (all_ans_df.OwnerUserId == ouid)
        # Get a pandas df with rows for all answers of one user:
        answers_by_ouid_df = all_ans_df[['Id', 'OwnerUserId', 'ParentId', 'Score']][answered_by_ouid_b]
        ouids_df_l.append(answers_by_ouid_df)

    hi_scoring_users_df = pd.concat(ouids_df_l)
    print('Length of hi_scoring_users_df: ', len(hi_scoring_users_df))
    #D print('hi_scoring_users_df: ')
    #D print(hi_scoring_users_df)

    # Get list of unique ParentId's:
    parent_id_l = list(set(hi_scoring_users_df['ParentId']))
    print('parent_id_l: ')
    print(parent_id_l)
    print()
    return parent_id_l


def combine_related_q_and_a(parent_id_l, all_ques_df, all_ans_df):
    """Get each Q in the list of ParentId's, and the related A's.
    Loop over all the question Id's and store all Q & A data in a list.
    Convert the list to a df.  Return that df.
    """
    ques_ans_l = []
    for qid in parent_id_l:
        # Get a pandas series of booleans to find the current question id.
        ques_match_b = (all_ques_df.Id == qid)
        ques_match_df = all_ques_df[['Id', 'OwnerUserId',  'CreationDate', 'Score', 'Title', 'Body']][ques_match_b]
        #
        # Append current question to the list.
        ques_ans_l.append(ques_match_df)
        #
        # Get a pandas series of booleans to find all A's related to the current Q.
        ans_match_b = (all_ans_df.ParentId == qid)
        df = all_ans_df[['Id', 'ParentId', 'OwnerUserId',  'CreationDate', 'Score', 'Body']][ans_match_b]
        #
        # Append all related answers to the list.
        ques_ans_l.append(df)
    q_with_a_df = pd.concat(ques_ans_l)
    return q_with_a_df


def write_df_to_file(in_df, wdir, wfile):
    """Write one column of a pandas data frame to a file w/ suffix 'qanda'.
    """
    # Used for testing and debugging.
    outfile = wdir + wfile + '.qanda'
    if os.path.exists(outfile):
        os.rename(outfile, outfile + '.bak')
        print('\nWARN: renamed o/p file to *.bak; save it manually if needed:'+ outfile)
    with open(outfile, 'w') as f:
        print('\nWriting Q and A to outfile: '+ outfile)
        print(in_df['Body'], file=f)


if __name__ == '__main__':
    # Set the number of top scoring owners to select from the data.
    num_owners = 10  # Default is 10.
    num_owners = 40  # Default is 10.
    main()

