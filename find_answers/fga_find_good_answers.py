# -*- coding: utf-8 -*-
#
# Using ~/anaconda3/bin/python: Python 3.5.2 :: Anaconda 4.2.0 (64-bit)
# Using Python 3.4.5 :: Anaconda 4.3.0 (64-bit), since Tue2017_0710
# Requires Python 3; does not work w/ Python 2.
#
"""fga_find_good_answers.py


Find answers in stackoverflow data that might be good,
but 'hidden' because they have low scores.
Look for such answers from contributors who have high scores
based on their other questions & answers.
Save the o/p to a file for further evaluation & processing.

Usage:
    fga_find_good_answers.sh
    python fga_find_good_answers.py
    pydoc  fga_find_good_answers


Initialization

    Set the value of num_owners at the bottom of the file;
    default is 10.  It determines how much o/p data will be saved.

    Set the value of num_hi_score_terms at the bottom of the file;
    It determines how many hi-score terms are used to search
    text in low-score answers.  A bigger number will cause the
    program to run longer, and might find more 'hidden' answers
    that could be valuable.

    Set the value of keyword at the bottom of the file;
    set it to False to skip this feature.  This feature is not
    complete; for now, it collects the questions and answers
    that contain the keyword and writes them to a file.


Overview of Program Actions

    Read the separate question and answer i/p files.

    Find popular questions, those which have several answers.

    Find top-scoring owners, those who have the highest mean scores.

    Select some questions that are popular AND that have answers
    from top-scoring owners.  These are the pop_and_top questions.

    Build a combined collection of pop_and_top questions with their
    answers into Q&A groups.  The NLP tools can work with these data.

    Show a menu with choices, to examine the data or to analyze
    the data and present various plots, eg, scatter plots or
    histogram plots.  Several menu choices require the NLP-ready
    data.

    Some menu actions use the following functions.

    build_stats():
    Compute and save statistics about the selected data, to use
    in further analysis and plotting, eg, owner reputation (mean
    score) and length of the body text of the record.

    check_owner_reputation():
    Provide reputation data from a data frame or a file.  If it
    does not exist, call the function to build it.

    group_data():
    Group the data by owner, and sort by mean score.

    analyze_text():
    Call functions in the nltk module to analyze the text data and
    to provide data that might reveal good but 'hidden' answers.


Other Actions

    Several functions draw specific plots, using matplotlib tools.

    Several functions write data to disk in csv or html formats.

    One program option is to select records that contain
    a specific keyword, and save them to a file.

----------------------------------------------------------

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

Output data format of popular_qa.csv o/p file from this program.
    Note: Question records have a Title but no ParentId.
    Answer records have a ParentId (which is the related
    question's Id) but no Title.

==> popular_qa.csv <==
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


----------------------------------------------------------
Plan
    Find top-scoring owners.
    Find all answers by top-scoring owners.
    Find the questions for each of those answers.
    Find all answers for each of those questions.
    Build a data frame with each question followed by all its answers.
    Find the subset of Q's and A's that contain a keyword.
    Save the subset of data to a csv file.
----------------------------------------------------------


Requirements
    Python 3, tested with v 3.6.1.
    pytest for tests, tested with v 3.0.7.
    TBD

"""

import argparse
import nltk
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np
import os
import pandas as pd
import random
from pandas.plotting import scatter_matrix

import config as cf
import nltk_ex25 as nl


log_msg = cf.log_file + ' - Start logging for ' + os.path.basename(__file__)
cf.logger.info(log_msg)


DATADIR = 'data/'
FILEA3 = 'a3_986.csv'
INDIR = 'indir/'
LINE_COUNT = 10
MAX_COL_WID = 20
MAX_PRINT_SIZE = 60
TMP = 'tmp/'
TMPDIR = 'data/'
popular_qa_df = pd.DataFrame()
search_qa_df = pd.DataFrame()

HEADER = '''
<html>
    <head>
        <style type="text/css">
        table{
            /* max-width: 850px; */
            width: 100%;
        }

        th {
            overflow: auto;  /* Use auto to get H scroll bars */
            text-align: left;
            max-width: 50px; /* Was 500px; */
            min-width: 1px; /* TBD,Wed2018_0103_19:14  New spec */
            width: auto;    /* TBD, Was 100%; */
        }

        td {
            overflow: auto;  /* Use auto to get H scroll bars */
            text-align: left;
            max-width: 500px;
            min-width: 10px;  /* TBD,Wed2018_0103_19:16  new */
            width: auto;    /* TBD, Was 100%; */
        }

        pre,img {
            padding: 0.1em 0.5em 0.3em 0.7em;
            border-left: 11px solid #ccc;
            margin: 1.7em 0 1.7em 0.3em;
            overflow: auto;  /* Use auto to get H scroll bars */
        }
        </style>
    </head>
    <body>
'''
FOOTER = '''
    </body>
</html>
'''


def main(popular_qa_df):
    """Read input data and prepare a subset of question
    and answer records for analysis by natural language
    processing (NLP) tools, which reside in external
    modules.
    When finished, present user with a menu and prompt
    them for the next action.
    """

    global all_ans_df
    #TBD,Fri2017_1222_00:19 , quick fix; may be a problem;; needs cleanup.
    global all_ques_df, num_selected_recs, a_fname, progress_msg_factor

    init()

    a_fname, a_infile, q_infile = \
        config_data()

    all_ans_df, all_ques_df, progress_msg_factor, numlines = \
        read_data(a_infile, q_infile)

    popular_ids = \
        find_popular_ques(all_ans_df, a_fname)

    popular_ids_a = popular_ids.index.values

    top_scoring_owners_a, owner_grouped_df = \
        group_data(all_ans_df)

    ques_ids_from_top_own_l = \
        find_ques_ids_from_top_owners(top_scoring_owners_a, all_ans_df)

    ques_ids_pop_and_top_l = \
        find_pop_and_top_ques_ids(ques_ids_from_top_own_l, popular_ids_a)

    num_selected_recs = compute_record_selector(numlines)

    popular_qa_df = \
        combine_related_q_and_a(
            ques_ids_pop_and_top_l, all_ques_df, all_ans_df, num_selected_recs, a_fname, progress_msg_factor)
    """
    #TBD,Sat2017_1223_13:24 , Disable all this code to debug problem; 
        # Maybe rm it and go directly to show_menu() for user i/p.
    #TBR, maybe not the right place for this test.
    #TBD,Thu2017_1221_20:44 , Wrong test, the df has len=130 here so it doesn't stop.
        # Problem is when trying to analyze text w/ empty i/p df.
    # Test df and stop program if empty.
    print('#D len(popular_qa_df): ', len(popular_qa_df))
    if popular_qa_df.empty:
        print('#D, main(), empty df from combine_related_q_and_a(); cannot proceed; debug.')
        raise SystemExit()

    # Save a df to a file for review & debug.
    write_full_df_to_csv_file(popular_qa_df, TMPDIR, 'popular_qa.csv')

    columns_l = []
    write_full_df_to_html_file(popular_qa_df, TMPDIR, 'popular_qa.html', columns_l)

    # Save the Q&A title & body data as HTML.
    columns_l = ['Id', 'Title', 'Body']
    write_full_df_to_html_file(popular_qa_df, TMPDIR, 'popular_qa_title_body.html', columns_l)
    """




    """
    #TBD,Fri2017_1222_16:51 , Rm this call, maybe it was for debugging; only search when user asks.
    if keyword:
        #TBR, maybe not the right place for this test  #TBD,Thu2017_1221_20:48 .
        # Test df and stop program if empty.
        print('#D len(popular_qa_df): ', len(popular_qa_df))
        if popular_qa_df.empty:
            print('#D, main(), empty df from combine_related_q_and_a(); cannot proceed; debug.')
            raise SystemExit()

        # Write records containing keywords to a csv file.
        q_a_group_with_keyword_df = select_keyword_recs(
            keyword, popular_qa_df, columns_l,
            all_ques_df, all_ans_df, num_selected_recs, a_fname, progress_msg_factor)
        outfile = DATADIR + 'qa_with_keyword.csv'
        q_a_group_with_keyword_df[columns_l].to_csv(
            outfile, header=True, index=None, sep=',', mode='w')
    """


def check_install():
    """Check that some required directories and files exist.
    """
    if not os.path.isdir(DATADIR):
        print("ERR, Did not find data/ dir; re-install the fga s/w.")
        raise SystemExit()

    if not os.path.isdir(INDIR):
        print("ERR, Did not find indir/ dir; re-install the fga s/w.")
        raise SystemExit()

    if not os.path.isfile(INDIR + FILEA3):
        print("ERR, Did not find this input file; re-install the fga s/w: ", INDIR + FILEA3)
        raise SystemExit()


def init():
    """Initialize some settings for the program.
    """
    if cli_args_d['debug']:
        end = 55
        print('Running in debug mode.')
        print('  end set to: ', end)
        print()

    # Initialize settings for pandas.
    pd.set_option('display.width', 0)  # 0=no limit, for debug
    pd.set_option('display.max_colwidth', MAX_COL_WID)  # -1=no limit, for debug

    # Don't show commas in large numbers.
    # Show OwnerUserId w/o '.0' suffix.
    pd.options.display.float_format = '{:.0f}'.format


def config_data():
    """Configure path and file names for i/o data.
    """
    #D a_fname = 'Answers.csv'
    #D q_fname = 'Questions.csv'

    # Smaller data sets, used for debugging.
    q_fname = 'q6_999994.csv'
    a_fname = 'a6_999999.csv'
    a_fname = 'a5_99998.csv'
    q_fname = 'q30_99993.csv'
    a_fname = 'a3_986.csv'
    q_fname = 'q3_992.csv'
    # D a_fname = 'a2.csv'
    # D q_fname = 'q2.csv'

    a_infile = INDIR + a_fname
    q_infile = INDIR + q_fname
    a_out_vocab_file = DATADIR + a_fname + '.vocab'

    print('Input files, q & a:\n' + q_infile + '\n' + a_infile)
    print('Output vocab file:  ' + a_out_vocab_file)
    print()

    return a_fname, a_infile, q_infile


def read_data(ans_file, ques_file):
    """Read the csv i/p files and store data into pandas data frames.
    Compute a factor that dictates how progress will be indicated
    during read operations.
    """
    ans_df = pd.read_csv(
        ans_file,
        encoding='latin-1',
        warn_bad_lines=False,
        error_bad_lines=False)
    ques_df = pd.read_csv(
        ques_file,
        encoding='latin-1',
        warn_bad_lines=False,
        error_bad_lines=False)

    numlines = len(ans_df)
    print('Number of answer records in i/p data frame, ans_df: ' + str(numlines))
    progress_msg_factor = int(round(numlines / 10))
    print('\n#D  progress_msg_factor : ', progress_msg_factor)
    print()
    return ans_df, ques_df, progress_msg_factor, numlines


def find_popular_ques(aa_df, a_fname):
    """Find the most frequent ParentIds in the answers dataframe.
    """
    popular_ids = pd.value_counts(aa_df['ParentId'])
    outfile = DATADIR + "fpq_popular_ids." + a_fname + ".csv"
    popular_ids.to_csv(outfile)
    return popular_ids


#TBD, Refactor the two group_data() funcs.
    # Thu2017_0907_23:30 , Should lo_score*df be returned for use elsewhere?
    # It is printed to file, maybe to log.
    # Is it needed to find good answers?
    # Also chk find_ques_ids_from_top_owners() ;  common code.
    # Has copied lines from group_data().
    # Does not have lo_score* vars & code.
def gd2_group_data(aa_df):
    """Group the contents of the answers dataframe by a specific column.
    Group by OwnerUserId, and sort by mean score for answers only
    for each owner (question scores are not counted).
    """
    print('#D gd2: owner_grouped_df: Group by owner and sort by mean score for each owner.')
    owner_grouped_df = aa_df.groupby('OwnerUserId')
    owner_grouped_df = owner_grouped_df[[
        'Score']].mean().sort_values(['Score'])

    # Copy index column into owner column; Change index column to integer
    owner_grouped_df['OwnerUserId'] = owner_grouped_df.index
    owner_grouped_df.reset_index(drop=True, inplace=True)
    owner_grouped_df.rename(columns={'Score': 'MeanScore'}, inplace=True)

    print()
    print('#D gd2: len(owner_grouped_df): number of unique OwnerUserId values: ' +
          str(len(owner_grouped_df)))
    print()
    cf.logger.info('gd2_group_data(): Show owners with highest MeanScores.')
    cf.logger.info(owner_grouped_df.tail(num_owners))
    if cli_args_d['verbose']:
        print('#D gd2: Show owners with ', str(num_owners), ' highest MeanScores.')
        # See highest scores at bottom of sorted list:
        print(owner_grouped_df.tail(num_owners))
        print()

    return owner_grouped_df


def group_data(aa_df):
    """Group the contents of the answers dataframe by a specific column.
    Group by OwnerUserId, and sort by mean score for answers only
    for each owner (question scores are not counted).
    Make a numpy array of owners w/ highest mean scores.

    TBD.1, Find low score answers for these hi-score owners;
    then mark the low score  answers for evaluation.
    Low score is any score below lo_score_limit.
    """
    print('group_data(): Group by owner and sort by mean score for each owner.')
    owner_grouped_df = aa_df.groupby('OwnerUserId')
    owner_grouped_df = owner_grouped_df[[
        'Score']].mean().sort_values(['Score'])

    # Copy index column into owner column; Change index column to integer
    owner_grouped_df['OwnerUserId'] = owner_grouped_df.index
    owner_grouped_df.reset_index(drop=True, inplace=True)
    owner_grouped_df.rename(columns={'Score': 'MeanScore'}, inplace=True)

    print()
    print('len(owner_grouped_df): number of unique OwnerUserId values: ' +
          str(len(owner_grouped_df)))
    print()
    cf.logger.info('group_data(): Show owners with ... highest MeanScores.')
    cf.logger.info(owner_grouped_df.tail(num_owners))
    if cli_args_d['verbose']:
        print('Show owners with ', str(num_owners), ' highest MeanScores.')
        # See highest scores at bottom:
        print(owner_grouped_df.tail(num_owners))
        print()

    # Take slice of owners w/ highest mean scores; convert to int.
    owners_a = owner_grouped_df['OwnerUserId'].values
    top_scoring_owners_a = np.vectorize(np.int)(owners_a[-num_owners:])
    # D print('top_scoring_owners_a: ', top_scoring_owners_a)
    # D print()

    owners_df_l = []
    lo_score_limit = cli_args_d['lo_score_limit']
    for owner in top_scoring_owners_a:
        # Get a pandas series of booleans for filtering:
        answered_by_o2_sr = (aa_df.OwnerUserId == owner)
        # Get a pandas df with rows for all answers of one user:
        answers_df = aa_df[[
            'Id', 'OwnerUserId', 'Score']][answered_by_o2_sr]

        # Get a pandas series of booleans for filtering:
        lo_score_by_o2_sr = (answers_df.Score < lo_score_limit)
        # Get a pandas df with rows for all low-score answers of one user:
        lo_score_answers_by_o2_df = answers_df[[
            'Id', 'OwnerUserId', 'Score']][lo_score_by_o2_sr]
        owners_df_l.append(lo_score_answers_by_o2_df)

    # TBD These are the answers to examine for useful data, even though
    # they have low scores.
    # View these answers and evaluate them manually; and analyze them
    # with other s/w.
    lo_scores_for_top_owners_df = pd.concat(owners_df_l)
    print('lo_score_limit: ', lo_score_limit)
    print('Length of lo_scores_for_top_owners_df: ',
          len(lo_scores_for_top_owners_df))
    outfile = DATADIR + 'lo_scores_for_top_owners.csv'
    lo_scores_for_top_owners_df.to_csv(
        outfile, header=True, index=None, sep=',', mode='w')
    cf.logger.info('group_data(): lo_scores_for_top_owners_df: ')
    cf.logger.info(lo_scores_for_top_owners_df)
    if cli_args_d['verbose']:
        print('lo_scores_for_top_owners_df: ')
        print(lo_scores_for_top_owners_df)
    print()

    return top_scoring_owners_a, owner_grouped_df


def find_ques_ids_from_top_owners(top_scoring_owners_a, aa_df):
    """Make a list of all answer records by the high-score owners.
    Use that list to build a list of ParentId's (ie, questions)
    to collect for evaluation.
    Return that list of ParentId's.
    """
    owners_df_l = []
    for owner in top_scoring_owners_a:
        # Get a pandas series of booleans for filtering:
        answered_by_owner_sr = (aa_df.OwnerUserId == owner)
        # Get a pandas df with rows for all answers of one user:
        answers_df = aa_df[['Id', 'OwnerUserId',
                            'ParentId', 'Score']][answered_by_owner_sr]
        owners_df_l.append(answers_df)

    hi_scoring_owners_df = pd.concat(owners_df_l)
    # D print('Length of hi_scoring_owners_df: ', len(hi_scoring_owners_df))
    # D print('hi_scoring_owners_df: ')
    # D print(hi_scoring_owners_df)

    # Get list of unique ParentId's:
    ques_ids_from_top_own_l = list(set(hi_scoring_owners_df['ParentId']))
    # D print('ques_ids_from_top_own_l: ')
    # D print(ques_ids_from_top_own_l)
    # D print()
    return ques_ids_from_top_own_l


def find_pop_and_top_ques_ids(ques_ids_from_top_own_l, popular_ids_a):
    """Make a list of question Id's to use for further processing.

    Select popular questions (those with several answers),
    that also have answers from top-scoring owners (owners with
    a high mean score).

    Set the size of the following list slices to get enough o/p to analyze.
    For i/p data files q6 and a6:
    With slice limits at 40 and 30, got 2 Q, 36 A.
    With slice limits at 400 and 300, got 26 Q, 308 A.

    Return the list of question id's in ques_ids_pop_and_top_l.
    """
    ques_ids_pop_and_top_l = list(
        # For q6 & a6 data: set(ques_ids_from_top_own_l[:500]).intersection(set(popular_ids_a[:500])))
        # For q3 & a3 data: set(ques_ids_from_top_own_l[:40]).intersection(set(popular_ids_a[:10])))
        #D set(ques_ids_from_top_own_l[:40]).intersection(set(popular_ids_a[:10])))
        set(ques_ids_from_top_own_l[:900]).intersection(set(popular_ids_a[:900])))
    log_msg = 'find_pop_and_top_ques_ids(): len(ques_ids_pop_and_top_l) : ' + str(len(ques_ids_pop_and_top_l))
    cf.logger.info(log_msg)
    log_msg = "find_pop_and_top_ques_ids(): ques_ids_pop_and_top_l, top-N parent id\'s to examine: " + str(ques_ids_pop_and_top_l[0:10])
    cf.logger.info(log_msg)
    if cli_args_d['verbose']:
        print('ques_ids_pop_and_top_l, parent id\'s to examine: ', ques_ids_pop_and_top_l[:])
    return ques_ids_pop_and_top_l


#TBD,Thu2017_1221_19:12  Rewrite combine*() to be more general w/ diff arg names.
#TBD,Thu2017_1221_19:12  Rewrite combine*() to be more general w/ diff arg names.
#TBD,Thu2017_1221_19:12  Rewrite combine*() to be more general w/ diff arg names.
def combine_related_q_and_a(ques_ids_pop_and_top_l, all_ques_df, aa_df, num_selected_recs, a_fname, progress_msg_factor):
    """Get each Q in the list of selected ParentId's, and the related A's.
    Loop over each question and store the Q & A data in a dataframe.

    TBD-move:
    Then call analyze_text() on each group of Q with A's,
    which calls the routines that perform the natural language processing
    of the text data.
    """
    global popular_qa_df

    ques_match_df = all_ques_df[all_ques_df['Id'].isin(ques_ids_pop_and_top_l)]
    ans_match_df = aa_df[aa_df['ParentId'].isin(ques_ids_pop_and_top_l)]

    print('#D len of ques_match_df: ', len(ques_match_df))
    print('#D len of ans_match_df: ', len(ans_match_df))
    print('\n#D ques_match_df.head():')
    print(ques_match_df.head())
    print()
    print('\n#D ans_match_df.head():')
    print(ans_match_df.head())


    # Build each Q&A group: one Q w/ all its A.'s
    # TBD, How to do this w/o explicit loop, using df tools?
    # TBD, Combine or replace the ques_match_df & ans*df code above w/ this?
    # Maybe delete those 'intermediate' results?
    qagroup_from_pop_top_ques_df = pd.DataFrame()
    for i, qid in enumerate(ques_ids_pop_and_top_l):
        # OK if(i % progress_msg_factor == 0):
        if(i % 20 == 0):
            print("#D combine_related_q_and_a():progress count: ", i)
        qm_df = ques_match_df[ques_match_df['Id'] == qid]
        am_df = ans_match_df[ans_match_df['ParentId'] == qid]
        qagroup_from_pop_top_ques_df = pd.concat([qm_df, am_df]).reset_index(drop=True)
        #
        # Change NaN values to zero, and the field type from float64 to int64,
        # to help with search functionality.
        #TBD,Tue2017_1226_17:01 , Must this be done elsewhere?
        qagroup_from_pop_top_ques_df['ParentId'] = qagroup_from_pop_top_ques_df['ParentId'].fillna(0).astype(int)
        qagroup_from_pop_top_ques_df['OwnerUserId'] = qagroup_from_pop_top_ques_df['OwnerUserId'].fillna(0).astype(int)
        #D print("#D combine*() after changing series: qagroup_from_pop_top_ques_df['ParentId']:\n", qagroup_from_pop_top_ques_df['ParentId'].head(9) , '\n')
        #
        #TBD,Fri2017_1222_00:02 , This maybe fixed big problem, don't analyze empty df:
        if qagroup_from_pop_top_ques_df.empty:
            #TBD, Skip this qid, it does not match what we seek?
            continue
        # D print("\n#D qagroup_from_pop_top_ques_df.head(): ")
        # D print(qagroup_from_pop_top_ques_df.head())
        cf.logger.info('qagroup_from_pop_top_ques_df.head(1): ')
        cf.logger.info(qagroup_from_pop_top_ques_df.head(5))  #TBD, reset to 1

        # Analyze data w/ nlp s/w.
        popular_qa_df = analyze_text(
             qagroup_from_pop_top_ques_df, num_selected_recs, a_fname, progress_msg_factor)

    # END combine_related_q_and_a().
    return popular_qa_df




def compute_record_selector(numlines):
    """Compute the number of records to use for computation and display.

    Return that integer.
    """

    SELECTOR_RATIO = 0.10  # TBD, use Default 0.01 in production? Move to global scope?
    num_selected_recs = max(5, int(numlines * SELECTOR_RATIO))
    log_msg = ("  SELECTOR_RATIO,  number of selected recs: " +
                str(SELECTOR_RATIO) + ", " + str(num_selected_recs))
    cf.logger.info(log_msg)

    return num_selected_recs


def analyze_text(qagroup_from_pop_top_ques_df, num_selected_recs, a_fname, progress_msg_factor):
    """Use a Q&A group of one Q w/ its A's for i/p.
    Process the text data w/ the routines in the nltk module, which use
    natural language tools.

    Important variables.

    qagroup_from_pop_top_ques_df: 
    A dataframe with one Q&A group (ie, one question with its related
    answers), which is selected from all such groups based on being 'pop'
    (popular, questions with several answers) and 'top' (having one
    or more answers by high-reputation owners).

    """
    global popular_qa_df
    global search_qa_df 

    cf.logger.info("NLP Step 2. Process the words of each input line.")
    #D print('#D analyze_text(): qagroup_from_pop_top_ques_df: ', qagroup_from_pop_top_ques_df)
    clean_ans_bodies_l = nl.clean_raw_data(qagroup_from_pop_top_ques_df)
    if not clean_ans_bodies_l:
        print('#D analyze_text(): clean_ans_bodies_l: ', clean_ans_bodies_l)
        print('#TBD, analyze_text(), missing data in clean_ans_bodies_l, no text to analyze; exit now.')
        raise SystemExit()
    #D print('\n#D, clean_ans_bodies_l[:1]')
    #D print(clean_ans_bodies_l[:1])

    cf.logger.info("NLP Step 3. Build a bag of words and their counts.")
    (vocab_l, dist_a) = nl.make_bag_of_words(clean_ans_bodies_l)
    #D print('\n#D, vocab_l[:1]')
    #D print(vocab_l[:1])
    words_sorted_by_count_l = nl.sort_vocab(vocab_l, dist_a)

    cf.logger.info("NLP Step 7. Search lo-score A's for hi-score text.")
    qa_with_hst_df = nl.find_hi_score_terms_in_bodies(
        words_sorted_by_count_l,
        clean_ans_bodies_l,
        num_hi_score_terms, qagroup_from_pop_top_ques_df)
    popular_qa_df = pd.concat(
        [popular_qa_df, qa_with_hst_df]).reset_index(drop=True)
    #TBD,Sat2017_1223_16:09 , Prob: search*df includes every record from hst df b/c of concat.
        # Same problem w/ popular_qa_df above.
        #TBD,Mon2017_1225_15:12 , So don't concat search df w/ hst df.
    #D search_qa_df = pd.concat(
        #D [search_qa_df, qa_with_hst_df]).reset_index(drop=True)
    #D print('#D analyze*(): len(qa_with_hst_df: ', len(qa_with_hst_df))
    #D print('#D analyze*(): len(search_qa_df: ', len(search_qa_df))
    cf.logger.debug('popular_qa_df: len')
    cf.logger.debug(len(popular_qa_df))

    return popular_qa_df


# TBD.1 , should select_keyword_recs()
# be called before combine_related*()?
# Use it to make the final ques_ids_pop_and_top_l?

def select_keyword_recs(keyword, qa_df, columns_l,
        all_ques_df, all_ans_df, num_selected_recs, a_fname, progress_msg_factor):
    """Find the Q's & A's from the filtered dataframe that contain the keyword,
    in Title or Body.
    Build a list of unique Id's in the desired order.
    Build a df of Q's w/ their A's.
    Save all the selected data for analysis.
    #
    Gather all A's with keyword and their related Q's;
    then get the Q and all related A's into one Q&A group
    in the right order for display and analysis.
    #
    TBD, Show the Q's that have the keyword.
    Show the A's from the filtered df that have the keyword;
    sort them in order of decreasing HSTCount.
    Show the Q's that have the keyword.
    #
    TBD, Show the Q&A items from the filtered df that have the keyword;
    sort them in order of decreasing HSTCount.
    #
    TBD, In future, search entire collection of Q&A, not just
    the filtered subset.
    #
    TBD, qa_df: filtered dataframe of questions and answers that are pop and top,
    ie, popular and from owners with high reputation scores.
    """
    #
    # Get a pandas series of booleans to find the current question id.
    # Check Question & Answer, both Title and Body columns.
    qt_sr = qa_df.Title.str.contains(keyword, regex=False)
    qab_sr = qa_df.Body.str.contains(keyword, regex=False)
    #
    # Combine two series into one w/ boolean OR.
    qa_contains_sr = qab_sr | qt_sr
    #
    # Build a df with Q's and A's that contain the keyword.
    qak_df = qa_df[columns_l][qa_contains_sr]
    if qak_df.empty:
        print('Search term not found, the dataframe is empty.')
        return
    #
    # Build a list of Id values of Q's and A's that contain the keyword.
    #
    parent_id = -1  # Initlz if needed outside the loop, and if search has no matches.
    ans_ids_from_search_l = []  # A's w/ keywords.
    ques_ids_from_search_l = [] # Q's w/ keywords and Q's of A's w/ keywords
    #
    for index, row in qak_df.iterrows():
        #TBD,Thu2018_0104_15:46  rm unneeded vars from here:
        rec_id = row['Id']
        title = row['Title']
        body = row['Body']
        parent_id = row['ParentId']
        hstcount = row['HSTCount']
        score = row['Score']
        #
        if pd.isnull(title):
            # Found an answer w/ keyword, add its Q.Id to ques list.
            ques_ids_from_search_l.append(parent_id)
            # Also add the A.Id to the answer list.
            ans_ids_from_search_l.append(rec_id)
        elif title:
            # Found a question w/ keyword; add it to ques list.
            ques_ids_from_search_l.append(rec_id)
        else:
            #TBD,Tue2017_1226_18:21 , Test this path, it should never be reached.
                # Need a bad data record for the test.
                # Maybe need record w/ no Title field?
            print("\nselect*(): Bad data found, problem with Title?  Id, Title:")
            print("Id: ", rec_id)
            print("ParentId: ", parent_id)
            print("Title: ", title)
            raise SystemExit()  # TBD too drastic?  Maybe show menu?
    #
    # Build sets of unique IDs.
    ques_ids_from_search_l = list(set(ques_ids_from_search_l))
    ans_ids_from_search_l = list(set(ans_ids_from_search_l))
    #
    # Print summary, 1 line/record:
    print()
    print('#D 1-line summary, qak_df, Q & A w/ keyword:\n', qak_df[:] , '\n')
    #
    #
    # Build a list of answer Id's.
    ans_ids_sorted_l = []
    ans_ids_l = []
    for aid in ans_ids_from_search_l:
        a_df = qak_df.loc[qak_df['Id'] == aid]
            #TBD, Using qa_df here is not obviously diff from qak.
        ans_ids_l.append([a_df['HSTCount'].values[0], a_df['Score'].values[0], a_df['Id'].values[0]])
    #
    # Build a list of answer Id's in sorted order.
    ans_ids_sorted_l = sorted(ans_ids_l, reverse=True)
    print('#D.1547 sorted ans_ids_sorted_l:\n', ans_ids_sorted_l)
    print()
    #
    #
    # Use sorted A's to build ordered Q's.
    #
    ques_ids_ord_l = []
    print('#D Q.Id, Q.Title:')
    for hstc, score, aid in ans_ids_sorted_l:
        q_df = qak_df.loc[qak_df['Id'] == aid]
        parent_id = q_df['ParentId'].iloc[0]
        print(parent_id)
        tmp_df = qa_df.loc[qa_df['Id'] == parent_id]
            #TBD, using qa_df is better than qak: it finds more Q.Titles.
        print(tmp_df['Title'].iloc[0])
        ques_ids_ord_l.append(parent_id)
    #
    # Append the chosen Q.Id's found by search to make the full Q list in desired order.
    #TBD, use list append or concat?
    ques_ids_ord_l += ques_ids_from_search_l
    #
    # Remove duplicate Q.Id's w/o changing order of the list.
    qid_ord_l = []
    for i in ques_ids_ord_l:
        if i not in qid_ord_l:
            qid_ord_l.append(i)
    print('\n\n#D.1702 qid_ord_l: ', qid_ord_l )
    #
    #
    # Build list of Q.Id and A.Id in the right order for output.
    qa_id_ord_l = []
    for qid in qid_ord_l:
        # Save the Q.Id.
        qa_id_ord_l.append(qid)
        #
        # Find & sort & save all A.Id's for this Q.
        #
        ans_match_df = qa_df[qa_df['ParentId'] == qid]
            #TBD,Wed2018_0103_18:51  qa_df is popular_qa_df; maybe use a df w/ more records in the call to this func?
        # Sort the df:
        #TBD, sort it inplace to avoid another temp var? Does inplace have problems?
        ams_df = ans_match_df.sort_values(['HSTCount', 'Score', 'Id'], ascending=[False, False, True])
        #
        for hstc, score, aid in ams_df[['HSTCount', 'Score', 'Id']].values:
            qa_id_ord_l.append(aid)
    print('\n\n#D.1550 qa_id_ord_l: ', qa_id_ord_l )
    #
    # Build o/p df w/ Q's and A's in right order, from the list of Id's.
    qa_keyword_df_l = []
    for item in qa_id_ord_l:
        for index, row in qa_df.iterrows():
            if row['Id'] == item:
                qa_keyword_df_l.append(row)
                continue
    qa_keyword_df = pd.DataFrame(qa_keyword_df_l)
    #

    # Write o/p to disk file.
    search_fname = 'search_result_full.csv'
    save_prior_file(DATADIR, search_fname)
    qa_keyword_df.to_csv(DATADIR + search_fname)

    search_fname = 'search_result_full.html'
    save_prior_file(DATADIR, search_fname)
    columns_l = ['HSTCount', 'Score', 'Id', 'Title', 'Body']
    write_full_df_to_html_file(qa_keyword_df, DATADIR, search_fname, columns_l)
    #
    return  qa_keyword_df





def show_menu(qa_df, all_ans_df, owner_reputation_df,
        all_ques_df, num_selected_recs, a_fname, progress_msg_factor):
    """Show prompt to user; get and handle their request.
    """
    user_menu = """    The menu choices:
    drm: calculate reputation matrix of owners
    dsm: draw and plot q&a statistics matrix
    d: draw default plot of current data
    dh: draw default histogram plot of current data
    dm: draw scatter matrix plot of current data
    h, ?: show help text, the menu
    lek: look for exact keyword in questions and answers
    m: show menu
    q: save data and quit the program
    s: show current item: question or answer
    sn: show next item: question or answer
    sp: show prior item: question or answer
    """
    user_cmd = ''
    saved_index = 0

    # Show prompt & wait for a cmd.
    print("======================\n")
    cmd_prompt = "Enter a command: q[uit] [m]enu  ...  [h]elp: "
    while user_cmd == "":  # Repeat the prompt if the Enter key is pressed.
        user_cmd = input(cmd_prompt)
    print("User entered this cmd: ", user_cmd)

    # Loop to handle user request.
    while user_cmd:
        # D print("#D while-loop: User entered this cmd: ", user_cmd)
        if user_cmd.lower() == 'm':
            print(user_menu)
            user_cmd = ''
        elif user_cmd.lower() == 'q':
            # print("Save data and Quit the program.")
            print("Quit the program.")
            log_msg = cf.log_file + ' - Quit by user request; Finish logging for ' + \
                os.path.basename(__file__) + '\n'
            cf.logger.warning(log_msg)
            raise SystemExit()
        elif user_cmd == '?' or user_cmd == 'h':
            print(user_menu)
            # Clear user_cmd here to avoid infinite repetition of while loop.
            user_cmd = ''
        elif user_cmd.lower() == 's':
            user_cmd = ''
            if qa_df.empty:
                print("WARN: dataframe empty or not found; try restarting.")
            else:
                print(qa_df[['Id', 'Title', 'Body']].iloc[[saved_index]])
        elif user_cmd.lower() == 'sn':  # show next item
            user_cmd = ''
            if qa_df.empty:
                print("WARN: dataframe empty or not found; try restarting.")
            else:
                saved_index += 1
                print(qa_df[['Id', 'Title', 'Body']].iloc[[saved_index]])
        elif user_cmd.lower() == 'sp':  # show prior item
            user_cmd = ''
            if qa_df.empty:
                print("WARN: dataframe empty or not found; try restarting.")
            else:
                saved_index -= 1
                print(qa_df[['Id', 'Title', 'Body']].iloc[[saved_index]])
        elif user_cmd.lower() == 'd':
            user_cmd = ''
            if popular_qa_df.empty:
                print("WARN: dataframe empty or not found; try restarting.")
            else:
                print("Drawing the default plot, with Score and HSTCount.")
                draw_scatter_plot(
                    popular_qa_df,
                    'Score',
                    'HSTCount',
                    'Score',
                    'HSTCount')
        elif user_cmd.lower() == 'dh':
            user_cmd = ''
            if popular_qa_df.empty:
                print("WARN: dataframe empty or not found; try restarting.")
            else:
                print("Drawing the default histogram plot.")
                draw_histogram_plot(popular_qa_df)
        elif user_cmd.lower() == 'dm':  # Scatter matrix plot
            user_cmd = ''
            if popular_qa_df.empty:
                print("WARN: dataframe empty or not found; try restarting.")
            else:
                print("Drawing the default scatter matrix plot.")
                draw_scatter_matrix_plot(popular_qa_df)
        # drm: Draw Reputation matrix, mean, answers only; scatter.
        elif user_cmd.lower() == 'drm':
            user_cmd = ''
            owner_reputation_df = check_owner_reputation(all_ans_df, owner_reputation_df)
            #
            if owner_reputation_df.empty:
                print("WARN: owner reputation dataframe empty or not found.")
            else:
                print("NOTE: Drawing the owner reputation scatter matrix plot.")
                draw_scatter_matrix_plot(owner_reputation_df[['MeanScore', 'OwnerUserId']])
        # dsm: Draw q&a statistics matrix
        elif user_cmd.lower() == 'dsm':
            user_cmd = ''
            owner_reputation_df = check_owner_reputation(all_ans_df, owner_reputation_df)
            #
            qa_stats_df = build_stats(popular_qa_df, owner_reputation_df)
            #
            if qa_stats_df.empty:
                print("WARN: qa_stats_df empty or not found.")
            else:
                print("NOTE: Drawing the qa_stats_df scatter matrix plot.")
                draw_scatter_matrix_plot(qa_stats_df[['Score', 'BodyLength', 'OwnerRep',  'HSTCount' ]])
        #TBD, Search cmd lek is now case sensitive.
        # lek: Look for exact keywords in the Q&A df.
        elif user_cmd.lower() == 'lek':
            user_cmd = 'lek'  # Force menu to always repeat the lek function.
            search_prompt = "\n\nPlease type a search term; or press Enter alone to return to main menu: "
            search_term = input(search_prompt)
            if search_term == "": # Return to main menu and ask for a cmd.
                user_cmd = 'm'
                continue
            print("User entered this search_term: ", search_term)
            log_msg = cf.log_file + ' - Search term entered by user: ' + \
                search_term + '\n'
            cf.logger.warning(log_msg)
            #
            columns_l = ['Id', 'ParentId', 'Title', 'Body', 'HSTCount', 'HiScoreTerms', 'Score' ]
            q_a_group_with_keyword_df = pd.DataFrame() # Initlz at each call
            #
            #TBD,Wed2018_0103_15:27  Chg popular_qa_df to a df w/ more records, for dbg & initial use.
            #TBD,Wed2018_0103_15:27  Chg popular_qa_df to a df w/ more records, for dbg & initial use.
            #TBD,Wed2018_0103_15:27  Chg popular_qa_df to a df w/ more records, for dbg & initial use.
                # Don't limit the df in use until necessary.
                # Also, only include needed vars in the func call.
            q_a_group_with_keyword_df = select_keyword_recs(
                search_term, popular_qa_df, columns_l,
                all_ques_df, all_ans_df, num_selected_recs, a_fname, progress_msg_factor)
            #
            """TBD,Sat2017_1223_15:50 , Big change for debug.
            if q_a_group_with_keyword_df.empty:
                print("WARN: dataframe empty or search term not found; try another term.")
            else:
                # Show full text of the record.
                show_q_a_with_keywords(q_a_group_with_keyword_df, search_term)
            """
            """
            TBD,Thu2018_0104_15:36  This code is not now used; keep temporarily.
                The select*() function now does the reqd o/p actions.
            if search_qa_df.empty:
                print("WARN: dataframe empty or search term not found; try another term.")
            else:
                # Show full text of the record.
                show_q_a_with_keywords(search_qa_df, search_term)
            """
        else:
            print("Got bad cmd from user: ", user_cmd)
            print(user_menu)
            # Clear user_cmd here to avoid infinite repetition of while loop.
            user_cmd = ''
        # Show prompt & wait for a cmd.
        print("======================\n")
        while user_cmd == "":  # Repeat the prompt if the Enter key is pressed.
            user_cmd = input(cmd_prompt)
    print("#D End of the cmd interpretation loop; return.")
    print()
    # D print("#D Last stmt of show_menu(); return.\n")

    # Save df before exit, if quit cmd is not used.
    # TBD:
    # Same code used for 'q' cmd; refactor both.
    # print("Save data and Quit the program.")
    # Save only the needed fields to the file.
    # TBD:
    # columns_l = ['Id', 'ParentId', 'Grade', 'Notes', 'Title', 'Body']
    # outfile = open(DATADIR + 'graded_popular_qa.csv', 'w')
    # graded_df[columns_l].to_csv(outfile, header=True, index=None, sep=',', mode='w')
    # outfile.flush()

    return


#ORG,Sat2017_1223_15:53  def show_q_a_with_keywords(q_a_group_with_keyword_df, search_term):
def show_q_a_with_keywords(search_qa_df, search_term):
    """
    #TBD,Sat2017_1216_15:43  Ideas
    Get i/p CLI arg for number of answers to show; default is 3.
    Set answers_to_show_count = 0
    Review each row in the i/p df.
        If a Q:
            examine all A's
            print Q
            print A.'s with highest HSTCount, w/ highest Scores, etc.
            increment num_answers_shown_count
            if num_answers_shown_count >= num_answers_to_show:
                break
        If an A:
            find ParentId, the related Q
            examine all A's for that Q
            print Q
            print A.'s with highest HSTCount, w/ highest Scores, etc.
            increment num_answers_shown_count
            if num_answers_shown_count >= num_answers_to_show:
                break
    Opt: Print summary o/p, 1 line per row of i/p df.
        Id, Title, HSTCount, Score.
    """
    pd.set_option('display.max_colwidth', -1)  # -1=no limit, for debug
    search_summary_l = []
    search_full_l = []
    """
    #TBD,Thu2017_1221, temp code, not run.
    for index, row in q_a_group_with_keyword_df.iterrows():
        ident = row['Id']
        parent_id = row['ParentId']
        if not parent_id: # Found a question; get all its answers.
            #TBD,Thu2017_1221_15:59 , copied from main(); is it needed here?
            popular_qa_df = \
                combine_related_q_and_a(
                    ques_ids_pop_and_top_l, all_ques_df, all_ans_df, num_selected_recs, a_fname, progress_msg_factor)
    """
    for index, row in search_qa_df.iterrows():
        ident = row['Id']
        parent_id = row['ParentId']
        score = row['Score']
        hstcount = row['HSTCount']
        hst = row['HiScoreTerms']
        title = row['Title']
        body = row['Body']
        row_for_summary_t = (index, ident, parent_id, score, hstcount, title) # Short 1-line/record
        row_with_body_t = (index, ident, parent_id, score, hstcount, title, body) # Record w/ Body field.
        #
        # Append fields of current record to o/p list.
        search_summary_l.append(row_for_summary_t) # For short 1-line/record summary
        search_full_l.append(row_with_body_t) # For most fields of matching output records
        #
        # Show some fields from the current record.
        #D,TBD,Fri2017_1222_17:09  print_row_as_key_value(index, row)
        #
        #TBD,Thu2017_1221_14:15  If record is a Q, find all its A's and include them in the o/p.

    # Print summary data for records w/ the search term
    print()
    print("Summary of Search for Term: {}\n===\nIndex, Id, ParentId, Score, HSTCount, Title:".format(search_term)) 
    #D print('\n'.join(str(r) for r in search_summary_l))
    print('#D show_q_a*(): len(search_qa_df: ', len(search_qa_df))
    print('#D show_q_a*(): len(search_summary_l: ', len(search_summary_l))
    print()

    # Convert list to df, then write df to disk.
    #TBR? search_result_full_df = pd.DataFrame() # Initlz at every call.
    search_result_full_df = pd.DataFrame(search_full_l, columns = ['Index', 'Id', 'ParentId', 'Score', 'HSTCount', 'Title', 'Body'])
    wdir = DATADIR
    search_fname = 'search_result_full.csv'
    save_prior_file(wdir, search_fname)
    search_result_full_df.to_csv(wdir + search_fname)
    pd.set_option('display.max_colwidth', MAX_COL_WID)  # -1=no limit, for debug

    search_fname = 'search_result_full.html'
    save_prior_file(wdir, search_fname)
    columns_l = ['Id', 'Title', 'Body']
    write_full_df_to_html_file(search_result_full_df, TMPDIR, search_fname, columns_l)

    return 1



def print_row_as_key_value(index, row):
    ident = row['Id']
    parent_id = row['ParentId']
    score = row['Score']
    hstcount = row['HSTCount']
    hst = row['HiScoreTerms']
    title = row['Title']
    body = row['Body']
    print("#D index: ", index)
    print("#D HSTCount: ", hstcount)
    print("#D Score: ", score)
    print("#D Id: ", ident)
    print("#D ParentId: ", parent_id)
    print("#D Title: ", title)
    print("#D HiScoreTerms: ", hst)
    #TBD print("#D Body: ", body)
    print("#D ==========\n")

    return 1



def build_stats(qa_df, or_df):
    """Build a table of statistical data about the data, for
    analysis and plotting.

    Plan:
    Loop on all records in qa_df:
        Read record from qa_df.
        Find OUId.
        Read Owner Reputation df.
            Find that OUId.
            Write Owner Rep to qa_stats_df in its column.
    Print table.
    Plot data in scatter matrix.
    Visually look for records with high Reputation and low Score.
    """
    qa_stats_df = qa_df[['Id', 'OwnerUserId', 'ParentId', 'Score', 'HSTCount']]
    # Add new column to df & initlz it.
    qa_stats_df = qa_stats_df.assign(BodyLength=qa_stats_df.Id)

    for index, row in qa_df.iterrows():
        ouid = row['OwnerUserId']
        #D print("#D qa_stats_df index, ouid: ", index, ouid)
        try:
            owner_rep = round(or_df.loc[or_df['OwnerUserId'] == ouid, 'MeanScore'].iloc[0])
            #D print("#D qa_stats_df index, owner_rep: ", index, owner_rep)
            # Add new column to df.
            qa_stats_df.loc[index, 'OwnerRep'] = owner_rep
        except IndexError:
            # TBD, Some answers in the data file were made by Owners
            # who are not yet in the reputation df.
            # This should only be an issue when using small data sets.
            print("NOTE: build_stats(): did not find ouid in owner reputation dataframe; index,ouid: ", index, ouid)
            print("#D NOTE: build_stats():data from the problem row:\n", row)
            print()

        # Save length of body text of each answer.
        qa_stats_df.loc[index, 'BodyLength'] = len(row['Body'])

    #D print('#D qa_stats_df.head(5):')
    #D print(qa_stats_df.head(5))
    qa_stats_df = qa_stats_df[['Id', 'ParentId', 'OwnerUserId', 'Score', 'BodyLength', 'OwnerRep', 'HSTCount']]

    stats_fname = DATADIR + 'qa_stats_by_dsm.csv'
    save_prior_file('', stats_fname)
    qa_stats_df.to_csv(stats_fname)

    stats_fname = DATADIR + 'qa_stats_by_dsm.html'
    save_prior_file('', stats_fname)
    qa_stats_df.to_html(stats_fname)

    return qa_stats_df


def check_owner_reputation(all_ans_df, owner_reputation_df):
    """Check for dataframe with reputation of each OwnerUserId.
    If not found, then calculate reputation of each OwnerUserId in the i/p data,
    based on Score of all answers they provided.
    Save the data to a disk file and use it when needed, so the
    calculation need not be done every time this program runs.
    """
    own_rep_file = DATADIR + 'owner_reputation.csv'
    #TBF.Fri2017_0804_14:54 , Must chk i/p file & replace owner_rep*.csv if
    #  a different file was used.  OR, just build this file from Answers.csv
    #  which should have all answers & produce good reputation data.

    if not owner_reputation_df.empty:
        return owner_reputation_df

    if os.path.exists(own_rep_file):
        print("NOTE: owner rep file, " + own_rep_file + ", found; read it.")
        owner_reputation_df = pd.read_csv(
            own_rep_file,
            encoding='latin-1',
            warn_bad_lines=False,
            error_bad_lines=False)
        return owner_reputation_df
    else:
        print("NOTE: owner rep file, " + own_rep_file + ", not found; build it.")
        print("NOTE: This should be a one-time operation w/ data saved on disk.")
        owner_reputation_df = gd2_group_data(all_ans_df)
        owner_reputation_df.to_csv(own_rep_file)
    return owner_reputation_df


def draw_histogram_plot(plot_df):
    """Draw a simple histogram plot using pandas tools.
    """
    fig, ax = plt.subplots(1, 1)
    ax.get_xaxis().set_visible(True)
    plot_df = plot_df[['Score']]
    # D These custom sized bins are used for debugging.
    # histo_bins = [-10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                  # 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45, 50]
    histo_bins = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    plot_df.plot.hist(ax=ax, figsize=(6, 6), bins=histo_bins)
    plt.show(block=False)

    # Write data set to a csv file.
    outfile = TMPDIR + 'dh_draw_histogram.csv'
    plot_df['Score'].to_csv(
        outfile,
        header=True,
        index=None,
        sep=',',
        mode='w')

    # Print data used for histogram
    count, division = np.histogram(plot_df['Score'], bins=histo_bins)
    print("#D histogram data: count[:5]: ", count[:5])
    print("#D histogram data: division[:5]: ", division[:5])
    return


def draw_scatter_matrix_plot(plot_df):
    """Draw a set of scatter plots showing each feature vs
    every other feature.
    """
    cf.logger.info('Summary stats from plot_df.describe(): ')
    cf.logger.info(plot_df.describe())

    print('Summary stats from plot_df.describe(): ')
    print(plot_df.describe())
    print('NOTE: Please wait for plot, 60 sec or more.')

    axs = scatter_matrix(plot_df, alpha=0.2, diagonal='hist')
    # TBD plt.xscale('log')  # Failed. Logarithm scale, Good to show outliers. Cannot show Score=0?

    plt.show(block=False)

    wdir = DATADIR
    wfile = 'scat_mat_plot.pdf'
    save_prior_file(wdir, wfile)
    plt.savefig(wdir + wfile)

    wfile = 'scat_mat_plot.png'
    save_prior_file(wdir, wfile)
    plt.savefig(wdir + wfile)
    return


def draw_scatter_plot(plot_df, xaxis, yaxis, xname, yname):
    """Draw a simple scatter plot using pandas tools.
    """
    ax = plot_df[[yname, xname]].plot.scatter(x=xaxis, y=yaxis, table=False)
    plt.show(block=False)
    return


#TBD.1, Simplify this entire func.
def save_prior_file(wdir, wfile):
    """Save backup copy of a file w/ same name; add '.bak' extension.

    Input wdir should have trailing slash, eg, data/.
    Input wfile should have trailing extension name, eg, .csv.
    """
    import shutil

    outfile = wdir + wfile
    if not os.path.isfile(outfile):
        # Skip the save if the file does not exist.
        return

    # Save all backups to tmp/ dir.
    dst_filename = os.path.join(TMP, os.path.basename(outfile))
    shutil.move(outfile, dst_filename)
    print(
        '\nWARN: Moved old file to tmp storage; save it manually if needed: ' +
        TMP + wfile)
    return


def write_full_df_to_csv_file(in_df, wdir, wfile):
    """Write full contents of all columns of a data frame to a csv file.
    """
    # Used for testing and debugging.
    pd.set_option('display.max_colwidth', -1)  # -1=no limit, for debug
    save_prior_file(wdir, wfile)
    outfile = wdir + wfile
    in_df.to_csv(outfile)
    pd.set_option('display.max_colwidth', MAX_COL_WID)  # -1=no limit, for debug
    return


def replace_line_breaks(in_s):
    """Replace escaped line break chars so text inside HTML
    pre-blocks and code-blocks (inside HTML table cells)
    is rendered properly on screen.
    """
    out_s = in_s.replace('\\r\\n', '\n')
    out_s = out_s.replace('\\n\\n', '\n')
    out_s = out_s.replace('\\n', '\n')
    return out_s


def write_full_df_to_html_file(in_df, wdir, wfile, columns_l):
    """Write full contents of some columns of a data frame to an html file.

    Use the list of columns included in this function
    if caller does not specify any.
    """
    if in_df.empty:
        print('Input dataframe empty or not found.')
        return
    pd.set_option('display.max_colwidth', -1)  # -1=no limit, for debug
    outfile = wdir + wfile
    save_prior_file(wdir, wfile)
    # Specify default output fields to use.
    if not columns_l:
        columns_l = ['Id',
                     'Title',
                     'Body',
                     'Score',
                     'HSTCount',
                     'HiScoreTerms',
                     'OwnerUserId',
                     'ParentId']
    #
    # Save o/p to a string and do not specify an output file in
    # calling to_html().
    # Use 'escape=False' to render HTML when outfile is opened in a browser.
    # Use 'index=False' to prevent showing index in column 1.
    in_s = in_df[columns_l].to_html(escape=False, index=False)

    # Clean the newlines in the string
    # so the HTML inside each table cell renders properly on screen.
    in_s = replace_line_breaks(in_s)

    # Concatenate css w/ html file to format the o/p better.
    with open(outfile, 'w') as f:
        print('\nNOTE: Writing data to html outfile: ' + outfile)
        f.write(HEADER)
        f.write(in_s)
        f.write(FOOTER)

    pd.set_option('display.max_colwidth', MAX_COL_WID)  # -1=no limit, for debug
    return


def get_parser():
    """Create parser to specify cmd line options for this program.
    """
    parser = argparse.ArgumentParser(
        description='find good answers hidden in stackoverflow data')

    parser.add_argument(
        '-d',
        '--debug',
        help='Use settings to help with debugging',
        action='store_true')

    parser.add_argument(
        '-L',
        '--lo_score_limit',
        help='lowest score for an answer to be included',
        default=10,
        type=int)

    parser.add_argument(
        '-q',
        '--quit',
        help='Stop the program before showing the menu; used for testing',
        action='store_true')

    """
    parser.add_argument(
        '-p',
        '--popular_questions',
        help='select questions with many answers',
        action='store_true')
    parser.add_argument(
        '-t',
        '--top_owners',
        help='find lo-score answers by hi-scoring owners',
        action='store_true')
    """
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser


if __name__ == '__main__':

    check_install()

    parser = get_parser()
    cli_args_d = vars(parser.parse_args())

    # Set the number of top scoring owners to select from the data.
    num_owners = 10  # Default is 10.
    num_owners = 40  # Default is 10.
    # D num_owners = 100  # Default is 10.
    print("num_owners: ", num_owners)

    keyword = False
    # D keyword = 'font'  # Found in a3* & q3* i/p files.
    # D keyword = 'beginner'
    # D keyword = 'yield'
    # D keyword = 'begin'
    # D keyword = 'pandas'
    # D keyword = 'Python'  # Both Title & Body of data sets have it; for debug
    # D keyword = 'library'
    print("Keyword: ", keyword)

    num_hi_score_terms = 21  # Use 3 for testing; 11 or more for use.
    num_hi_score_terms = 3  # Use 3 for testing; 11 or more for use.
    print("num_hi_score_terms: ", num_hi_score_terms)

    main(popular_qa_df)

    owner_reputation_df = pd.DataFrame()

    if cli_args_d['quit']:
        print('Quit the program and don\'t show menu.')
        log_msg = cf.log_file + ' - Quit by user request; Finish logging for ' + \
            os.path.basename(__file__) + '\n'
        cf.logger.warning(log_msg)
        raise SystemExit()

    show_menu(popular_qa_df, all_ans_df, owner_reputation_df,
        all_ques_df, num_selected_recs, a_fname, progress_msg_factor)

    log_msg = cf.log_file + ' - Finish logging for ' + \
        os.path.basename(__file__) + '\n'
    cf.logger.warning(log_msg)

'bye'

