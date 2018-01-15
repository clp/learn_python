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

    Set the value of global constant MAX_OWNERS near the top
    of the file; default is 10.  It determines how much o/p data
    will be saved.

    Set the value of global constant MAX_HI_SCORE_TERMS
    near the top of the file.
    It determines how many hi-score terms are used to search
    for text in low-score answers.  A bigger number will cause the
    program to process more data, and might find more 'hidden'
    answers that could be valuable.


Overview of Program Actions

    Read and parse the command-line options.

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
    Provide reputation data from a dataframe or a file.  If it
    does not exist, call the function to build it.

    group_data():
    Group the data by owner, and sort by mean score.

    analyze_text():
    Call functions in the nltk module to analyze the text data and
    to provide data that might reveal good but 'hidden' answers.


Other Actions

    Several functions draw specific plots, using matplotlib tools.

    Several functions write data to disk in csv or html formats.

    One program option is to search for a text string, and to show
    records that contain it, and to save them to a file.

----------------------------------------------------------

Input data format of stackoverflow.com python file from kaggle.com.

==> Answers.csv <==
    Id,OwnerUserId,CreationDate,ParentId,Score,Body
    497,50,2008-08-02T16:56:53Z,469,4,
    "<p>open up a terminal (Applications-&gt;Utilities-&gt;Terminal)
    and type this in:</p>
    ..."

==> Questions.csv <==
    Id,OwnerUserId,CreationDate,Score,Title,Body
    469,147,2008-08-02T15:11:16Z,21,How can I find the full path to a font
    from its display name on a Mac?,
    "<p>I am using the Photoshop's javascript API to find the fonts
    in a given PSD.</p>
    ..."

#TBD.Mon2018_0108_22:23  Update this text.

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


Requirements
    Python 3, tested with v 3.6.1.
    pytest for tests, tested with v 3.0.7.

"""

import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np
import os
import pandas as pd
from pandas.plotting import scatter_matrix

import config as cf
import draw_plots as dr
import nltk_ex25 as nl
import sga_show_good_answers as sga
import util.write as wr


cf.logger.info(
    cf.log_file +
    ' - Start logging for ' +
    os.path.basename(__file__))


DATADIR = 'data/'
FILEA3 = 'a3_986.csv'
INDIR = 'indir/'
LINE_COUNT = 10
MAX_COL_WID = 20
MAX_HI_SCORE_TERMS = 100 # 10
MAX_OWNERS = 100 # 20
MAX_POPULAR_QUES = 900
MAX_TOP_OWNERS = 900
TMP = 'tmp/'
popular_qa_df = pd.DataFrame()

HEADER = '''
<html>
    <head>
        <style type="text/css">
        table{
            /* max-width: 850px; */
            width: 100%;
        }


        th, td {
            overflow: auto;  /* Use auto to get H scroll bars */
            text-align: left;
            /* max-width helps line-wrap in a cell, and most code */
            /* samples in cells have no H.scroll when width=700px: */
            max-width: 700px;
            /* max-width: 50%; Using % breaks line-wrap inside a cell */
            width: auto;    /* auto is better than using % */
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
            ques_ids_pop_and_top_l, all_ques_df, all_ans_df)

    #ORG columns_l = ['Id', 'ParentId', 'Title', 'Body', 'HSTCount', 'Score']
    columns_l = ['HSTCount', 'Score', 'Id', 'ParentId', 'Title', 'Body']
    if keyword:
        qa_with_keyword_df = sga.select_keyword_recs(
            keyword, popular_qa_df, columns_l, opt_ns, DATADIR)
        #
        log_msg = 'Search for a term: ' + keyword + '\n'
        cf.logger.info(log_msg)

        if qa_with_keyword_df.empty:
            log_msg = 'fga.main(): Missing data, qa_with_keyword_df is empty.'
            print(log_msg)
            cf.logger.warning(log_msg)
            return  # TBD. What debug data to print here?

        outfile = DATADIR + 'qa_with_keyword.csv'
        qa_with_keyword_df[columns_l].to_csv(
            outfile, header=True, index=None, sep=',', mode='w')

    # Save a df to a file for review & debug.
    wr.write_full_df_to_csv_file(popular_qa_df, DATADIR, 'popular_qa.csv')

    columns_l = []
    wr.write_full_df_to_html_file(
        popular_qa_df,
        DATADIR,
        'popular_qa.html',
        columns_l)

    # Save the Q&A title & body data as HTML.
    columns_l = ['Id', 'Title', 'Body']
    wr.write_full_df_to_html_file(
        popular_qa_df,
        DATADIR,
        'popular_qa_title_body.html',
        columns_l)


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
        print(
            "ERR, Did not find this input file; re-install the fga s/w: [",
            INDIR + FILEA3 + "]")
        raise SystemExit()


def init():
    """Initialize some settings for the program.
    """

    # Initialize settings for pandas.
    pd.set_option('display.width', 0)  # 0=no limit, for debug
    pd.set_option(
        'display.max_colwidth',
        MAX_COL_WID)  # -1=no limit, for debug

    # Don't show commas in large numbers.
    # Show OwnerUserId w/o '.0' suffix.
    pd.options.display.float_format = '{:.0f}'.format


def config_data():
    """Configure path and file names for i/o data.
    """
    # Uncomment the wanted i/p files & hide the others.
    # a_fname = 'Answers.csv'
    # q_fname = 'Questions.csv'

    # Smaller data sets, used for debugging.
    # q_fname = 'q6_999994.csv'
    # a_fname = 'a6_999999.csv'
    # a_fname = 'a5_99998.csv'
    # q_fname = 'q30_99993.csv'
    a_fname = 'a3_986.csv'
    q_fname = 'q3_992.csv'
    # a_fname = 'a2.csv'
    # q_fname = 'q2.csv'

    a_infile = INDIR + a_fname
    q_infile = INDIR + q_fname

    if not os.path.isfile(a_infile):
        print("ERR, Did not find the requested input file: [", a_infile + "]")
        print("  Check that the requested Q & A files exist in the i/p dir.")
        print("  Chk src, fga:config_data() for correct a_fname and q_fname.")
        raise SystemExit()

    print('Input files, q & a:\n' + q_infile + '\n' + a_infile)

    return a_fname, a_infile, q_infile


def read_data(ans_file, ques_file):
    """Read the csv i/p files and store data into pandas dataframes.
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
    print('Num of answer records in i/p dataframe, ans_df: ' + str(numlines))
    progress_msg_factor = int(round(numlines / 10))
    return ans_df, ques_df, progress_msg_factor, numlines


def find_popular_ques(aa_df, a_fname):
    """Find the most frequent ParentIds in the answers dataframe.
    """
    popular_ids = pd.value_counts(aa_df['ParentId'])
    outfile = DATADIR + "fpq_popular_ids." + a_fname + ".csv"
    popular_ids.to_csv(outfile)
    return popular_ids


# TBD, Refactor the two group_data() funcs.
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
    # print('#D gd2: owner_grouped_df: Group by owner and sort by mean score
    # for each owner.')
    owner_grouped_df = aa_df.groupby('OwnerUserId')
    owner_grouped_df = owner_grouped_df[[
        'Score']].mean().sort_values(['Score'])

    # Copy index column into owner column; Change index column to integer
    owner_grouped_df['OwnerUserId'] = owner_grouped_df.index
    owner_grouped_df.reset_index(drop=True, inplace=True)
    owner_grouped_df.rename(columns={'Score': 'MeanScore'}, inplace=True)

    if opt_ns.verbose:
        print()
        print('gd2: len(owner_grouped_df): num of unique OwnerUserId values: ' +
              str(len(owner_grouped_df)))
        print()
    cf.logger.info('gd2_group_data(): Show owners with highest MeanScores.')
    cf.logger.info(owner_grouped_df.tail(MAX_OWNERS))

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
    # print('group_data(): Group by owner and sort by mean score for each
    # owner.')
    owner_grouped_df = aa_df.groupby('OwnerUserId')
    owner_grouped_df = owner_grouped_df[[
        'Score']].mean().sort_values(['Score'])

    # Copy index column into owner column; Change index column to integer
    owner_grouped_df['OwnerUserId'] = owner_grouped_df.index
    owner_grouped_df.reset_index(drop=True, inplace=True)
    owner_grouped_df.rename(columns={'Score': 'MeanScore'}, inplace=True)

    if opt_ns.verbose:
        print()
        print('len(owner_grouped_df): num of unique OwnerUserId values: ' +
              str(len(owner_grouped_df)))
        print()
    cf.logger.info('group_data(): Show owners with highest MeanScores.')
    cf.logger.info(owner_grouped_df.tail(MAX_OWNERS))

    # Take slice of owners w/ highest mean scores; convert to int.
    owners_a = owner_grouped_df['OwnerUserId'].values
    top_scoring_owners_a = np.vectorize(np.int)(owners_a[-MAX_OWNERS:])
    if opt_ns.verbose:
        print('top_scoring_owners_a: ', top_scoring_owners_a)
        print()

    owners_df_l = []
    lo_score_limit = opt_ns.lo_score_limit
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
    if opt_ns.verbose:
        print('lo_score_limit: ', lo_score_limit)
        print('Length of lo_scores_for_top_owners_df: ',
              len(lo_scores_for_top_owners_df))
        print()
    outfile = DATADIR + 'lo_scores_for_top_owners.csv'
    lo_scores_for_top_owners_df.to_csv(
        outfile, header=True, index=None, sep=',', mode='w')
    cf.logger.info('group_data(): lo_scores_for_top_owners_df: ')
    cf.logger.info(lo_scores_for_top_owners_df)

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

    # Make list of unique ParentId's:
    ques_ids_from_top_own_l = list(set(hi_scoring_owners_df['ParentId']))

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
    # Notes on settings used below:
    # For q6 & a6 data:
    # set(ques_ids_from_top_own_l[:500]).intersection(
    # set(popular_ids_a[:500])))
    #
    # For q3 & a3 data:
    # set(ques_ids_from_top_own_l[:40]).intersection(
    # set(popular_ids_a[:10])))
    ques_ids_pop_and_top_l = list(
        set(ques_ids_from_top_own_l[:MAX_TOP_OWNERS]).intersection(set(popular_ids_a[:MAX_POPULAR_QUES])))
        #OK set(ques_ids_from_top_own_l[:900]).intersection(set(popular_ids_a[:900])))
    log_msg = 'find_pop_and_top_ques_ids(): len(ques_ids_pop_and_top_l) : ' + \
        str(len(ques_ids_pop_and_top_l))
    cf.logger.info(log_msg)
    log_msg = "find_pop*(): ques_ids_pop_and_top_l, top-N parent id\'s to chk: " + \
        str(ques_ids_pop_and_top_l[0:10])
    cf.logger.info(log_msg)
    if opt_ns.verbose:
        print(
            'ques_ids_pop_and_top_l, parent id\'s to examine: ',
            ques_ids_pop_and_top_l[:])
    return ques_ids_pop_and_top_l


def combine_related_q_and_a(ques_ids_pop_and_top_l, all_ques_df, aa_df):
    """Get each Q in the list of selected ParentId's, and the related A's.
    Loop over each question and store the Q & A data in a dataframe.

    Then call analyze_text() on each group of Q with A's,
    which calls the routines that perform the natural language processing
    of the text data.
    """
    global popular_qa_df

    ques_match_df = all_ques_df[all_ques_df['Id'].isin(ques_ids_pop_and_top_l)]
    ans_match_df = aa_df[aa_df['ParentId'].isin(ques_ids_pop_and_top_l)]

    # Print a blank line before showing Question data from analyze*().
    if opt_ns.debug:
        print()

    # Build each Q&A group: one Q w/ all its A.'s
    for i, qid in enumerate(ques_ids_pop_and_top_l):
        if opt_ns.verbose:
            if i % 20 == 0:
                print("combine_related_q_and_a():progress count: ", i)
        qm_df = ques_match_df[ques_match_df['Id'] == qid]
        am_df = ans_match_df[ans_match_df['ParentId'] == qid]
        qagroup_from_pop_top_ques_df = pd.concat(
            [qm_df, am_df]).reset_index(drop=True)
        #
        # Change NaN values to zero, and the field type from float64 to int64,
        # to help with search functionality.
        qagroup_from_pop_top_ques_df['ParentId'] = qagroup_from_pop_top_ques_df['ParentId'].fillna(
            0).astype(int)
        qagroup_from_pop_top_ques_df['OwnerUserId'] = qagroup_from_pop_top_ques_df['OwnerUserId'].fillna(
            0).astype(int)
        #
        if qagroup_from_pop_top_ques_df.empty:
            # Skip this qid, it does not match what we seek.
            continue
        cf.logger.info('qagroup_from_pop_top_ques_df.head(1): ')
        cf.logger.info(qagroup_from_pop_top_ques_df.head(1))

        # Analyze data w/ nlp s/w.
        popular_qa_df = analyze_text(qagroup_from_pop_top_ques_df)

    # End combine_related_q_and_a().
    return popular_qa_df


def compute_record_selector(numlines):
    """Compute the number of records to use for computation and display.

    Return that integer.
    """

    # TBD, use Default 0.01 in production? Move to global scope?
    SELECTOR_RATIO = 0.10
    num_selected_recs = max(5, int(numlines * SELECTOR_RATIO))
    log_msg = ("  SELECTOR_RATIO,  number of selected recs: " +
               str(SELECTOR_RATIO) + ", " + str(num_selected_recs))
    cf.logger.info(log_msg)

    return num_selected_recs


def analyze_text(qagroup_from_pop_top_ques_df):
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

    cf.logger.info("NLP Step 2. Process the words of each input line.")
    if opt_ns.debug:
        print(
            'analyze_text(): Questions in qagroup_from_pop_top_ques_df: Id, Score, Title:')
        print(
            #ORG qagroup_from_pop_top_ques_df)
            qagroup_from_pop_top_ques_df[['Id', 'Score', 'Title']].iloc[0].values)
    clean_ans_bodies_l = nl.clean_raw_data(qagroup_from_pop_top_ques_df)
    if not clean_ans_bodies_l:
        print('analyze_text(), No text to analyze; exit now.')
        raise SystemExit()

    cf.logger.info("NLP Step 3. Build a bag of words and their counts.")
    (vocab_l, dist_a) = nl.make_bag_of_words(clean_ans_bodies_l)
    words_sorted_by_count_l = nl.sort_vocab(vocab_l, dist_a)

    cf.logger.info("NLP Step 7. Search lo-score A's for hi-score text.")
    qa_with_hst_df = nl.find_hi_score_terms_in_bodies(
        words_sorted_by_count_l,
        clean_ans_bodies_l,
        MAX_HI_SCORE_TERMS, qagroup_from_pop_top_ques_df)
    popular_qa_df = pd.concat(
        [popular_qa_df, qa_with_hst_df]).reset_index(drop=True)

    return popular_qa_df


def show_menu(qa_df, all_ans_df, owner_reputation_df, opt_ns):
    """Show prompt to user; get and handle their request.
    """
    user_menu = """    The menu choices:
    drm: calculate reputation matrix of owners
    dsm: draw and plot q&a statistics matrix
    d: draw default plot of current data
    dh: draw default histogram plot of current data
    dm: draw scatter matrix plot of current data
    h, ?: show help text, the menu
    lek: look for exact keywords in questions and answers
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
    while user_cmd == "":  # Repeat the prompt if no i/p given.
        user_cmd = input(cmd_prompt)
    print("User entered this cmd: ", user_cmd)

    # Loop to handle user request.
    while user_cmd:
        if user_cmd.lower() == 'm':
            print(user_menu)
            # Clear user_cmd here to avoid infinite repetition of while loop.
            user_cmd = ''
        elif user_cmd.lower() == 'q':
            print("Quit the program.")
            log_msg = cf.log_file + ' - Quit, user req; Finish logging for ' + \
                os.path.basename(__file__) + '\n'
            cf.logger.warning(log_msg)
            raise SystemExit()
        elif user_cmd == '?' or user_cmd == 'h':
            print(user_menu)
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
                dr.draw_scatter_plot(
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
                dr.draw_histogram_plot(popular_qa_df)
        elif user_cmd.lower() == 'dm':  # Scatter matrix plot
            user_cmd = ''
            if popular_qa_df.empty:
                print("WARN: dataframe empty or not found; try restarting.")
            else:
                print("Drawing the default scatter matrix plot.")
                dr.draw_scatter_matrix_plot(popular_qa_df)
        # drm: Draw Reputation matrix, mean, answers only; scatter.
        elif user_cmd.lower() == 'drm':
            user_cmd = ''
            owner_reputation_df = check_owner_reputation(
                all_ans_df, owner_reputation_df)
            #
            if owner_reputation_df.empty:
                print("WARN: owner reputation dataframe empty or not found.")
            else:
                print("NOTE: Drawing the owner reputation scatter matrix plot.")
                dr.draw_scatter_matrix_plot(
                    owner_reputation_df[['MeanScore', 'OwnerUserId']])
        # dsm: Draw q&a statistics matrix
        elif user_cmd.lower() == 'dsm':
            user_cmd = ''
            owner_reputation_df = check_owner_reputation(
                all_ans_df, owner_reputation_df)
            #
            qa_stats_df = build_stats(popular_qa_df, owner_reputation_df)
            #
            if qa_stats_df.empty:
                print("WARN: qa_stats_df empty or not found.")
            else:
                print("NOTE: Drawing the qa_stats_df scatter matrix plot.")
                dr.draw_scatter_matrix_plot(
                    qa_stats_df[['Score', 'BodyLength', 'OwnerRep', 'HSTCount']])
        # lek: Look for exact keywords in the Q&A df; now case sensitive.
        elif user_cmd.lower() == 'lek':
            user_cmd = 'lek'  # Force menu to always repeat the lek prompt.
            search_prompt = "\nType a search term; or press Enter for main menu: "
            search_term = input(search_prompt)
            if search_term == "":  # Return to main menu and ask for a cmd.
                user_cmd = 'm'
                continue
            print("User entered this search_term: ", search_term)
            log_msg = cf.log_file + ' - Search term entered by user: ' + \
                search_term + '\n'
            cf.logger.warning(log_msg)
            #
            columns_l = [
                'HSTCount',
                'Score',
                'Id',
                'ParentId',
                'Title',
                'Body',
                'HiScoreTerms']
            q_a_group_with_keyword_df = pd.DataFrame()  # Initlz at each call
            #
            # TBD Chg popular_qa_df to a df w/ more records, for dbg & initial
            # use.  Beware of memory & performance issues.
            q_a_group_with_keyword_df = sga.select_keyword_recs(
                search_term, popular_qa_df, columns_l, opt_ns, DATADIR)
        else:
            print("Got bad cmd from user: ", user_cmd)
            print(user_menu)
            # Clear user_cmd here to avoid infinite repetition of while loop.
            user_cmd = ''
        # Show prompt & wait for a cmd.
        print("======================\n")
        while user_cmd == "":  # Repeat the prompt if no i/p given.
            user_cmd = input(cmd_prompt)
    print("#D End of the cmd interpretation loop; return.\n")

    return


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
        try:
            owner_rep = round(
                or_df.loc[or_df['OwnerUserId'] == ouid, 'MeanScore'].iloc[0])
            # Add new column to df.
            qa_stats_df.loc[index, 'OwnerRep'] = owner_rep
        except IndexError:
            # TBD, Some answers in the data file were made by Owners
            # who are not yet in the reputation df.
            print(
                "build_stats(): ouid not in owner reputation dataframe;",
                " index,ouid: ",
                index,
                ouid)
            print("build_stats():data from the problem row in qa_df:\n", row)
            print()

        # Save length of body text of each answer.
        qa_stats_df.loc[index, 'BodyLength'] = len(row['Body'])

    qa_stats_df = qa_stats_df[['Id',
                               'ParentId',
                               'OwnerUserId',
                               'Score',
                               'BodyLength',
                               'OwnerRep',
                               'HSTCount']]

    stats_fname = DATADIR + 'qa_stats_by_dsm.csv'
    wr.save_prior_file('', stats_fname)
    qa_stats_df.to_csv(stats_fname)

    stats_fname = DATADIR + 'qa_stats_by_dsm.html'
    wr.save_prior_file('', stats_fname)
    qa_stats_df.to_html(stats_fname)

    return qa_stats_df


def check_owner_reputation(all_ans_df, owner_reputation_df):
    """Check for dataframe with reputation of each OwnerUserId.
    If not found, then calculate reputation of each OwnerUserId
    in the i/p data, based on Score of all answers they provided.
    Save the data to a disk file and use it when needed, so the
    calculation need not be done every time this program runs.
    """
    own_rep_file = DATADIR + 'owner_reputation.csv'
    # TBD Must chk i/p file & replace owner_rep*.csv if
    #  a different file was used.  OR, just build this file from Answers.csv
    #  which should have all answers & produce good reputation data.

    if not owner_reputation_df.empty:
        return owner_reputation_df

    if os.path.exists(own_rep_file):
        print("owner rep file, " + own_rep_file + ", found; read it.")
        owner_reputation_df = pd.read_csv(
            own_rep_file,
            encoding='latin-1',
            warn_bad_lines=False,
            error_bad_lines=False)
        return owner_reputation_df
    else:
        print(
            "owner rep file, " +
            own_rep_file +
            ", not found; build it.")
        print("This should be a one-time operation w/ data saved on disk.")
        owner_reputation_df = gd2_group_data(all_ans_df)
        owner_reputation_df.to_csv(own_rep_file)
    return owner_reputation_df


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

    # Specify an option that takes a string arg: -s word1 word2 ...
    parser.add_argument(
        '-s',
        '--search',
        help='Search the Q & A Collection for this term',
        type=str
    )

    parser.add_argument(
        '-q',
        '--quit',
        help='Stop the program before showing the menu; used for testing',
        action='store_true')

    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-vv', '--veryverbose', action='store_true')
    return parser


if __name__ == '__main__':

    check_install()

    parser = get_parser()
    opt_ns = parser.parse_args()

    keyword = False  # Init before reading CLI argument.

    if opt_ns.search:
        keyword = opt_ns.search
        print('Search the data for this term: ', keyword)
        columns_l = ['HSTCount', 'Score', 'Id', 'ParentId', 'Title', 'Body']

    main(popular_qa_df)

    if opt_ns.quit:
        print('Quit the program and don\'t show menu.')
        log_msg = cf.log_file + ' - Quit, user req; Finish logging for ' + \
            os.path.basename(__file__) + '\n'
        cf.logger.warning(log_msg)
        raise SystemExit()

    owner_reputation_df = pd.DataFrame()

    show_menu(popular_qa_df, all_ans_df, owner_reputation_df, opt_ns)

    log_msg = cf.log_file + ' - Finish logging for ' + \
        os.path.basename(__file__) + '\n'
    cf.logger.warning(log_msg)
