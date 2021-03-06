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
Use Natural Language Processing, NLP, tools and techniques for text analysis.
Save the o/p to a file for further evaluation & processing.

Usage:
    fga_find_good_answers.sh [-h]
    python fga_find_good_answers.py
    pydoc  fga_find_good_answers


Initialization

    Set the value of global constant MAX_OWNERS in config.py.
    It affects how much o/p data to save.

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

    Some actions use the following functions.

    find_top_owners():
    Group the data by owner, and sort by mean score.

    analyze_text():
    Call functions in the module w/ NLP routines to analyze the text
    data and to provide data that might reveal good but 'hidden' answers.


Other Actions

    TBD.

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
import numpy as np
import os
import pandas as pd
from pandas.plotting import scatter_matrix
import re
import time

import config as cf
import nltk_ex25 as nl
import save_output as sav
import sga_show_good_answers as sga
import text_ui as tui
import util.misc as ut


CURRENT_FILE = os.path.basename(__file__)
DATADIR = cf.DATADIR
FILEA3 = 'a3_986.csv'
INDIR = 'indir/'
MAX_COL_WID = cf.MAX_COL_WID
MAX_HI_SCORE_TERMS = 100 # 10
MAX_OWNERS = cf.MAX_OWNERS
MAX_POPULAR_QUES = 900
MAX_TOP_OWNERS = 900

cf.logger.info( cf.log_file + ' - Start logging for ' + CURRENT_FILE)
popular_qa_df = pd.DataFrame()


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

    all_ques_df = \
        read_df_in_csv_file(q_infile)

    all_ans_df = \
        read_df_in_csv_file(a_infile)

    popular_ids_a = \
        find_popular_ques(all_ans_df, a_fname)

    owner_reputation_df = \
        tui.group_owners(all_ans_df, opt_ns)

    top_scoring_owners_a = \
        find_top_owners(all_ans_df, owner_reputation_df)

    ques_ids_from_top_own_l = \
        find_ques_ids_from_top_owners(top_scoring_owners_a, all_ans_df)

    ques_ids_pop_and_top_l = \
        find_pop_and_top_ques_ids(ques_ids_from_top_own_l, popular_ids_a)

    popular_qa_df = \
        combine_related_q_and_a(
            ques_ids_pop_and_top_l, all_ques_df, all_ans_df)

    columns_l = []
    qa_with_keyword_df = pd.DataFrame()
    if keyword:
        qa_with_keyword_df = \
            sga.select_keyword_recs(keyword, opt_ns, popular_qa_df, columns_l)
        sav.save_search_output(qa_with_keyword_df)
        sav.save_full_search_output(qa_with_keyword_df, keyword)

    sav.save_basic_output(popular_qa_df)


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
    q_fname = 'q6_999994.csv'
    a_fname = 'a6_999999.csv'
    # a_fname = 'a5_99998.csv'
    # q_fname = 'q30_99993.csv'
    a_fname = 'a3_986.csv'
    q_fname = 'q3_992.csv'
    # a_fname = 'a2.csv'
    # q_fname = 'q2.csv'
    # q_fname = 'q25.csv'
    # a_fname = 'a26.csv'
    # q_fname = 'q26.csv'
    # a_fname = 'a1_5430_q5419.csv'
    # q_fname = 'q1_5419.csv'

    a_infile = INDIR + a_fname
    q_infile = INDIR + q_fname

    if not os.path.isfile(a_infile):
        print("ERR, Did not find the requested input file: [", a_infile + "]")
        print("  Check that the requested Q & A files exist in the i/p dir.")
        print("  Chk src, fga:config_data() for correct a_fname and q_fname.")
        raise SystemExit()

    #D print('Input files, q & a:\n' + q_infile + '\n' + a_infile)

    return a_fname, a_infile, q_infile


def read_df_in_csv_file(infile):
    """Read a csv file and return its data in a pandas dataframes.
    """
    out_df = pd.read_csv(
        infile,
        encoding='latin-1',
        warn_bad_lines=False,
        error_bad_lines=False)

    print('read*(): infile: ' + infile)
    print('read*(): Num of records in i/p dataframe: ' + str(len(out_df)))
    return out_df


def find_popular_ques(aa_df, a_fname):
    """Find the most frequent ParentIds in the answers dataframe.
    """
    popular_ids = pd.value_counts(aa_df['ParentId'])
    outfile = DATADIR + "fpq_popular_ids." + a_fname + ".csv"
    popular_ids.to_csv(outfile)
    popular_ids_a = popular_ids.index.values
    return popular_ids_a


# TBD, Refactor the find_top_owners() func.
    # Thu2017_0907_23:30 , Should lo_score*df be returned for use elsewhere?
    # It is printed to file, maybe to log.
    # Is it needed to find good answers?
    # Also chk find_ques_ids_from_top_owners() ;  common code.
    # Has copied lines from find_top_owners().
    # Does not have lo_score* vars & code.
def find_top_owners(aa_df, owner_reputation_df):
    """Use df grouped by OwnerUserId, and sorted by
    mean score for answers only
    for each owner (question scores are not counted).
    Make a numpy array of owners w/ highest mean scores.

    TBD.1, Find low score answers for these hi-score owners;
    then mark the low score  answers for evaluation.
    Low score is any score below lo_score_limit.
    """

    # Take slice of owners w/ highest mean scores; convert to int.
    owners_a = owner_reputation_df['OwnerUserId'].values
    top_scoring_owners_a = np.vectorize(np.int)(owners_a[-MAX_OWNERS:])
    if opt_ns.verbose:
        print('top_scoring_owners_a: ', top_scoring_owners_a)
        print()

    # Build list of owners w/ rows from the answers df.
    owners_l = []
    lo_score_limit = opt_ns.lo_score_limit
    for owner in top_scoring_owners_a:
        # Build a pandas series of booleans for filtering.
        answered_by_o2_sr = (aa_df.OwnerUserId == owner)
        # Build a pandas df with rows for all answers by one owner.
        answers_df = aa_df[[
            'Id', 'OwnerUserId', 'Score']][answered_by_o2_sr]

        # Build filter and df with rows for all low-score answers by one owner.
        lo_score_by_o2_sr = (answers_df.Score < lo_score_limit)
        lo_score_answers_by_o2_df = answers_df[[
            'Id', 'OwnerUserId', 'Score']][lo_score_by_o2_sr]
        owners_l.append(lo_score_answers_by_o2_df)

    # TBD These are the answers to examine for useful data, even though
    # they have low scores.
    # View these answers and evaluate them manually; and analyze them
    # with other s/w.
    lo_scores_for_top_owners_df = pd.concat(owners_l)
    if opt_ns.verbose:
        print('lo_score_limit: ', lo_score_limit)
        print('Length of lo_scores_for_top_owners_df: ',
              len(lo_scores_for_top_owners_df))
        print()
    outfile = DATADIR + 'lo_scores_for_top_owners.csv'
    lo_scores_for_top_owners_df.to_csv(
        outfile, header=True, index=None, sep=',', mode='w')
    cf.logger.info('fga.find_top_owners(): lo_scores_for_top_owners_df: ')
    cf.logger.info(lo_scores_for_top_owners_df)

    return top_scoring_owners_a


def find_ques_ids_from_top_owners(top_scoring_owners_a, aa_df):
    """Make a list of all answer records by the high-score owners.
    Use that list to build a list of ParentId's (ie, questions)
    to collect for evaluation.
    Return that list of ParentId's.
    """
    owners_l = []
    for owner in top_scoring_owners_a:
        # Build a filter and df with rows for all answers by one owner.
        answered_by_owner_sr = (aa_df.OwnerUserId == owner)
        answers_df = aa_df[['Id', 'OwnerUserId',
                            'ParentId', 'Score']][answered_by_owner_sr]
        owners_l.append(answers_df)

    hi_scoring_owners_df = pd.concat(owners_l)

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
    # MAX_TOP_OWNERS = 500
    # MAX_POPULAR_QUES = 500
    #
    # For q3 & a3 data:
    # MAX_TOP_OWNERS = 40
    # MAX_POPULAR_QUES = 10
    # OR
    # MAX_POPULAR_QUES = 900
    # MAX_TOP_OWNERS = 900
    #
    ques_ids_pop_and_top_l = list(
        set(ques_ids_from_top_own_l[:MAX_TOP_OWNERS]).intersection(
            set(popular_ids_a[:MAX_POPULAR_QUES])))

    cf.logger.info('fga.find_pop_and_top*(): len(ques_ids_pop_and_top_l): ' + \
        str(len(ques_ids_pop_and_top_l)))

    cf.logger.info("fga.find_pop_and_top*(): ques_ids_pop_and_top_l, " + \
        "top-N parent id\'s to chk: " + str(ques_ids_pop_and_top_l[0:5]))

    if opt_ns.verbose:
        print(
            'ques_ids_pop_and_top_l, parent id\'s to chk: ',
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
    #
    # Prepare to show progress.
    numlines = len(ques_ids_pop_and_top_l)
    progress_i = ut.calc_progress(numlines)
    prior_time = time.time()
    # Build each Q&A group: one Q w/ all its A.'s
    for i, qid in enumerate(ques_ids_pop_and_top_l):
        if opt_ns.verbose:
            if i % progress_i == 0:
                lap_time = int(round(time.time() - prior_time))
                prior_time = time.time()
                print("fga:combine_related_q*a(): progress count {} of total {} in {} seconds."
                        .format(i, numlines, lap_time))
        qm_df = ques_match_df[ques_match_df['Id'] == qid]
        am_df = ans_match_df[ans_match_df['ParentId'] == qid]
        qagroup_poptop_df = pd.concat([qm_df, am_df]).reset_index(drop=True)
        if qagroup_poptop_df.empty:
            # Skip this qid, it does not match what we seek.
            continue
        #
        # Change NaN values to zero, and the field type from float64 to int64,
        # to help with search functionality.
        qagroup_poptop_df['ParentId'] = \
                qagroup_poptop_df['ParentId'].fillna( 0).astype(int)
        qagroup_poptop_df['OwnerUserId'] = \
                qagroup_poptop_df['OwnerUserId'].fillna( 0).astype(int)
        #
        cf.logger.info('fga.combine*(): qagroup_poptop_df.head(1): ')
        cf.logger.info(qagroup_poptop_df.head(1))

        # Analyze data w/ nlp s/w.
        popular_qa_df = analyze_text(qagroup_poptop_df)

    # End combine_related_q_and_a().
    return popular_qa_df 


def analyze_text(qagroup_poptop_df):
    """Use a Q&A group of one Q w/ its A's for i/p.
    Process the text data w/ NLP routines, in the nltk module or others.

    Important variables.

    qagroup_poptop_df:
    A dataframe with one Q&A group (ie, one question with its related
    answers), which is selected from all such groups based on being 'pop'
    (popular, questions with several answers) and 'top' (having one
    or more answers by high-reputation owners).

    """
    global popular_qa_df

    cf.logger.info("fga.ana*(): NLP Step 2. Process the words of each input line.")
    if opt_ns.debug:
        print('analyze*(): Ques in qagroup_poptop_df: Id, Score, Title:')
        print(qagroup_poptop_df[['Id', 'Score', 'Title']].iloc[0].values)
    clean_ans_bodies_l = nl.clean_raw_data(qagroup_poptop_df)
    if not clean_ans_bodies_l:
        print('analyze_text(), No text to analyze; exit now.')
        raise SystemExit()

    cf.logger.info("fga.ana*(): NLP Step 3. Build a bag of words and their counts.")
    (vocab_l, dist_a) = nl.make_bag_of_words(clean_ans_bodies_l)
    words_sorted_by_count_l = nl.sort_vocab(vocab_l, dist_a)

    cf.logger.info("fga.ana*(): NLP Step 7. Search lo-score A's for hi-score text.")
    qa_with_hst_df = nl.find_hi_score_terms_in_bodies(
        words_sorted_by_count_l,
        clean_ans_bodies_l,
        MAX_HI_SCORE_TERMS, qagroup_poptop_df)
    popular_qa_df = pd.concat(
        [popular_qa_df, qa_with_hst_df]).reset_index(drop=True)

    return popular_qa_df


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
    return parser


if __name__ == '__main__':

    check_install()

    parser = get_parser()
    opt_ns = parser.parse_args()

    #ORG keyword = False  # Init before reading CLI argument.
    keyword = ''  # Init before reading CLI argument.

    if opt_ns.search:
        keyword = opt_ns.search
        print('Search the data for this term: ', keyword)

    main(popular_qa_df)

    if opt_ns.quit:
        print('Quit the program and don\'t show menu.')
        cf.logger.warning(cf.log_file + \
                ' - Quit by user; Finish logging for ' + CURRENT_FILE + '\n')
        raise SystemExit()

    tui.show_menu(popular_qa_df, all_ans_df, opt_ns)

    cf.logger.warning(
            cf.log_file + ' - Finish logging for ' +  CURRENT_FILE + '\n')
