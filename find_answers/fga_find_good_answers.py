# -*- coding: utf-8 -*-

# Using ~/anaconda3/bin/python: Python 3.5.2 :: Anaconda 4.2.0 (64-bit)
# Using Python 3.4.5 :: Anaconda 4.3.0 (64-bit), since Tue2017_0710

"""fga_find_good_answers.py


   Find answers in stackoverflow that might be good, but 'hidden'
   because they have low scores.
   Look for such answers from contributors who have high scores
   based on their other questions & answers.
   Save the o/p to a file for further evaluation & processing.

   Usage:
     ./fga_find_good_answers.sh
     pydoc  fga_find_good_answers
     python fga_find_good_answers.py

     Set the value of num_owners at the bottom of the file;
     default is 10.  It determines how much o/p data will be saved.

     Set the value of num_hi_score_terms at the bottom of the file;
     It determines how many hi-score terms are used to search
     text in low-score answers.  A bigger number will cause the
     program to run longer, and might find more 'hidden' answers
     that could be valuable.

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

Output data format of popular_qa.csv o/p file from this program.
    Note: question records have a Title but no ParentId;
    answer records have a ParentId (which is the related
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
"""

version = '0.0.6'

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
INDIR = 'indir/'
MAXCOLWID = 20
TMP = 'tmp/'
TMPDIR = 'data/'
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
            max-width: 500px;
            width: 100%;
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
    """Analyze input data and produce o/p by calling various functions.
    """
    # all_ans_df must be global to calculate reputations of owners, because
    # it is changed here in main(), and used outside main().
    global all_ans_df

    init()

    a_fname, a_infile, q_infile = \
        config_data()

    all_ans_df, all_ques_df, progress_msg_factor, numlines = \
        read_data( a_infile, q_infile)

    popular_ids = \
        find_popular_ques(all_ans_df, a_fname)

    popular_ids_a = popular_ids.index.values

    top_scoring_owners_a, owner_grouped_df = \
        group_data(all_ans_df)

    parent_id_l = \
        find_question_ids(top_scoring_owners_a, all_ans_df)

    pop_and_top_l = \
        select_questions(parent_id_l, popular_ids_a)

    popular_qa_df = \
        combine_related_q_and_a(
            pop_and_top_l, all_ques_df, all_ans_df, numlines, a_fname, progress_msg_factor)

    # Save a df to a file for review & debug.
    write_full_df_to_csv_file(popular_qa_df, TMPDIR, 'popular_qa.csv')

    columns_l = []
    write_full_df_to_html_file(popular_qa_df, TMPDIR, 'popular_qa.html', columns_l)

    # Save the Q&A title & body data as HTML.
    columns_l = ['Id', 'Title', 'Body']
    write_full_df_to_html_file(popular_qa_df, TMPDIR, 'all_qa_title_body.html', columns_l)

    #TBD Chg this if needed; or remove.
    if keyword:
        # Write records containing keywords to a csv file.
        qa_with_keyword_df = select_keyword_recs(
            keyword, popular_qa_df, columns_l)
        outfile = DATADIR + 'qa_with_keyword.csv'
        qa_with_keyword_df[columns_l].to_csv(
            outfile, header=True, index=None, sep=',', mode='w')


def init():
    """Initialize some settings for the program.
    """
    if args['debug']:
        end = 55
        print('Running in debug mode.')
        print('  end set to: ', end)
        print()

    # Initialize settings for pandas.
    pd.set_option('display.width', 0)  # 0=no limit, for debug
    pd.set_option('display.max_colwidth', MAXCOLWID) # -1=no limit, for debug

    # Don't show commas in large numbers.
    # Show OwnerUserId w/o '.0' suffix.
    pd.options.display.float_format = '{:.0f}'.format


def config_data():
    """Configure path and file names for i/o data.
    """
    # TBD Make the in & out dirs w/ this program, if they don't exist?
    # TBD Include the test data files w/ this project.
    #D a_fname = 'Answers.csv'
    #D q_fname = 'Questions.csv'

    # Smaller data sets, used for debugging.
    q_fname = 'q6_999994.csv'
    a_fname = 'a6_999999.csv'
    #D a_fname = 'a5_99998.csv'
    #D q_fname = 'q30_99993.csv'
    a_fname = 'a3_986.csv'
    q_fname = 'q3_992.csv'
    # D a_fname = 'a2.csv'
    # D q_fname = 'q2.csv'

    a_infile = INDIR + a_fname
    q_infile = INDIR + q_fname

    print('Input files, q & a:\n' + q_infile + '\n' + a_infile)
    print()

    return a_fname, a_infile, q_infile


def read_data(ans_file, ques_file):
    """Read the csv i/p files and store data into a pandas data frame.
    Compute a factor that dictates how progress will be indicated
    during read operations.
    """
    # TBD, Maybe rm latin-1 encoding here also?
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
    if args['verbose']:
        print('#D gd2: Show owners with ', str(num_owners), ' highest MeanScores.')
        # See highest scores at bottom:
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
    print('owner_grouped_df: Group by owner and sort by mean score for each owner.')
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
    if args['verbose']:
        print('Show owners with ', str(num_owners), ' highest MeanScores.')
        # See highest scores at bottom:
        print(owner_grouped_df.tail(num_owners))
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
    lo_scores_for_top_users_df = pd.concat(owners_df_l)
    print('lo_score_limit: ', lo_score_limit)
    print('Length of lo_scores_for_top_users_df: ',
          len(lo_scores_for_top_users_df))
    outfile = DATADIR + 'lo_scores_for_top_users.csv'
    lo_scores_for_top_users_df.to_csv(
        outfile, header=True, index=None, sep=',', mode='w')
    cf.logger.info('group_data(): lo_scores_for_top_users_df: ')
    cf.logger.info(lo_scores_for_top_users_df)
    if args['verbose']:
        print('lo_scores_for_top_users_df: ')
        print(lo_scores_for_top_users_df)
    print()

    return top_scoring_owners_a, owner_grouped_df


def find_question_ids(top_scoring_owners_a, aa_df):
    """Make a list of all answer records by the high-score owners.
    Use that list to build a list of Question Id's (= ParentId)
    to collect for evaluation.
    Return that list of question Id's.
    """
    owners_df_l = []
    for owner in top_scoring_owners_a:
        # Get a pandas series of booleans for filtering:
        answered_by_owner_sr = (aa_df.OwnerUserId == owner)
        # Get a pandas df with rows for all answers of one user:
        answers_df = aa_df[['Id', 'OwnerUserId',
                                 'ParentId', 'Score']][answered_by_owner_sr]
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


def select_questions(parent_id_l, popular_ids_a):
    """Make a list of questions to use in further processing.
    pop_and_top_l:
    Find parent IDs that are popular (have several answers), and
    assume that some of those answers come from top owners
    (owners who have high mean scores).

    Set the size of the following list slices to get enough o/p to analyze.
    For i/p data files q6 and a6:
    With slice limits at 40 and 30, got 2 Q, 36 A.
    With slice limits at 400 and 300, got 26 Q, 308 A.
    """
    pop_and_top_l = list(
        # For q6 & a6 data: set(parent_id_l[:500]).intersection(set(popular_ids_a[:500])))
        # For q3 & a3 data: set(parent_id_l[:40]).intersection(set(popular_ids_a[:10])))
        #D set(parent_id_l[:40]).intersection(set(popular_ids_a[:10])))
        set(parent_id_l[:900]).intersection(set(popular_ids_a[:900])))
    log_msg = 'select_questions(): len(pop_and_top_l) : ' + str(len(pop_and_top_l))
    cf.logger.info(log_msg)
    log_msg = "select_questions(): pop_and_top_l, top-N parent id\'s to examine: " + str(pop_and_top_l[0:10])
    cf.logger.info(log_msg)
    if args['verbose']:
        print('pop_and_top_l, parent id\'s to examine: ', pop_and_top_l[:])
    return pop_and_top_l


def combine_related_q_and_a(pop_and_top_l, all_ques_df, aa_df, numlines, a_fname, progress_msg_factor):
    """Get each Q in the list of ParentId's, and the related A's.
    Loop over all the question Id's and store all Q & A data in a dataframe.

    Then call analyze_text() on each group of Q with A's,
    which calls the routines that perform the natural language processing
    of the text data.
    """
    global popular_qa_df

    ques_match_df = all_ques_df[all_ques_df['Id'].isin(pop_and_top_l)]
    ans_match_df = aa_df[aa_df['ParentId'].isin(pop_and_top_l)]

    print('#D len of ques_match_df: ', len(ques_match_df))
    print('#D len of ans_match_df: ', len(ans_match_df))
    print('\n#D ques_match_df.head() & ans_match_df.head(): ')
    print(ques_match_df.head())
    print()
    print(ans_match_df.head())

    i = 0
    # Build each Q&A group: one Q w/ all its A.'s
    # TBD, How to do this w/o explicit loop, using df tools?
    # TBD, Combine or replace the ques_match_df & ans*df code above w/ this?
    # Maybe delete those 'intermediate' results?
    for qid in pop_and_top_l:
        i += 1
        # OK if(i % progress_msg_factor == 0):
        if(i % 20 == 0):
            print("#D combine_related_q_and_a():progress count: ", i)
        qm_df = ques_match_df[ques_match_df['Id'] == qid]
        am_df = ans_match_df[ans_match_df['ParentId'] == qid]
        qagroup_df = pd.concat([qm_df, am_df]).reset_index(drop=True)
        # D print("\n#D qagroup_df.head(): ")
        # D print(qagroup_df.head())
        cf.logger.info('qagroup_df.head(1): ')
        cf.logger.info(qagroup_df.head(1))

        popular_qa_df = analyze_text(
             qagroup_df, numlines, a_fname, progress_msg_factor)

        #
        # TBD
        # Analyze qagroup_df w/ nlp s/w in this loop;
        # or save each qagroup_df to a separate df for later processing;
        # or save each qagroup_df to a disk file for later processing;
        # or move this code to the nlp processing section of the program.
        #

    # D print('\n#D fga, End of debug code; exiting.')
    # D raise SystemExit()

    return popular_qa_df


def analyze_text(qagroup_df, numlines, a_fname, progress_msg_factor):
    """Use a Q&A group of one Q w/ its A's for i/p.
    Process the text data w/ the routines in the nltk module, which use
    natural language tools.
    """
    global popular_qa_df

    #TBD.1, This code may write o/p to file but should not; fix nl module.
    cf.logger.info("NLP Step 2. Process the words of each input line.")
    clean_ans_bodies_l = nl.clean_raw_data(
        a_fname, progress_msg_factor, qagroup_df, TMPDIR)
    # D print('\n#D, clean_ans_bodies_l[:1]')
    # D print(clean_ans_bodies_l[:1])

    cf.logger.info("NLP Step 3. Build a bag of words and their counts.")
    (vocab, dist) = nl.make_bag_of_words(clean_ans_bodies_l)
    # D print('\n#D, vocab[:1]')
    # D print(vocab[:1])
    words_sorted_by_count_l = nl.sort_save_vocab(
        '.vocab', vocab, dist, a_fname, TMPDIR)
    # Save the original list for later searching.
    words_sorted_by_count_main_l = words_sorted_by_count_l

    cf.logger.info('NLP Step 4. Sort Answers by Score.')
    score_df, num_selected_recs = nl.sort_answers_by_score(numlines, qagroup_df)

    cf.logger.info('NLP Step 5. Find most freq words for top-scoring Answers.')
    score_top_n_df = score_df[['Id']]

    cf.logger.debug("score_top_n_df.tail():")
    cf.logger.debug(score_top_n_df.tail(20))
    # Use top_n Answers & count their words.
    cf.logger.info(
        "For top ans: Cleaning and parsing the training set bodies...")

    top = True
    top_n_bodies = nl.find_freq_words(
        top, score_top_n_df, num_selected_recs, progress_msg_factor, qagroup_df)
    cf.logger.info('make_bag_of_words(top_n_bodies)')
    (vocab, dist) = nl.make_bag_of_words(top_n_bodies)
    nl.sort_save_vocab('.vocab.hiscore', vocab, dist, a_fname, TMPDIR)

    cf.logger.info(
        "NLP Step 6. Find most freq words for low-score Answers, "
        "if program started in debug mode.")
    if args['debug']:
        # Keep these data to compare w/ words for top-scoring Answers; s/b some diff.
        # If they are identical, there may be a logic problem in the code,
        # or the data set may be too small.
        score_bot_n_df = score_df[['Id']]
        cf.logger.debug("score_bot_n_df.head():")
        cf.logger.debug(score_bot_n_df.head(20))
        top = False
        bot_n_bodies = nl.find_freq_words(
            top, score_top_n_df, num_selected_recs, progress_msg_factor, qagroup_df)
        cf.logger.info('make_bag_of_words(bot_n_bodies)')
        (vocab, dist) = nl.make_bag_of_words(bot_n_bodies)
        nl.sort_save_vocab('.vocab.loscore', vocab, dist, a_fname, TMPDIR)

    cf.logger.info("NLP Step 7. Search lo-score A's for hi-score text.")
    qa_with_hst_df = nl.search_for_terms(
        words_sorted_by_count_main_l,
        clean_ans_bodies_l,
        num_hi_score_terms, qagroup_df)
    popular_qa_df = pd.concat(
        [popular_qa_df, qa_with_hst_df]).reset_index(drop=True)
    return popular_qa_df


# TBD.1 , should select_keyword_recs()
# be called before combine_related*()?
# Use it to make the final pop_and_top_l?


def select_keyword_recs(keyword, qa_df, columns_l):
    """Find the Q's & A's from the filtered dataframe that contain the keyword,
    in Title or Body.
    Combine the sets into one set of unique Q's w/ their A's.
    Save all the selected data for analysis.
    TBD, In future, search entire collection of Q&A, not just
    the filtered subset.
    """
    # Get a pandas series of booleans to find the current question id.
    # Check Question & Answer, both Title and Body columns.
    qt_sr = qa_df.Title.str.contains(keyword, regex=False)
    qab_sr = qa_df.Body.str.contains(keyword, regex=False)
    # Combine two series into one w/ boolean OR.
    qa_contains_sr = qab_sr | qt_sr
    qak_df = qa_df[columns_l][qa_contains_sr]
    return qak_df


def show_menu(qa_df, all_ans_df, owner_reputation_df ):
    """Show prompt to user; get and handle their request.
    """
    user_menu = """    The menu choices:
    drm: calculate reputation matrix of owners
    dsm: draw and plot q&a statistics matrix
    d: draw default plot of current data
    dh: draw default histogram plot of current data
    dm: draw scatter matrix plot of current data
    h, ?: show help text, the menu
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
    # TBD print("Scroll up to read current question and answer.")
    cmd_prompt = "Enter a command: q[uit] [m]enu  ...  [h]elp: "
    while user_cmd == "":  # Repeat the request if only the Enter key is pressed.
        user_cmd = input(cmd_prompt)
    print("User entered this cmd: ", user_cmd)

    # Loop to handle user request.
    while user_cmd:
        # D print("#D while-loop: User entered this cmd: ", user_cmd)
        if user_cmd.lower() == 'm':
            print(user_menu)
            user_cmd = ''
        elif user_cmd.lower() == 'q':
            print("Save data and Quit the program.")
            # TBD, show summary & quit?
            # TBD outfile.flush()
            # TBD, save more data?
            log_msg = cf.log_file + ' - Quit by user request; Finish logging for ' + \
                os.path.basename(__file__) + '\n'
            cf.logger.warning(log_msg)
            raise SystemExit()
        elif user_cmd == '?' or user_cmd == 'h':
            print(user_menu)
            # TBD print some detailed help?
            # Clear user_cmd here to avoid infinite repetition of while loop.
            user_cmd = ''
        elif user_cmd.lower() == 's':
            # TBD print('\n### Next ###################\n')
            # TBD user_cmd = show_current_q_a(q_id, q_title, q_body, row)
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
                print("Drawing the default plot, with Score and hstCount.")
                draw_scatter_plot(
                    popular_qa_df,
                    'Score',
                    'hstCount',
                    'Score',
                    'hstCount')
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
            owner_reputation_df = check_owner_reputation(all_ans_df, owner_reputation_df )
            #
            if owner_reputation_df.empty:
                print("WARN: owner reputation dataframe empty or not found.")
            else:
                print("NOTE: Drawing the owner reputation scatter matrix plot.")
                draw_scatter_matrix_plot(owner_reputation_df[['MeanScore', 'OwnerUserId']])
        # dsm: Draw q&a statistics matrix
        elif user_cmd.lower() == 'dsm':
            user_cmd = ''
            owner_reputation_df = check_owner_reputation(all_ans_df, owner_reputation_df )
            #
            qa_stats_df = build_stats(popular_qa_df, owner_reputation_df)
            #
            if qa_stats_df.empty:
                print("WARN: qa_stats_df empty or not found.")
            else:
                print("NOTE: Drawing the qa_stats_df scatter matrix plot.")
                draw_scatter_matrix_plot(qa_stats_df[['Score', 'BodyLength', 'OwnerRep',  'hstCount' ]])
        else:
            print("Got bad cmd from user: ", user_cmd)
            print(user_menu)
            # Clear user_cmd here to avoid infinite repetition of while loop.
            user_cmd = ''
        # Show prompt & wait for a cmd.
        print("======================\n")
        while user_cmd == "":  # Repeat the request if only the Enter key is pressed.
            user_cmd = input(cmd_prompt)
    print("#D End of the cmd interpretation loop; return.")
    print()
    # D print("#D Last stmt of show_menu(); return.\n")

    # Save df before exit, if quit cmd is not used.
    # TBD Same code used for 'q' cmd; refactor both.
    # TBD print("Save data and Quit the program.")
    # Save only the needed fields to the file.
    # TBD columns_l = ['Id', 'ParentId', 'Grade', 'Notes', 'Title', 'Body']
    # TBD outfile = open(DATADIR + 'graded_popular_qa.csv', 'w')
    # TBD graded_df[columns_l].to_csv(outfile, header=True, index=None, sep=',', mode='w')
    # TBD outfile.flush()

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
    qa_stats_df = qa_df[['Id','OwnerUserId','ParentId','Score','hstCount']]
    # Add new column to df & initlz it.
    qa_stats_df = qa_stats_df.assign(BodyLength = qa_stats_df.Id)

    #TBD.1.Sun2017_0806_14:16  Chg qa_df to qa_stats_df, unless other columns needed:
    for index, row in qa_df.iterrows():
        ouid = row['OwnerUserId']
        #D print("#D qa_stats_df index, ouid: ", index, ouid )
        try:
            owner_rep = or_df.loc[or_df['OwnerUserId'] == ouid, 'MeanScore'].iloc[0]
            #D print("#D qa_stats_df index, owner_rep: ", index, owner_rep )
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
    qa_stats_df = qa_stats_df[['Id', 'ParentId', 'OwnerUserId', 'Score', 'BodyLength', 'OwnerRep', 'hstCount']]

    own_rep_file = DATADIR + 'qa_stats.csv'
    save_prior_file('', own_rep_file)
    qa_stats_df.to_csv(own_rep_file)

    own_rep_file = DATADIR + 'qa_stats.html'
    save_prior_file('', own_rep_file)
    qa_stats_df.to_html(own_rep_file)

    return qa_stats_df


def check_owner_reputation(all_ans_df, owner_reputation_df ):
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
    # TBD, These bins used for debugging; replace.
    histo_bins = [-10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                  13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45, 50]
    plot_df.plot.hist(ax=ax, figsize=(6, 6), bins=histo_bins)
    plt.show(block=False)

    # Write data set to a csv file.
    outfile = TMPDIR + 'tmp_plot.csv'
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
    pd.set_option('display.max_colwidth', -1) # -1=no limit, for debug
    save_prior_file(wdir, wfile)
    outfile = wdir + wfile 
    in_df.to_csv(outfile)
    pd.set_option('display.max_colwidth', MAXCOLWID) # -1=no limit, for debug
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

    Use the list of columns included in this function.
    """
    pd.set_option('display.max_colwidth', -1) # -1=no limit, for debug
    outfile = wdir + wfile
    save_prior_file(wdir, wfile)
    # Specify default output fields to use.
    if not columns_l:
        columns_l = ['Id',
                       'Title',
                       'Body',
                       'Score',
                       'hstCount',
                       'HiScoreTerms',
                       'OwnerUserId',
                       'ParentId']
    #
    # Save o/p to a string and do not specify an output file in
    # calling to_html().
    # Use 'escape=False' to render HTML when outfile is opened in a browser.
    in_s = in_df[columns_l].to_html(escape=False)

    # Clean the newlines in the string
    # so the HTML inside each table cell renders properly on screen.
    in_s = replace_line_breaks(in_s)

    # Concatenate css w/ html file to format the o/p better.
    with open(outfile, 'w') as f:
        print('\nNOTE: Writing data to html outfile: ' + outfile)
        f.write(HEADER)
        f.write(in_s)
        f.write(FOOTER)

    pd.set_option('display.max_colwidth', MAXCOLWID) # -1=no limit, for debug
    return


#TBD, Sun2017_0903_17:00 , not used.
def write_df_to_file(in_df, wdir, wfile):
    """Write one column of a pandas data frame to a file w/ suffix '.qanda'.
    """
    # Used for testing and debugging.
    outfile = wdir + wfile + '.qanda'
    if os.path.exists(outfile):
        os.rename(outfile, outfile + '.bak2')
        print(
            '\nWARN: renamed o/p file to *.bak2; save it manually if needed:' +
            outfile)
    with open(outfile, 'w') as f:
        print('\nNOTE: Writing data to outfile: ' + outfile)
        print(in_df['Body'], file=f)
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

    """
    parser.add_argument(
        '-p',
        '--popular_questions',
        help='select questions with many answers',
        action='store_true')
    parser.add_argument(
        '-t',
        '--top_users',
        help='find lo-score answers by hi-scoring owners',
        action='store_true')
    """
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser


if __name__ == '__main__':

    parser = get_parser()
    args = vars(parser.parse_args())

    # Set the number of top scoring owners to select from the data.
    num_owners = 10  # Default is 10.
    num_owners = 40  # Default is 10.
    # D num_owners = 100  # Default is 10.
    print("num_owners: ", num_owners)

    keyword = False
    # D keyword = 'beginner'
    # D keyword = 'yield'
    # D keyword = 'begin'
    # D keyword = 'pandas'
    # D keyword = 'Python'  # Both Title & Body of data sets have it; for debug
    print("Keyword: ", keyword)

    num_hi_score_terms = 21  # Use 3 for testing; 11 or more for use.
    print("num_hi_score_terms: ", num_hi_score_terms)

    main(popular_qa_df)

    owner_reputation_df = pd.DataFrame()
    show_menu(popular_qa_df, all_ans_df, owner_reputation_df )

    log_msg = cf.log_file + ' - Finish logging for ' + \
        os.path.basename(__file__) + '\n'
    cf.logger.warning(log_msg)

'bye'
