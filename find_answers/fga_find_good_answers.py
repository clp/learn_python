# -*- coding: utf-8 -*-

# Using ~/anaconda3/bin/python: Python 3.5.2 :: Anaconda 4.2.0 (64-bit)
# Using Python 3.4.5 :: Anaconda 4.3.0 (64-bit), since Tue2017_0710

#   Time-stamp: <Thu 2017 Jul 13 03:28:58 PMPM clpoda>
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

version = '0.0.5'

import argparse
import nltk
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np
import os
import pandas as pd
import random
from pandas.tools.plotting import scatter_matrix

import config as cf
import nltk_ex25 as nl


log_msg = cf.log_file + ' - Start logging for ' + os.path.basename(__file__)
cf.logger.info(log_msg)

q_with_a_df = pd.DataFrame()
all_ans_with_hst_df = pd.DataFrame()
owner_grouped_df = pd.DataFrame()


def main(q_with_a_df):
    """Analyze input data and produce o/p by calling functions.
    """
    init()

    cf.a_fname, a_infile, q_infile, indir, outdir = \
        config_data()

    cf.all_ans_df, cf.all_ques_df, cf.progress_msg_factor, numlines = \
        read_data( a_infile, q_infile)

    popular_ids = \
        find_popular_ques(cf.all_ans_df, cf.a_fname)
    popular_ids_a = popular_ids.index.values

    top_scoring_owners_a, owner_grouped_df = \
        group_data(cf.all_ans_df)
    parent_id_l = \
        find_question_ids(top_scoring_owners_a, cf.all_ans_df)

    pop_and_top_l = \
        select_questions(parent_id_l, popular_ids_a)

    q_with_a_df, all_ans_with_hst_df = \
        combine_related_q_and_a(
            pop_and_top_l, cf.all_ques_df, cf.all_ans_df, numlines)

    # TBD Save the df to a file for review & debug; later processing may
    # use the df & the file is not needed.
    outfile = "tmpdir/all_ans_with_hst.csv"
    all_ans_with_hst_df.to_csv(outfile)
    outfile = "tmpdir/all_ans_with_hst.html"
    save_prior_file('', outfile)
    all_ans_with_hst_df[['Id',
                         'Title',
                         'Score',
                         'hstCount',
                         'HiScoreTerms',
                         'OwnerUserId',
                         'ParentId']].to_html(outfile)

    # Write full data set to a csv file.
    outfields_l = [
        'Id',
        'ParentId',
        'OwnerUserId',
        'CreationDate',
        'Score',
        'Title',
        'Body']
    outfile = 'outdir/q_with_a.csv'
    q_with_a_df[outfields_l].to_csv(
        outfile, header=True, index=None, sep=',', mode='w')
    # DBG  write_df_to_file(q_with_a_df, outdir, cf.a_fname)

    if keyword:
        # Write keyword-containing records to a csv file.
        qa_with_keyword_df = select_keyword_recs(
            keyword, q_with_a_df, outfields_l)
        outfile = 'outdir/qa_with_keyword.csv'
        qa_with_keyword_df[outfields_l].to_csv(
            outfile, header=True, index=None, sep=',', mode='w')


def init():
    """Initialize some settings for the program.
    """
    if args['debug']:
        cf.end = 55
        print('Running in debug mode.')
        print('  cf.end set to: ', cf.end)
        print()

    # Initialize settings for pandas.
    pd.set_option('display.width', 0)  # 0=no limit, for debug
    pd.set_option('display.max_colwidth', 100) # -1=no limit, for debug

    # Don't show commas in large numbers.
    # Show OwnerUserId w/o '.0' suffix.
    pd.options.display.float_format = '{:.0f}'.format


def show_menu(qa_df):
    """Show prompt to user; get and handle their request.
    """
    user_menu = """    The menu choices:
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
                print("Warn: dataframe empty or not found; try restarting.")
            else:
                # D print('Show current Q&A at this saved_index: ',
                # saved_index)
                print(qa_df[['Id', 'Title', 'Body']].iloc[[saved_index]])
        elif user_cmd.lower() == 'sn':  # show next item
            user_cmd = ''
            if qa_df.empty:
                print("Warn: dataframe empty or not found; try restarting.")
            else:
                saved_index += 1
                # D print('#D Show current Q&A at this saved_index: ',
                # saved_index)
                print(qa_df[['Id', 'Title', 'Body']].iloc[[saved_index]])
        elif user_cmd.lower() == 'sp':  # show prior item
            user_cmd = ''
            if qa_df.empty:
                print("Warn: dataframe empty or not found; try restarting.")
            else:
                saved_index -= 1
                # D print('#D Show current Q&A at this saved_index: ',
                # saved_index)
                print(qa_df[['Id', 'Title', 'Body']].iloc[[saved_index]])
        elif user_cmd.lower() == 'd':
            user_cmd = ''
            if all_ans_with_hst_df.empty:
                print("Warn: dataframe empty or not found; try restarting.")
            else:
                print("Drawing the default plot.")
                draw_scatter_plot(
                    all_ans_with_hst_df,
                    'Score',
                    'hstCount',
                    'Score',
                    'hstCount')
        elif user_cmd.lower() == 'dh':
            user_cmd = ''
            if all_ans_with_hst_df.empty:
                print("Warn: dataframe empty or not found; try restarting.")
            else:
                print("Drawing the default histogram plot.")
                draw_histogram_plot(all_ans_with_hst_df)
        elif user_cmd.lower() == 'dm':  # Scatter matrix plot
            user_cmd = ''
            if all_ans_with_hst_df.empty:
                print("Warn: dataframe empty or not found; try restarting.")
            else:
                print("Drawing the default scatter matrix plot.")
                draw_scatter_matrix_plot(all_ans_with_hst_df)
        # rma: Reputation, mean, answers only; scatter.
        elif user_cmd.lower() == 'dr':
            user_cmd = ''
            # TBD.1 use right df in both branches:
            if qa_df.empty:
                print("Warn: dataframe empty or not found; try restarting.")
            else:
                print("TBD, Drawing the default reputation scatter plot. NOT READY.\n")
                # TBD Prepare data to plot: owner reputation (mean or total score), answer score.
                # TBD, draw_scatter_plot(owner_grouped_df, xaxis, yaxis, xname, yname)
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
    # TBD outfields_l = ['Id', 'ParentId', 'Grade', 'Notes', 'Title', 'Body']
    # TBD outfile = open('outdir/graded_q_with_a.csv', 'w')
    # TBD graded_df[outfields_l].to_csv(outfile, header=True, index=None, sep=',', mode='w')
    # TBD outfile.flush()

    return


def config_data():
    """Configure path and file names for i/o data.
    """
    # TBD Make the in & out dirs w/ this program, if they don't exist?
    # TBD Include the test data files w/ this project.
    indir = 'indir/'  # Relative to pwd, holds i/p files.
    outdir = 'outdir/'  # Relative to pwd, holds o/p files.
    # D cf.a_fname = 'Answers.csv'
    # D cf.q_fname = 'Questions.csv'

    # Smaller data sets, used for debugging.
    # D cf.q_fname = 'q6_999994.csv'
    # D cf.a_fname = 'a6_999999.csv'
    # D cf.a_fname = 'a5_99998.csv'
    # D cf.q_fname = 'q30_99993.csv'
    cf.a_fname = 'a3_986.csv'
    cf.q_fname = 'q3_992.csv'
    # D cf.a_fname = 'a2.csv'
    # D cf.q_fname = 'q2.csv'

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
    # TBD, Sat2017_0506_15:34  Maybe rm latin-1 encoding here also?
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
    cf.progress_msg_factor = int(round(numlines / 10))
    print('\n#D  cf.progress_msg_factor : ', cf.progress_msg_factor)
    print()
    return ans_df, ques_df, cf.progress_msg_factor, numlines


def find_popular_ques(aa_df, a_fname):
    """Find the most frequent ParentIds in the answers df.
    """
    popular_ids = pd.value_counts(aa_df['ParentId'])
    outfile = "outdir/fpq_popular_ids." + a_fname + ".csv"
    popular_ids.to_csv(outfile)
    return popular_ids


def group_data(aa_df):
    """Group the contents of the answers df by a specific column.
    Group by OwnerUserId, and sort by mean score for answers only
    for each owner (question scores are not counted).
    Make a numpy array of owners w/ highest mean scores.
    TBD.1, Find low score answers for these hi-score owners;
    then mark the low score  answers for evaluation.
    Low score is any score below lo_score_limit.
    """
    print('=== owner_grouped_df: Group by owner and sort by mean score for each owner.')
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
    outfile = 'outdir/lo_scores_for_top_users.csv'
    lo_scores_for_top_users_df.to_csv(
        outfile, header=True, index=None, sep=',', mode='w')
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
    With slice limits at 40 and 30, got 2 Q, 36 A.
    With slice limits at 400 and 300, got 26 Q, 308 A.
    TBD, what data set was used?
    """
    pop_and_top_l = list(
        set(parent_id_l[:500]).intersection(set(popular_ids_a[:500])))
    print('len(pop_and_top_l) : ', len(pop_and_top_l))
    if args['verbose']:
        print('pop_and_top_l, parent id\'s to examine: ', pop_and_top_l[:])
    return pop_and_top_l


def combine_related_q_and_a(pop_and_top_l, all_ques_df, aa_df, numlines):
    """Get each Q in the list of ParentId's, and the related A's.
    Loop over all the question Id's and store all Q & A data in a df.

    Then call analyze_text() on each group of Q with A's,
    which calls the routines that perform the natural language processing
    of the text data.
    """
    global q_with_a_df
    global all_ans_with_hst_df

    ques_match_df = all_ques_df[all_ques_df['Id'].isin(pop_and_top_l)]
    ans_match_df = aa_df[aa_df['ParentId'].isin(pop_and_top_l)]
    q_with_a_df = pd.concat(
        [ques_match_df, ans_match_df]).reset_index(drop=True)
    # Full list w/ all Q's at top, A's after.

    print('#D len of ques_match_df: ', len(ques_match_df))
    print('#D len of ans_match_df: ', len(ans_match_df))
    print('\n#D ques_match_df.head() & ans_match_df: ')
    print(ques_match_df.head())
    print('#D')
    print(ans_match_df.head())

    i = 0
    # Build each Q&A group: one Q w/ all its A.'s
    # TBD, How to do this w/o explicit loop, using df tools?
    # TBD, Combine or replace the ques_match_df & ans*df code above w/ this?
    # Maybe delete those 'intermediate' results?
    for qid in pop_and_top_l:
        i += 1
        # OK if(i % cf.progress_msg_factor == 0):
        if(i % 20 == 0):
            print("#D combine_related_q_and_a:for-qid-loop count i: ", i)
        qm_df = ques_match_df[ques_match_df['Id'] == qid]
        am_df = ans_match_df[ans_match_df['ParentId'] == qid]
        qag_df = pd.concat([qm_df, am_df]).reset_index(drop=True)
        print("\n#D qag_df.head(): ")
        print(qag_df.head())
        cf.logger.info('qag_df.head(1): ')
        cf.logger.info(qag_df.head(1))

        all_ans_with_hst_df = analyze_text(qag_df, numlines)

        #
        # TBD.Thu2017_0608_23:58
        # Analyze qag_df w/ nlp s/w in this loop;
        # or save each qag_df to a separate df for later processing;
        # or save each qag_df to a disk file for later processing;
        # or move this code to the nlp processing section of the program.
        #

    # D print('\n#D fga, End of debug code; exiting.')
    # D raise SystemExit()

    return q_with_a_df, all_ans_with_hst_df


def analyze_text(qag_df, numlines):
    """Use a Q&A group of one Q w/ its A's for i/p.
    Process the text data w/ the routines in the nltk module, which use
    natural language tools.
    """
    global all_ans_with_hst_df

    # TBD.1 Assign the global var cf.all_ans_df here.  Find a better soln w/o
    # global.  cf.all_ans_df is used in the nltk module.
    #TBD.1 cf.all_ans_df = qag_df  # TMP to avoid renaming all_ans_df in many places
    # TBD.1.Thu2017_0713_15:08 :
    #   Fix: Chg global cf.all_ans_df to local qag_df wherever it is needed;
    #   Add it to func arg list where needed.
    #   These will be chgs in nltk program; Don't chg earlier uses of cf.all_ans_df in fga.
    #   Then chg global cf.all*df to local all_ans_df.
    #
    cf.logger.info("NLP Step 2. Process the words of each input line.")
    clean_ans_bodies_l = nl.clean_raw_data(
        cf.a_fname, cf.progress_msg_factor, qag_df)
    print('\n#D, clean_ans_bodies_l[:1]')
    print(clean_ans_bodies_l[:1])

    cf.logger.info("NLP Step 3. Build a bag of words and their counts.")
    (vocab, dist) = nl.make_bag_of_words(clean_ans_bodies_l)
    print('\n#D, vocab[:1]')
    print(vocab[:1])
    words_sorted_by_count_l = nl.sort_save_vocab(
        '.vocab', vocab, dist, cf.a_fname)
    # Save the original list for later searching.
    words_sorted_by_count_main_l = words_sorted_by_count_l

    cf.logger.info('NLP Step 4. Sort Answers by Score.')
    score_df, num_selected_recs = nl.sort_answers_by_score(numlines, qag_df)

    cf.logger.info('NLP Step 5. Find most freq words for top-scoring Answers.')
    score_top_n_df = score_df[['Id']]

    # TBD, Maybe convert df to string so logger can print title & data w/ one cmd:
    # TBD log_msg = "score_top_n_df.tail():" + CONVERT_DF_TO_STRING(score_top_n_df.tail())
    # TBD cf.logger.debug(log_msg)
    cf.logger.debug("score_top_n_df.tail():")
    cf.logger.debug(score_top_n_df.tail(20))
    # Use top_n Answers & count their words.
    cf.logger.info(
        "For top ans: Cleaning and parsing the training set bodies...")

    top = True
    top_n_bodies = nl.find_freq_words(
        top, score_top_n_df, num_selected_recs, cf.progress_msg_factor, qag_df)
    cf.logger.info('make_bag_of_words(top_n_bodies)')
    (vocab, dist) = nl.make_bag_of_words(top_n_bodies)
    nl.sort_save_vocab('.vocab.hiscore', vocab, dist, cf.a_fname)

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
            top, score_top_n_df, num_selected_recs, cf.progress_msg_factor, qag_df)
        cf.logger.info('make_bag_of_words(bot_n_bodies)')
        (vocab, dist) = nl.make_bag_of_words(bot_n_bodies)
        nl.sort_save_vocab('.vocab.loscore', vocab, dist, cf.a_fname)

    cf.logger.info("NLP Step 7. Search lo-score A's for hi-score text.")
    ans_with_hst_df = nl.search_for_terms(
        words_sorted_by_count_main_l,
        clean_ans_bodies_l,
        num_hi_score_terms, qag_df)
    all_ans_with_hst_df = pd.concat(
        [all_ans_with_hst_df, ans_with_hst_df]).reset_index(drop=True)
    return all_ans_with_hst_df


# TBD.1 Sat2017_0211_22:55 , should select_keyword_recs()
# be called before combine_related*()?
# Use it to make the final pop_and_top_l?


def select_keyword_recs(keyword, qa_df, outfields_l):
    """Find the Q's & A's from the filtered df that contain the keyword,
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
    qak_df = qa_df[outfields_l][qa_contains_sr]
    return qak_df


def draw_histogram_plot(plot_df):
    """Draw a simple histogram plot using pandas tools.
    """
    fig, ax = plt.subplots(1, 1)
    ax.get_xaxis().set_visible(True)
    plot_df = plot_df[['Score']]
    histo_bins = [-10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                  13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45, 50]
    plot_df.plot.hist(ax=ax, figsize=(6, 6), bins=histo_bins)
    plt.show(block=False)

    # Write data set to a csv file.
    outfile = 'tmpdir/tmp_plot.csv'
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


def draw_scatter_matrix_plot(plot_df):
    """Draw a set of scatter plots showing each feature vs
    every other feature.
    """
    axs = scatter_matrix(plot_df, alpha=0.2, diagonal='hist')
    plt.show(block=False)
    wdir = 'outdir/'
    wfile = 'scat_mat_plot.pdf'
    save_prior_file(wdir, wfile)
    plt.savefig(wdir + wfile)


def draw_scatter_plot(plot_df, xaxis, yaxis, xname, yname):
    """Draw a simple scatter plot using pandas tools.
    """
    ax = plot_df[[yname, xname]].plot.scatter(x=xaxis, y=yaxis, table=False)
    plt.show(block=False)


def save_prior_file(wdir, wfile):
    """Save backup copy of a file w/ same name and '.bak' extension.
    """
    outfile = wdir + wfile
    if os.path.exists(outfile):
        os.rename(outfile, outfile + '.bak')
        print(
            '\nWARN: renamed o/p file to *.bak; save it manually if needed:' +
            outfile)


def write_df_to_file(in_df, wdir, wfile):
    """Write one column of a pandas data frame to a file w/ suffix '.qanda'.
    """
    # Used for testing and debugging.
    outfile = wdir + wfile + '.qanda'
    if os.path.exists(outfile):
        os.rename(outfile, outfile + '.bak')
        print(
            '\nWARN: renamed o/p file to *.bak; save it manually if needed:' +
            outfile)
    with open(outfile, 'w') as f:
        print('\nWriting Q and A to outfile: ' + outfile)
        print(in_df['Body'], file=f)


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
    num_owners = 100  # Default is 10.
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

    main(q_with_a_df)

    show_menu(q_with_a_df)

    log_msg = cf.log_file + ' - Finish logging for ' + \
        os.path.basename(__file__) + '\n\n'
    cf.logger.warning(log_msg)

'bye'
