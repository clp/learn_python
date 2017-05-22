# -*- coding: utf-8 -*-

# Using ~/anaconda3/bin/python: Python 3.5.2 :: Anaconda 4.2.0 (64-bit)

#   Time-stamp: <Mon 2017 May 22 11:24:24 AMAM clpoda>
"""grade_each_answer.py


   Show the user a question and an answer from the data files;
   ask user to grade the answer.
   That grade will be compared to the grade for that answer
   from the NLP s/w, to measure the quality of the NLP s/w.
   The NLP s/w is currently implemented in fga_find_good_answers.py.

   Usage:
     pydoc  grade_each_answer
     python grade_each_answer.py


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

TBD,Mon2017_0522_10:47 , add Grade and Notes fields.

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
# Manually review the A's.
#   Mark the answer w/ a grade tag: Is it worth reading?
#     Use A,B,C,D,F to identify excellent(A), good, fair, poor, useless(F) answers.
# Build tools to analyze a larger test set.
# ----------------------------------------------------------


version = '0.0.1'

import argparse
#TBR import nltk
#TBR import numpy as np
import os
import pandas as pd
#TBR import random
from shutil import copyfile


# ----------------------------------------------------------
# Specify question and parent ID's to find.
#DBG pid_l = [469, 535, 231767]
#DBG.many pid_l = [469, 502, 535, 594, 683, 742, 766, 773, 972, 1476, 766, 1734, 1829, 1854, 1983, 2311, 2933, 3061, 3976, 4942, 5102, 5313, 1983, 5909, 5966, 8692, 8948, 10123, 11060, 2933, 1983]
pid_l = [469, 502, 535, 594, 683, 742, 766, 773, 972]

# ----------------------------------------------------------

def main():
    init()
    a_fname, a_infile, q_infile, indir, outdir = config_data()
    all_ans_df, all_ques_df, progress_msg_factor = read_data(a_infile, q_infile)
    if args['user_eval']:
        print('Open the data frame for user to evaluate some answers.\n')
        read_and_grade_answers(all_ans_df, all_ques_df)
        exit
    #TBR? popular_ids = find_popular_ques(all_ans_df, a_fname)
    #TBR? popular_ids_a = popular_ids.index.values
    #TBR? top_scoring_owners_a = group_data(all_ans_df)
    #TBR? parent_id_l = find_question_ids(top_scoring_owners_a, all_ans_df)
    #TBR? #
    #TBR? # pop_and_top_l:
    #TBR? # Find parent IDs that are popular (have several answers) and
    #TBR? # some of those answers come from top owners (owners have high scores).
    #TBR? #TBD, Set the size of these lists to get enough o/p to analyze.
    #TBR? pop_and_top_l = list(set(parent_id_l[:20]).intersection(set(popular_ids_a[:40])))
    #TBR? if args['verbose']:
        #TBR? print('len(pop_and_top_l) : ', len(pop_and_top_l))
        #TBR? print('pop_and_top_l, parent id\'s to examine: ', pop_and_top_l[:])
    #TBR? q_with_a_df = combine_related_q_and_a(pop_and_top_l, all_ques_df, all_ans_df)
    #TBR? qa_with_keyword_df = select_keyword_recs(keyword, pop_and_top_l, q_with_a_df, all_ques_df, all_ans_df)
#TBR? 
    #TBR? # Write qa_with_keyword_df, a subset of the full data set, to a csv file.
    #TBR? outfields_l = ['Id', 'ParentId', 'OwnerUserId', 'CreationDate', 'Score', 'Title', 'Body']
    #TBR? outfile = 'outdir/qa_with_keyword.csv'
    #TBR? qa_with_keyword_df[outfields_l].to_csv(outfile, header=True, index=None, sep=',', mode='w')
#TBR? 
    #TBR? # Write full data set to a csv file.
    #TBR? outfile = 'outdir/q_with_a.csv'
    #TBR? q_with_a_df[outfields_l].to_csv(outfile, header=True, index=None, sep=',', mode='w')
    #TBR? # DBG  write_df_to_file(q_with_a_df, outdir, a_fname)


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
    #D a_fname = 'Answers.csv'
    #D q_fname = 'Questions.csv'

    # Smaller data sets, used for debugging.
    q_fname = 'q6_999994.csv'
    a_fname = 'a6_999999.csv'
    # D a_fname = 'a5_99998.csv'
    # D q_fname = 'q30_99993.csv'
    #D a_fname = 'a3_986.csv'
    #D q_fname = 'q3_992.csv'
    # D a_fname = 'a2.csv'
    # D q_fname = 'q2.csv'

    a_infile = indir + a_fname
    q_infile = indir + q_fname

    print('Input files, q & a:\n' + q_infile + '\n' + a_infile)
    print()

    return a_fname, a_infile, q_infile, indir, outdir


def read_data(ans_file, ques_file):
    """Read the csv i/p files and store into a pandas data frame.
    Compute a factor that dictates how progress will be indicated
    during read operations.
    """
    #TBD, Sat2017_0506_15:34  Maybe rm latin-1 encoding here also?
    ans_df = pd.read_csv(ans_file, encoding='latin-1', warn_bad_lines=False, error_bad_lines=False)
    ques_df = pd.read_csv(ques_file, encoding='latin-1', warn_bad_lines=False, error_bad_lines=False)

    numlines = len(ans_df)
    print('Number of answer records in i/p data frame, ans_df: ' + str(numlines))
    progress_msg_factor = int(round(numlines/10))
    print()
    return ans_df, ques_df, progress_msg_factor


def read_and_grade_answers(all_ans_df, all_ques_df):
    """Ask user to enter a quality grade for each answer.
    
    Read the data file that holds the user grades 
    for each answer.
    If that file does not exist, create it, and begin
    to add a grade for each answer by following these steps.
    
    Print a question and one un-graded answer to the console.
    
    Prompt the user to evaluate the answer and grade it,
    and to write any additional comments.
    Save those data into a data frame,
    and include the answer_id field with each record.
    Repeat these steps for each un-evaluated answer,
    until the user terminates the action, or until all
    answers have been graded.
    
    Write that df to a csv file.
    """
    
    qa_master_file = 'outdir/q_with_a.csv'
    # Handle case of the master Q&A data file is missing.
    if not os.path.exists(qa_master_file):
        print('WARN: Q&A master file not found: ', qa_master_file)
        print('  Verify i/p files exist.')
        print('  Then create master file by running fga*.py w/o "-u" option.')
        exit()
    #
    ungr_file = 'outdir/ungraded_answers.csv'
    if not os.path.exists(ungr_file):
        # If ungraded_answers.csv file is missing, copy from a master file with no graded answers.
        #TBD Initialize the grade file only if it does not exist, ie,
        # the first time this function runs, or if the file
        # has been lost or deleted.
        print('\nWARN: file not found, creating it by copying from q_with_a.csv:' + ungr_file + '\n')
        copyfile('outdir/q_with_a.csv', ungr_file)
        #TBD Removing latin-1 in 2 read_csv() calls  fixed a problem:
            #ORG ungraded_answers_df = pd.read_csv(ungr_file, encoding='latin-1', warn_bad_lines=False, error_bad_lines=False, usecols=['Id', 'ParentId', 'Title', 'Body'])
        ungraded_answers_df = pd.read_csv(ungr_file,  warn_bad_lines=False, error_bad_lines=False, usecols=['Id', 'ParentId', 'Title', 'Body'])

        #TBD Create & initlz new columns only the first time the file is read.
        ungraded_answers_df['Grade'] = 'N'
        ungraded_answers_df['Notes'] = 'None'
        outfields_l = ['Id', 'ParentId', 'Grade', 'Notes', 'Title', 'Body']
        ungraded_answers_df[outfields_l].to_csv(ungr_file, header=True, index=None, sep=',', mode='w')

    # Otherwise:
    # The data file exists; read it; it might have some graded answers.
    # Removing latin-1 in 2 read_csv() calls  fixed a problem:
    #ORG.Sat2017_0506_14:28   ungraded_answers_df = pd.read_csv(ungr_file, encoding='latin-1', warn_bad_lines=False, error_bad_lines=False)
    #
    ungraded_answers_df = pd.read_csv(ungr_file, warn_bad_lines=False, error_bad_lines=False)
    
    user_menu = """    The menu choices to grade an answer:
    a: excellent value
    b: good value
    c: fair value
    d: poor value
    f: no value
    i: ignore this item for now; leave its grade 'N' for none
    u: unknown value; skip it for now, evaluate it later
    .........................................................

    Other menu items:
    h, ?: show help text, the menu
    m: show menu
    q: save data and quit the program
    TBD s: show next answer
    """
    user_cmd = ''

    # Find first ungraded answer; show Q&A; ask user for input.
    print('Body text of some ungraded_answers_df records that have not been graded:\n')
    for index, row in  ungraded_answers_df.iterrows():
        if row['Grade'] == 'N':
            # Show Q then A then ask user to grade the A.
            if not pd.isnull(row['Title']):  # Found a question.
                print("\n##############################\n#D Found a question.")
                q_id = row['Id']
                q_title = row['Title']
                q_body = row['Body']
                print("Q.Title:\n", q_title)
                continue  # Finished w/ Q; jump to next item in for-loop & look for next A.
            elif pd.isnull(row['Title']): # Found an answer.
                print("#D Found an answer.")
                user_cmd = show_current_q_a(q_id, q_title, q_body, row)
            else: # Found a problem
                print("ERR. Found an unusual record; this branch should not be reached.")
                print("  Problem found at this ID:")
                print(row['Id'])
                exit()
            #
            # Loop to handle user request.
            while user_cmd:
                print("#D while-loop: User entered this cmd: ", user_cmd)
                if user_cmd.lower() == 'm':
                    print(user_menu)
                    user_cmd = ''
                elif user_cmd.lower() == 'a':  # Excellent
                    # User graded this answer as excellent value; save grade & ask for a note.
                    store_grade('A', index, ungraded_answers_df)
                    #D print("#D while-loop-h: Show next item.")
                    break  # examine next item in ungr*df for-loop
                elif user_cmd.lower() == 'b':  # Good
                    store_grade('B', index, ungraded_answers_df)
                    break
                elif user_cmd.lower() == 'c':  # Fair
                    store_grade('C', index, ungraded_answers_df)
                    break
                elif user_cmd.lower() == 'd':  # Poor
                    store_grade('D', index, ungraded_answers_df)
                    break
                elif user_cmd.lower() == 'f':  # Answer is not useful
                    store_grade('F', index, ungraded_answers_df)
                    break
                elif user_cmd.lower() == 'u':  # Unknown; no opinion.
                    # Unknown items will not be shown in normal user eval mode.
                    store_grade('U', index, ungraded_answers_df)
                    break
                elif user_cmd.lower() == 'i':  # Ignore for now; leave grade=N,
                    # Ignored items will be shown in normal user evaluation mode.
                    break
                elif user_cmd.lower() == 'q':
                    print("Save data and Quit the program.")
                    # Save only the needed fields to the file.
                    outfields_l = ['Id', 'ParentId', 'Grade', 'Notes', 'Title', 'Body']
                    outfile = open('outdir/ungraded_answers.csv', 'w')
                    ungraded_answers_df[outfields_l].to_csv(outfile, header=True, index=None, sep=',', mode='w')
                    outfile.flush()
                    #
                    exit()
                elif user_cmd == '?' or user_cmd == 'h':
                    print(user_menu)
                    #TBD print some detailed help?
                    # Clear user_cmd here to avoid infinite repetition of while loop.
                    user_cmd = ''
                elif user_cmd.lower() == 's':
                    user_cmd = show_current_q_a(q_id, q_title, q_body, row)
                else:
                    print("Got bad cmd from user: ", user_cmd)
                    print(user_menu)
                    # Clear user_cmd here to avoid infinite repetition of while loop.
                    user_cmd = ''
                # Show prompt & wait for a cmd.
                print("======================\n")
                print("Scroll up to read current question and answer.")
                cmd_prompt = "Enter a grade or command: a b c d f ... i [m]enu [h]elp: "
                while user_cmd == "":  # Repeat the request if only the Enter key is pressed.
                    user_cmd = input(cmd_prompt)
                #D print("User entered this cmd: ", user_cmd)
            print("#D Last stmt of the cmd interpretation if-clause; go to next item.")
        print("#D Last stmt of the for-loop; go to next item.")
    print()
    print("#D Last stmt of read_and_grade_answers(); return.\n")
    return


def store_grade(grade, index, df):
    # Save the grade and notes to the current record.
    note_text = input("Enter text for a note; end it with the Enter key: ")
    #ORG df.ix[index, 'Grade'] = 'h'
    df.ix[index, 'Grade'] = grade
    df.ix[index, 'Notes'] = note_text
    #TBF print("#D Current row:\n###\n", df[index])
    return 


def show_current_q_a(q_id, q_title, q_body, row):
    print("Q.Id:", q_id, "   Q.Title:\n", q_title[:55], '\n')
    print("Q.Body[:55]:\n", q_body[:55])
    print("----------------------\n")
    print("A.Id, A.Body[:55]:\n")
    print(row['Id'], "\n", row['Body'][:55])
    print("======================\n")
    print("Scroll up to read current question and answer.")
    cmd_prompt = "Enter a grade or command: a b c d f ... i [m]enu [h]elp: "
    user_cmd = ''
    while user_cmd == "":  # Repeat the request if only the Enter key is pressed.
        user_cmd = input(cmd_prompt)
    #D print("User entered this cmd: ", user_cmd)
    return user_cmd


def find_popular_ques(all_ans_df, a_fname):
    # Find the most frequent ParentIds found in the answers df.
    popular_ids = pd.value_counts(all_ans_df['ParentId'])
    #
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


def combine_related_q_and_a(pop_and_top_l, all_ques_df, all_ans_df):
    """Get each Q in the list of ParentId's, and the related A's.
    Loop over all the question Id's and store all Q & A data in a list.
    Convert the list to a df.  Return that df.
    """
    ques_ans_l = []
    for qid in pop_and_top_l:
        # Get a pandas series of booleans to find the current question id.
        ques_match_sr = (all_ques_df.Id == qid)
        ques_match_df = all_ques_df[['Id', 'OwnerUserId',  'CreationDate', 'Score', 'Title', 'Body']][ques_match_sr]
        #
        # Append current question to the list.
        ques_ans_l.append(ques_match_df)
        #
        # Get a pandas series of booleans to find all A's related to the current Q.
        ans_match_sr = (all_ans_df.ParentId == qid)
        df = all_ans_df[['Id', 'ParentId', 'OwnerUserId',  'CreationDate', 'Score', 'Body']][ans_match_sr]
        #
        # Append all related answers to the list.
        ques_ans_l.append(df)
    q_with_a_df = pd.concat(ques_ans_l)
    if args['verbose']:
        print('ques_match_sr: ')
        print(ques_match_sr)
    return q_with_a_df


# TBD.1 Sat2017_0211_22:55 , should this func be called before combine_related*()?
# Use it to make the final pop_and_top_l?
# TBD, Thu2017_0504_18:52 . pop_and_top_l is not used here, yet.
#TBD, all_ques_df, all_ans_df are not used here yet.
#
def select_keyword_recs(keyword, pop_and_top_l, q_with_a_df, all_ques_df, all_ans_df):
    """Find all Q's that contain the keyword, in Title or Body.
    TBD, Find all A's that contain the keyword; select the corresponding Q's.
    Combine the two sets into one set of unique Q's w/ their A's.
    Save all the selected data for analysis.
    """
    #
    # Get a pandas series of booleans to find the current question id.
    # Check both Question Title and Body columns.
    qt_sr = q_with_a_df.Title.str.contains(keyword, regex=False)
    qb_sr = q_with_a_df.Body.str.contains(keyword, regex=False)
    # Combine two series into one w/ boolean OR.
    ques_contains_sr = qb_sr | qt_sr
    qm_df = q_with_a_df[['Id', 'ParentId', 'OwnerUserId',  'CreationDate', 'Score', 'Title', 'Body']][ques_contains_sr]

    # TBD, Now check the Answer Body column in the same way.
    #  See b.2, for future.
    #  ab_sr = (q_with_a_df.Body.str.contains(keyword, regex=False))
    #  Find the Q's that correspond to these A's.
    #  Save those Q's & all related A's.

    return qm_df


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
    parser = argparse.ArgumentParser(description='find good answers hidden in stackoverflow data')

    parser.add_argument('-L', '--lo_score_limit', help='lowest score for an answer to be included', default=10, type=int)

    #TBD parser.add_argument('-p', '--popular_questions', help='select questions with many answers', action='store_true')
    #TBD parser.add_argument('-t', '--top_users', help='find lo-score answers by hi-scoring owners', action='store_true')
    parser.add_argument('-u', '--user_eval', help='user can evaluate answers and add data', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser


if __name__ == '__main__':
    # Set the number of top scoring owners to select from the data.
    num_owners = 10  # Default is 10.
    num_owners = 40  # Default is 10.
    num_owners = 100  # Default is 10.
    print("num_owners: ", num_owners)
    keyword = 'beginner'
    keyword = 'yield'
    # D keyword = 'begin'
    # D keyword = 'pandas'
    #D keyword = 'Python'  # Both Title & Body of data sets have it; for debug
    print("Keyword: ", keyword)

    parser = get_parser()
    args = vars(parser.parse_args())

    main()

