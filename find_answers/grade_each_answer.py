# -*- coding: utf-8 -*-

# Using ~/anaconda3/bin/python: Python 3.5.2 :: Anaconda 4.2.0 (64-bit)

#   Time-stamp: <Tue 2017 May 23 07:29:36 PMPM clpoda>
"""grade_each_answer.py

   A utility program to prepare data files for analysis by the
   fga_find_good_answers.py (fga*) program.

   Show the user a question and an answer from a data file;
   ask user to grade the answer; save the grade.

   In other s/w, that grade will be compared to the grade
   for that answer from the NLP s/w, to measure the quality
   of the NLP s/w.  The NLP s/w is currently implemented
   in fga*.

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

# Note. This code was initially written inside fga*,
# then moved from there to this separate program.


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
import os
import pandas as pd
from shutil import copyfile


def main():
    init()

    global end
    end = 999999
    if args['debug']:
        end = 55
        print('Running in debug mode.')
        print('  end set to: ', end)
        print()

    if args['user_eval']:
        print('Open the data frame for user to evaluate some answers.\n')
        read_and_grade_answers()
        exit


def init():
    """Initialize some settings for the program.
    """
    # Initialize settings for pandas.
    pd.set_option('display.width', 0)  # 0=no limit, use for debugging

    # Don't show commas in large numbers.
    # Show OwnerUserId w/o '.0' suffix.
    pd.options.display.float_format = '{:.0f}'.format


def read_and_grade_answers():
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
        # If ungraded_answers.csv file is missing,
        # copy from a master file with no graded answers.
        # Create the grade file only if it does not exist, ie,
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
    s: show next answer
    """
    user_cmd = ''

    # Find first ungraded answer; show Q&A; ask user for input.
    print('Some answer records that have not been graded:\n')
    for index, row in  ungraded_answers_df.iterrows():
        if row['Grade'] == 'N':
            # Show Q then A then ask user to grade the A.
            if not pd.isnull(row['Title']):  # Found a question.
                q_id = row['Id']
                q_title = row['Title']
                q_body = row['Body']
                print("########## #D Found question id: ", q_id)
                #D print("Q.Title:\n", q_title)
                continue  # Finished w/ Q; jump to next item in for-loop & look for next A.
            elif pd.isnull(row['Title']): # Found an answer.
                #D print("#D Found an answer.")
                user_cmd = show_current_q_a(q_id, q_title, q_body, row)
            else: # Found a problem
                print("ERR. Bad data, neither Q nor A at ID: " + row['Id'])
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
                    print('\n### Next ###################\n')
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
                    print('\n### Next ###################\n')
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
            #D print("#D Last stmt of the cmd interpretation if-clause; go to next item.")
        #D print("#D Last stmt of the for-loop; go to next item.")
    print()
    #D print("#D Last stmt of read_and_grade_answers(); return.\n")
    print('Finished all Q&A in the file: ', ungr_file, ' ;  Exiting.')
    return


def store_grade(grade, index, df):
    # Save the grade and notes to the current record.
    note_text = input("Enter text for a note; end it with the Enter key: ")
    #ORG df.ix[index, 'Grade'] = 'h'
    df.ix[index, 'Grade'] = grade
    df.ix[index, 'Notes'] = note_text
    #TBF print("#D Current row:\n###\n", df[index])
    print('\n### Next ###################\n')
    return 


def show_current_q_a(q_id, q_title, q_body, row):
    print("Q.Id:", q_id, "   Q.Title:\n", q_title[:end], '\n')
    print("Q.Body:\n", q_body[:end])
    print("----------------------\n")
    print("A.Id, A.Body:\n")
    print(row['Id'], "\n", row['Body'][:end])
    print("======================\n")
    if not args['debug'] and len(q_body) > end:
        print('WARN, Question body may be truncated; ID: ', q_id)
    if not args['debug'] and len(row['Body']) > end:
        print('WARN, Answer body may be truncated; ID: ', row['Id'])
    print("Scroll up to read current question and answer.")
    cmd_prompt = "Enter a grade or command: a b c d f ... i [m]enu [h]elp: "
    user_cmd = ''
    while user_cmd == "":  # Repeat the request if only the Enter key is pressed.
        user_cmd = input(cmd_prompt)
    #D print("User entered this cmd: ", user_cmd)
    return user_cmd


def get_parser():
    parser = argparse.ArgumentParser(description='Grade answers found in stackoverflow data')

    parser.add_argument('-d', '--debug', help='Use settings to help with debugging', action='store_true')
    parser.add_argument('-u', '--user_eval', help='user can evaluate answers and add data', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = vars(parser.parse_args())

    main()

