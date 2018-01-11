# -*- coding: utf-8 -*-
#
# Using ~/anaconda3/bin/python: Python 3.5.2 :: Anaconda 4.2.0 (64-bit)
# Using Python 3.4.5 :: Anaconda 4.3.0 (64-bit), since Tue2017_0710
# Requires Python 3; does not work w/ Python 2.
#
#TBD,Wed2018_0110_18:07  to update
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

    Set the value of keyword in response to the prompt.  The i/p
    data will be searched, and matching records will be shown
    and saved to disk files.


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


#TBD.Mon2018_0108_22:23  Update this text.

Output data format of popular_qa.csv o/p file from this program.


Requirements
    Python 3, tested with v 3.6.1.
    pytest for tests, tested with v 3.0.7.

"""

import os
import pandas as pd

import config as cf


cf.logger.info(cf.log_file + ' - Start logging for ' + os.path.basename(__file__))

DATADIR = 'data/'
FILEA3 = 'a3_986.csv'
INDIR = 'indir/'
LINE_COUNT = 10
MAX_COL_WID = 20
MAX_HI_SCORE_TERMS = 10
MAX_OWNERS = 20
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

#TBD,Wed2018_0110_18:07  to update

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


#TBD,Wed2018_0110_18:07  to update
def init():
    """Initialize some settings for the program.
    """

    # Initialize settings for pandas.
    pd.set_option('display.width', 0)  # 0=no limit, for debug
    pd.set_option('display.max_colwidth', MAX_COL_WID)  # -1=no limit, for debug

    # Don't show commas in large numbers.
    # Show OwnerUserId w/o '.0' suffix.
    pd.options.display.float_format = '{:.0f}'.format


#TBD.1, Simplify this entire func.
def save_prior_file(wdir, wfile):
    """Save backup copy of a file w/ same name; add '.bak' extension.

    Input wdir should have trailing slash, eg, data/.
    Input wfile should have trailing extension name, eg, .csv.
    """
    import shutil

    outfile = wdir + wfile
    if not os.path.isfile(outfile):
        # Skip the save operation if the file does not exist.
        return

    # Save all backups to tmp/ dir.
    dst_filename = os.path.join(TMP, os.path.basename(outfile))
    shutil.move(outfile, dst_filename)
    cf.logger.info('WARN: Moved old file to tmp storage; save it manually if needed: ' +
        TMP + wfile + '\n')
    return


#NEW def write_df_to_csv(in_df, wdir, wfile):
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
        print('WARN: write_full*(): Input dataframe empty or not found.')
        return
    pd.set_option('display.max_colwidth', -1)  # -1=no limit, for debug
    outfile = wdir + wfile
    save_prior_file(wdir, wfile)
    # Specify default output columns to use.
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
        cf.logger.info('NOTE: Writing data to html outfile: ' + outfile)
        f.write(HEADER)
        f.write(in_s)
        f.write(FOOTER)

    pd.set_option('display.max_colwidth', MAX_COL_WID)  # -1=no limit, for debug
    return


if __name__ == '__main__':

    main()

    log_msg = cf.log_file + ' - Finish logging for ' + \
        os.path.basename(__file__) + '\n'
    cf.logger.warning(log_msg)

'bye'
