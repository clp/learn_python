# -*- coding: utf-8 -*-
#
# Using ~/anaconda3/bin/python: Python 3.5.2 :: Anaconda 4.2.0 (64-bit)
# Using Python 3.4.5 :: Anaconda 4.3.0 (64-bit), since Tue2017_0710
# Requires Python 3; does not work w/ Python 2.
#
"""write.py

A module that contains functions that write data to disk files.

Input data from the calling routine is a pandas dataframe.

Output includes files on disk in csv and html formats.



Reference:

    This file is part of the learn_python/find_answers project
    at this URL:

        https://github.com/clp/learn_python/tree/master/find_answers



Usage:

    In the calling module:
        import util/write as wr
        wr.save_prior_file(wdir, wfile)
        wr.write_df_to_csv(popular_qa_df, 'popular_qa.csv')

    pydoc  util/write.py

    See the fga_find_good_answers.py program for an example that
    uses this module.



Initialization and Operation:

    Run the fga*.py program to build the required data set.
    It calls this module's functions to write processed data
    to the o/p files.  See these files in the data/ dir:
        popular_qa.csv
        popular_qa.html
        popular_qa_title_body.html

    When prompted from the fga menu, choose "lek: look for
    exact keyword in questions and answers".  Enter a search
    term that is in the data set and that produces output.
    See these additional files in the data/ dir:
        search_result*.csv
        search_result*.html



Overview of Program Actions:

    Receive the search term, Q & A data, and other data
    from the calling code.

    Prepare the data by selecting columns from the dataframe.

    Filter the data as needed, eg, modify line breaks so the
    HTML o/p is correct.

    Write a pandas dataframe to a disk file using the pandas
    functions, to_csv() and to_html().

----------------------------------------------------------


Requirements
    Python 3, tested with v 3.6.1.
    pytest for tests, tested with v 3.0.7.

"""

import os
import pandas as pd

import config as cf


cf.logger.info(cf.log_file + ' - Start logging for ' + os.path.basename(__file__))

FOOTER = cf.FOOTER
HEADER = cf.HEADER
MAX_COL_WID = cf.MAX_COL_WID
TMP = cf.TMP


def main():
    pass


def init():
    """Initialize some settings for the program.
    """

    # Initialize settings for pandas.
    pd.set_option('display.width', 0)  # 0=no limit, for debug
    pd.set_option('display.max_colwidth', MAX_COL_WID)  # -1=no limit, for debug

    # Don't show commas in large numbers.
    # Show OwnerUserId w/o '.0' suffix.
    pd.options.display.float_format = '{:.0f}'.format


def save_prior_file(wdir, wfile):
    """Save backup copy of a file w/ same name.

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


def write_df_to_csv(in_df, wdir, wfile):
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
    is rendered properly.
    """
    out_s = in_s.replace('\\r\\n', '\n')
    out_s = out_s.replace('\\n\\n', '\n')
    out_s = out_s.replace('\\n', '\n')
    return out_s


def write_df_to_html(in_df, wdir, wfile, columns_l):
    """Write full contents of some columns of a data frame to an html file.

    Use the list of columns included in this function
    if caller does not specify any.
    """
    if in_df.empty:
        print('WARN: write*html(): Input dataframe empty or not found.')
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
    # so the HTML inside each table cell renders properly.
    in_s = replace_line_breaks(in_s)

    # Concatenate css w/ html file to format the o/p.
    with open(outfile, 'w') as f:
        cf.logger.info('NOTE: Writing data to html outfile: ' + outfile)
        f.write(HEADER)
        f.write(in_s)
        f.write(FOOTER)

    pd.set_option('display.max_colwidth', MAX_COL_WID)  # -1=no limit, for debug
    return


init()


if __name__ == '__main__':
    main()

    log_msg = cf.log_file + ' - Finish logging for ' + \
        os.path.basename(__file__) + '\n'
    cf.logger.warning(log_msg)

