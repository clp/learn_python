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
import re

import config as cf
import nltk_ex25 as nl


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
    cf.logger.info('WARN: Moved old file to tmp/ & it will be overwritten next; save it if needed: ' +
        TMP + wfile + '\n')
    return


def write_df_to_csv(in_df, wdir, wfile):
    """Write full contents of all columns of a data frame to a csv file.
    """
    if in_df.empty:
        print('WARN: write*csv(): Missing or empty dataframe for wfile: ', wfile)
        return
    # Used for testing and debugging.
    pd.set_option('display.max_colwidth', -1)  # -1=no limit, for debug
    save_prior_file(wdir, wfile)
    outfile = wdir + wfile
    in_df.to_csv(outfile)
    pd.set_option('display.max_colwidth', MAX_COL_WID)  # -1=no limit, for debug
    return



def write_part_df_to_csv(in_df, wdir, wfile, columns_l, hdr, indx):
    """Write full contents of some columns of a data frame to a csv file.
    """
    if in_df.empty:
        print('WARN: write*csv(): Missing or empty dataframe for wfile: ', wfile)
        return
    # TBD Used for testing and debugging.
    pd.set_option('display.max_colwidth', -1)  # -1=no limit, for debug
    save_prior_file(wdir, wfile)
    outfile = wdir + wfile
    in_df[columns_l].to_csv(
        outfile, header=hdr, index=indx, sep=',', mode='w')
    pd.set_option('display.max_colwidth', MAX_COL_WID)  # -1=no limit, for debug
    return


def replace_line_breaks(in_s):
    """Replace escaped line break chars so text inside HTML
    pre-blocks and code-blocks (inside HTML table cells)
    is rendered properly.
    """
    out_s = in_s
    out_s = out_s.replace('\\r\\n', '\n')
    out_s = out_s.replace('\\n\\n', '\n')
    out_s = out_s.replace('\\n', '\n')
    return out_s


def replace_line_breaks_for_otl(in_s):
    """TBD, fix comment:
    Replace escaped line break chars so text inside HTML
    pre-blocks and code-blocks (inside HTML table cells)
    is rendered properly.

    Add indentation after a newline for the OTL output.
    """

    out_s = in_s
    out_s = out_s.replace(r'\r\n', '\n')  # Pattern seen in a3*.csv.
    out_s = out_s.replace(r'\n\n', '\n')  # Pattern seen in a3*.csv.
    out_s = out_s.replace(r'\n\r', '\n')  # Pattern not seen.
    out_s = out_s.replace(r'\n', '\n    ')
    #D print('#D-replace_lb_otl in_s: ', in_s[:999])
    #D print()
    #D print('#D-replace_lb_otl out_s: ', out_s[:999])
    return out_s


def write_df_to_html(in_df, wdir, wfile, columns_l):
    """Write full contents of some columns of a data frame to an html file.

    Use the list of columns included in this function
    if caller does not specify any.
    """
    if in_df.empty:
        print('WARN: write*html(): Missing or empty dataframe for wfile: ', wfile)
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
    out_s = replace_line_breaks(in_s)

    # Concatenate css w/ html file to format the o/p.
    with open(outfile, 'w') as f:
        cf.logger.info('NOTE: Writing data to html outfile: ' + outfile)
        f.write(HEADER)
        f.write(out_s)
        f.write(FOOTER)

    pd.set_option('display.max_colwidth', MAX_COL_WID)  # -1=no limit, for debug
    return


def write_df_to_otl(in_df, wdir, wfile, columns_l):
    """Write full contents of some columns of a data frame to an otl file.

    Open that file w/ Vim + VimOutliner for easy overview of all questions,
    and quick navigation.

    TBD, Use the list of columns specified in this function
    if caller does not specify such a list.
    """
    if in_df.empty:
        print('WARN: write*otl(): Input dataframe empty or not found.')
        return
    pd.set_option('display.max_colwidth', -1)  # -1=no limit, for debug
    outfile = wdir + wfile
    save_prior_file(wdir, wfile)
    #
    # Save o/p to a string and do not specify an output file in
    # calling to_string().
    # Use 'index=False' to prevent showing index in column 1.
    in_s = in_df[columns_l].to_string(header=False, index=False)
    print('#D-write1, len in_s: ', len(in_s) )

    #D #TBD,Sat2018_0630_14:58 Debug, 
    #D import pdb
    #D pdb.set_trace()
    #D print()
    #D print('#D-write_otl in_s: ', in_s[:999])

    #
    # Delete long strings of spaces at end of each line.
    # Replace blank spaces at end of each line w/ only the newline char.
    # Do this for all matching patterns in the string in one cmd.
    out_s = in_s
    out_s = re.sub('  +\n', '\n', out_s)

    # Convert html line breaks to newlines before stripping html.
    out_s = re.sub(r'<br>', '\n    ', out_s)
    out_s = re.sub(r'<br/>', '\n    ', out_s)

    # Clean the newlines in the string so each line has proper indent.
    out_s = nl.strip_html(out_s, "lxml")
    out_s = replace_line_breaks_for_otl(out_s)
    #
    print('#D-write2, len out_s: ', len(out_s) )
    #D print('#D-write3, out_s: ', out_s[:599] )
    #
    # Replace empty lines w/ INDENT+##
    out_s = re.sub(r'\n\s*\n', r'\n        ##\n', out_s)
    print('#D-write4, len out_s: ', len(out_s) )

    #D print()
    #D print('#D-write_otl out_s: ', out_s[:999])

    with open(outfile, 'w') as f:
        cf.logger.info('NOTE: Writing data to otl outfile: ' + outfile)
        f.write(out_s)

    pd.set_option('display.max_colwidth', MAX_COL_WID)  # -1=no limit, for debug
    return


init()


if __name__ == '__main__':
    main()

    log_msg = cf.log_file + ' - Finish logging for ' + \
        os.path.basename(__file__) + '\n'
    cf.logger.warning(log_msg)

