# -*- coding: utf-8 -*-
#
# Using ~/anaconda3/bin/python: Python 3.5.2 :: Anaconda 4.2.0 (64-bit)
# Using Python 3.4.5 :: Anaconda 4.3.0 (64-bit), since Tue2017_0710
# Requires Python 3; does not work w/ Python 2.
#
"""search_for_keyword.py

OBSOLETE, Mon2018_0702, Replaced by sga_show_good_answers.py.
OBSOLETE, Mon2018_0702, Replaced by sga_show_good_answers.py.
OBSOLETE, Mon2018_0702, Replaced by sga_show_good_answers.py.


A module that searches the i/p data for one or more terms.

Input data from the calling routine are
the desired keyword to find,
the command line option namespace,
a pandas dataframe holding questions, answers, and owner data,
and an optional list of columns to include in the o/p.



Reference:

    This file is part of the learn_python/find_answers project
    at this URL:

        https://github.com/clp/learn_python/tree/master/find_answers



Usage:

    In the calling module:
        import search_for_keyword as sfk
        sfk.search_for_keyword(keyword, opt_ns, popular_qa_df, columns_l)

    pydoc search_for_keyword

    See the fga_find_good_answers.py program for an example that
    uses this module.



Overview of Program Actions:

    Receive the Q & A data, and other data from the calling code.

    Call sga.select_keyword_recs() w/ the desired arguments,
    which returns a dataframe.

    Return that dataframe to the caller.

----------------------------------------------------------


Requirements
    Python 3, tested with v 3.6.1.
    pytest for tests, tested with v 3.0.7.

OBSOLETE, Mon2018_0702, Replaced by sga_show_good_answers.py.
OBSOLETE, Mon2018_0702, Replaced by sga_show_good_answers.py.
OBSOLETE, Mon2018_0702, Replaced by sga_show_good_answers.py.

"""

import os

import config as cf
import sga_show_good_answers as sga
import util.write as wr


CURRENT_FILE = os.path.basename(__file__)
DATADIR = cf.DATADIR

cf.logger.info(cf.log_file + ' - Start logging for ' + CURRENT_FILE)


def search_for_keyword(keyword, opt_ns, popular_qa_df, columns_l):
    """Search for keywords specified by caller.

    Save ID's of Q & A that have keyword, or that are related to those
    records, to a file.

    Return the dataframe that has the Q & A with keyword & related Q & A.
    """
    # Specify default output columns to use.
    if not columns_l:
        columns_l = ['HSTCount', 'Score', 'Id', 'ParentId', 'Title', 'Body']
    cf.logger.info('sfk.search*keyword(): Search for a term: ' + \
            keyword + '\n')

    qa_with_keyword_df = sga.select_keyword_recs(
        keyword, opt_ns, popular_qa_df, columns_l)

    if qa_with_keyword_df.empty:
        cf.logger.warning('sfk.search*keyword(): keyword [' + \
                keyword + '] not found in popular_qa_df.')
        return qa_with_keyword_df  # TBD. What debug data to print here?

    return qa_with_keyword_df 


