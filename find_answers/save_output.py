# -*- coding: utf-8 -*-
#
# Using ~/anaconda3/bin/python: Python 3.5.2 :: Anaconda 4.2.0 (64-bit)
# Using Python 3.4.5 :: Anaconda 4.3.0 (64-bit), since Tue2017_0710
# Requires Python 3; does not work w/ Python 2.
#
"""save_output.py


A module that prepares data to save in various formats to disk
files, eg, csv, html, and otl (outline format).

Input data from the calling routine are several pandas dataframes,
holding questions, answers, and data about them.



Reference:

    This file is part of the learn_python/find_answers project
    at this URL:

        https://github.com/clp/learn_python/tree/master/find_answers



Usage:

    In the calling module:
        import save_output as sav
        sav.save_basic_output(popular_qa_df, qa_with_keyword_df)

    pydoc save_output

    See the fga_find_good_answers.py program for an example that
    uses this module.



Overview of Program Actions:

    #TBD,Thu2018_0510_22:14  , update overview text:

    Receive the Q & A data, and other data from the calling code.

    TBD


----------------------------------------------------------


Requirements
    Python 3, tested with v 3.6.1.
    pytest for tests, tested with v 3.0.7.

"""

import os
import pandas as pd

import config as cf
import util.write as wr


CURRENT_FILE = os.path.basename(__file__)
DATADIR = cf.DATADIR

cf.logger.info(cf.log_file + ' - Start logging for ' + CURRENT_FILE)


def save_basic_output(popular_qa_df, qa_with_keyword_df):
    """Save the dataframe with chosen Q&A groups to csv and html
    and otl (outline format) files.
    """
    wr.write_df_to_csv(popular_qa_df, DATADIR, 'popular_qa.csv')

    columns_l = []
    wr.write_df_to_html(
        popular_qa_df,
        DATADIR,
        'popular_qa.html',
        columns_l)

    columns_l = ['Id', 'Title', 'Body']
    wr.write_df_to_html(
        popular_qa_df,
        DATADIR,
        'popular_qa_title_body.html',
        columns_l)


    # Write search o/p to html file in one column: q.title, q.body, a1, a2, ...
    if qa_with_keyword_df.empty:
        print('#D save_basic*()-html: qa_with_keyword_df is empty.')
        cf.logger.warning('save_basic_output(): ' + \
                'qa_with_keyword_df is empty.')
        return qa_with_keyword_df # TBD. What debug data to print here?

    out_l = list()
    prefix_s = 'Id, Score, HSTCount, CreDate: ' 
    columns_l = ['Id', 'Score', 'HSTCount', 'CreationDate']
    for index, row in qa_with_keyword_df.iterrows():
        if not pd.isnull(row['Title']):
            # Found a question.
            columns_s = ', '.join(str(row[x]) for x in columns_l)
            columns_s = prefix_s + columns_s 
            out_l.append('###')
            out_l.append('Question:')
            out_l.append(columns_s)
            out_l.append(row['Title'])
            #D out_l.append(row['Body'][:99])
            out_l.append(row['Body'])
        else:
            # Found an answer.
            columns_s = ', '.join(str(row[x]) for x in columns_l)
            columns_s = prefix_s + columns_s 
            out_l.append('Answer:')
            out_l.append(columns_s)
            #D out_l.append(row['Body'][:99])
            out_l.append(row['Body'])

    # Convert list to df & make html file from it.
    # User can open that file in a web browser to see data.
    qa_title_body_df = pd.DataFrame(out_l)
    columns_l = [0]
    wr.write_df_to_html(
        qa_title_body_df,
        DATADIR,
        'qa_title_body.html',
        columns_l
        )


    # Write search o/p in outline format to otl file: q.title, q.body, a1, a2, ...
    if qa_with_keyword_df.empty:
        print('#D save_basic*()-otl: qa_with_keyword_df is empty.')
        cf.logger.warning('save_basic_output(): ' + \
                'qa_with_keyword_df is empty.')
        return qa_with_keyword_df # TBD. What debug data to print here?

    out_l = list()
    prefix_s = 'Id, Score, HSTCount, CreDate: ' 
    columns_l = ['Id', 'Score', 'HSTCount', 'CreationDate']
    for index, row in qa_with_keyword_df.iterrows():
        if not pd.isnull(row['Title']):
            # Found a question.
            columns_s = ', '.join(str(row[x]) for x in columns_l)
            columns_s = prefix_s + columns_s 
            out_l.append('Question: ' + row['Title'])
            out_l.append('    ' + 'Q.Body: ' + columns_s)
            #
            body = ''
            # Process each line in the body.
            # O/p has line1 of q.body at col5 & other lines indented at col9.
            lines = list()
            lines = row['Body'].split('\n')
            for line in lines:
                if line.rstrip():
                    # Found line that is not empty.
                    body +=  ' '*4 + line + '\n'
            #D print('#D q.body: ')
            #D print(body)
            #D print('#D end of q.body.')
            #D print()
            #
            out_l.append(body)
        else:
            # Found an answer.
            columns_s = ', '.join(str(row[x]) for x in columns_l)
            columns_s = prefix_s + columns_s 
            out_l.append('    ' + 'Answer: ' + columns_s)
            #
            body = ''
            # Process each line in the body.
            lines = list()
            lines = row['Body'].split('\n')
            for line in lines:
                if line.rstrip():
                    # Found line that is not empty.
                    body +=  ' '*4 + line + '\n'
            #D print('#D a.body: ')
            #D print(body)
            #D print('#D end of a.body.')
            #D print()
            #
            out_l.append(body)

    # Convert list to df & make otl file from it.
    # User can open that file in editor (Vim w/ VimOutliner) to see data.
    qa_title_body_df = pd.DataFrame(out_l)
    columns_l = [0]
    wr.write_df_to_otl(
        qa_title_body_df,
        DATADIR,
        'qa_title_body.otl',
        columns_l
        )


    #D # Write all columns of df to disk file.
    #D wr.write_part_df_to_csv(
        #D qa_with_keyword_df, DATADIR,
        #D 'qa_with_keyword.csv', columns_l, True, None)

    # Write only the Id column of df to disk, sorted by Id.
    # Sort it to match the ref file, so out-of-order data
    # does not cause test to fail.
    id_df = qa_with_keyword_df[['Id']].sort_values(['Id'])
    wr.write_part_df_to_csv(
        id_df, DATADIR,
        'qa_withkey_id.csv', ['Id'], True, None)
    return

