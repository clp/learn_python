# -*- coding: utf-8 -*-
#
# Using ~/anaconda3/bin/python: Python 3.5.2 :: Anaconda 4.2.0 (64-bit)
# Using Python 3.4.5 :: Anaconda 4.3.0 (64-bit), since Tue2017_0710
# Requires Python 3; does not work w/ Python 2.
#
"""sga_show_good_answers.py


Search the input data for term(s) specified on the cmd line;
then show the records that contain the term(s).

Input data is built by the fga_find_good_answers program,
ie, answers in stackoverflow data that might be good,
but 'hidden' because they have low scores.

Usage:
    sga_show_good_answers.sh
    python sga_show_good_answers.py
    pydoc  sga_show_good_answers


Initialization

    Run the fga*.py program to build the required data set.

    Enter the search terms as a cmd line argument.

    Other CLI options are available, to specify the size
    and type of the output from the program.


Overview of Program Actions

    Read the cmd line options for search term and options.

    Read the question and answer i/p file.

    Check the cache for the search term; use it if avbl.

    Search the collection of question and answer title
    and body fields for the search term;
    use any matching records for output.

    Do a fuzzy search of the collection of question and answer title
    and body fields for the search term;
    use any matching records for output.

    Output choices:
        Show short summary of top-N matches.

        Show full text of top-N matches.

        Save full text of all matches to a temporary cached file.
        Tell user where it is and they can open it w/ an editor.

    If no match found, print that message and stop.


----------------------------------------------------------

#TBD,Tue2017_1212_21:05  Rvw & Update

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


import os
import pandas as pd

import config as cf
import fga_find_good_answers as fga


log_msg = cf.log_file + ' - Start logging for ' + os.path.basename(__file__)
cf.logger.info(log_msg)


def main():
    pass


def select_keyword_recs(keyword, qa_df, columns_l, opt_ns, DATADIR):
    """Find the Q's & A's from the filtered dataframe
    that contain the keyword, in Title or Body.
    Use those Id's to find the related* Q's and A's that do not
    contain the keyword.
    Collect all these Id's into a list, and eliminate all
    the redundant entries.
    Build a list of unique Id's in the desired** order.
    Show the data with those Id's, and save it for analysis.

    [*]A related question is a question that has an answer that
    contains a keyword.

    A related answer is an answer for a question that
    contains a keyword.

    [**]Desired order is to organize Q&A groups, and the answers
    inside each group, in this manner:

    1.  Sort answers by highest HSTCount, highest Score, lowest Id.

    2.  Starting with the highest answer, build its Q&A group
    from its related question, then include all that question's
    related answers.  The answers should be sorted in order
    from highest to lowest.

    3.  Continue with the next highest answer remaining, and
    build its Q&A group.  Repeat until all Q&A groups have been
    built.


    Important variables.

    qa_df: filtered dataframe of questions and answers that
    are pop and top, ie, popular and from owners with high
    reputation scores.
    """

    print('\n#D sga.select*(), at start.')
    if qa_df.empty:
        print('sga.select*(): Missing data, qa_df is empty.')
        return
    #
    # Get a pandas series of booleans for filtering:
    # Check Question & Answer, both Title and Body columns.
    qt_sr = qa_df.Title.str.contains(keyword, regex=False)
    qab_sr = qa_df.Body.str.contains(keyword, regex=False)
    #
    # Combine two series into one w/ boolean OR.
    qa_contains_sr = qab_sr | qt_sr
    #
    # Build a df with Q's and A's that contain the keyword.
    qak_df = qa_df[columns_l][qa_contains_sr]
    if qak_df.empty:
        print('sga.select*():qak: Search term not found, qak_df is empty.')
        return
    #
    # Print summary, 1 line/record:
    print()
    print('#D qak_df summary, Q & A that contain the keyword:\n', qak_df[:] , '\n')
    #
    # Build a list of Id values of Q's and A's that contain the keyword.
    #
    parent_id = -1  # Initlz if needed outside the loop, and if search has no matches.
    ans_ids_from_search_l = []  # A's w/ keywords.
    ques_ids_from_search_l = [] # Q's w/ keywords and Q's of A's w/ keywords
    #
    for index, row in qak_df.iterrows():
        rec_id = row['Id']
        title = row['Title']
        parent_id = row['ParentId']
        #
        if pd.isnull(title):
            # Found an answer w/ keyword, add its Q.Id to ques list.
            ques_ids_from_search_l.append(parent_id)
            # Also add the A.Id to the answer list.
            ans_ids_from_search_l.append(rec_id)
        elif title:
            # Found a question w/ keyword; add it to ques list.
            ques_ids_from_search_l.append(rec_id)
        else:
            #TBD, How to test this path, it should never be reached.
                # Need a bad data record for the test.
                # Maybe need record w/ no Title field.
            print("\nselect*(): Bad data found, problem with Title?  Id, Title:")
            print("Id: ", rec_id)
            print("ParentId: ", parent_id)
            print("Title: ", title)
            raise SystemExit()  # TBD too drastic?  Maybe return, show menu?
    #
    # Build sets of unique Id's of Q's and A's w/ keyword.
    #
    ques_ids_from_search_l = list(set(ques_ids_from_search_l))
    ans_ids_from_search_l = list(set(ans_ids_from_search_l))
    #
    # Build a list of answer Id's.
    #
    ans_ids_sorted_l = []
    ans_ids_l = []
    for aid in ans_ids_from_search_l:
        a_df = qak_df.loc[qak_df['Id'] == aid]
            # Using qa_df here is not obviously diff from qak.
        ans_ids_l.append([a_df['HSTCount'].values[0], a_df['Score'].values[0], a_df['Id'].values[0]])
    #
    # Sort the list of answer Id's by HSTCount, Score, Id.
    #
    ans_ids_sorted_l = sorted(ans_ids_l, reverse=True)
    #D print('#D.1547 , ans_ids_sorted_l sorted by hstc,score,id:\n', ans_ids_sorted_l)
    #
    # Use sorted A's to build list of ordered Q's.
    #
    ques_ids_ord_l = []
    if opt_ns.verbose:
        print('#D Q.Id, Q.Title:')
    for hstc, score, aid in ans_ids_sorted_l:
        q_df = qak_df.loc[qak_df['Id'] == aid]
        parent_id = q_df['ParentId'].iloc[0]
        ques_ids_ord_l.append(parent_id)
        #
        if opt_ns.verbose:
            print(parent_id)
            tmp_df = qa_df.loc[qa_df['Id'] == parent_id]
            print(tmp_df['Title'].iloc[0])
    #
    # Extend the list of ordered Q's w/ Q.Id's that have the keyword.
    #
    ques_ids_ord_l.extend(ques_ids_from_search_l)
    #
    # Remove duplicate Q.Id's w/o changing the order of the list.
    #
    qid_ord_l = []
    for i in ques_ids_ord_l:
        if i not in qid_ord_l:
            qid_ord_l.append(i)
    #
    # Build list of Q.Id and A.Id in the right order for output.
    #
    qa_id_ord_l = []
    for qid in qid_ord_l:
        # Save the Q.Id.
        qa_id_ord_l.append(qid)
        #
        # Find & sort & save all A.Id's for this Q.
        #
        ans_match_df = qa_df[qa_df['ParentId'] == qid]
        ams_df = ans_match_df.sort_values(['HSTCount', 'Score', 'Id'], ascending=[False, False, True])
        for hstc, score, aid in ams_df[['HSTCount', 'Score', 'Id']].values:
            qa_id_ord_l.append(aid)
    #
    # Build o/p df w/ Q's and A's in right order, from the list of Id's.
    #
    qa_keyword_df_l = []
    for item in qa_id_ord_l:
        for index, row in qa_df.iterrows():
            if row['Id'] == item:
                qa_keyword_df_l.append(row)
                continue
    qa_keyword_df = pd.DataFrame(qa_keyword_df_l)
    #
    # Write o/p to disk files.
    #
    search_fname = 'search_result_full.csv'
    fga.save_prior_file(DATADIR, search_fname)
    qa_keyword_df.to_csv(DATADIR + search_fname)
    #
    search_fname = 'search_result_full.html'
    fga.save_prior_file(DATADIR, search_fname)
    columns_l = ['HSTCount', 'Score', 'Id', 'Title', 'Body']
    fga.write_full_df_to_html_file(qa_keyword_df, DATADIR, search_fname, columns_l)
    #
    return  qa_keyword_df


if __name__ == '__main__':
    main()

'bye'
