# -*- coding: utf-8 -*-
#
# Using ~/anaconda3/bin/python: Python 3.5.2 :: Anaconda 4.2.0 (64-bit)
# Using Python 3.4.5 :: Anaconda 4.3.0 (64-bit), since Tue2017_0710
# Requires Python 3; does not work w/ Python 2.
#
"""text_ui.py


A module that prints a menu of actions on screen, and prompts
the user to enter a command, and processes that command.

Input data from the calling routine is several pandas dataframes,
holding questions, answers, and owner data.



Reference:

    This file is part of the learn_python/find_answers project
    at this URL:

        https://github.com/clp/learn_python/tree/master/find_answers



Usage:

    In the calling module:
        import text_ui as tui
        tui.show_menu(popular_qa_df, all_ans_df, opt_ns)

    pydoc  text_ui

    See the fga_find_good_answers.py program for an example that
    uses this module.



Overview of Program Actions:

    Receive the Q & A data, and other data from the calling code.

    Display the prompt and menu and other data in response
    to the user command.

    Call other functions in this module or in other modules to handle
    the user command.


    Some menu actions use the following functions in this module.

    build_stats():
    Compute and save statistics about the selected data, to use
    in further analysis and plotting, eg, owner reputation (mean
    score) and length of the body text of the record.

    check_owner_reputation():
    Provide reputation data from a dataframe or a file.  If it
    does not exist, call the function to build it.

----------------------------------------------------------


Requirements
    Python 3, tested with v 3.6.1.
    pytest for tests, tested with v 3.0.7.

"""

import os
import pandas as pd

import config as cf
import draw_plots as dr
import sga_show_good_answers as sga
import util.write as wr


CURRENT_FILE = os.path.basename(__file__)
DATADIR = cf.DATADIR

cf.logger.info(cf.log_file + ' - Start logging for ' + CURRENT_FILE)

def main():
    pass


def print_short_record(popular_qa_df, saved_index):
    print(popular_qa_df[['Id', 'Title', 'Body']].iloc[[saved_index]])


def show_menu(popular_qa_df, all_ans_df, opt_ns, progress_i):
    """Show prompt to user; get and handle their request.
    """
    user_menu = """    The menu choices:
    drm: calculate reputation matrix of owners
    dsm: draw and plot q&a statistics matrix
    d: draw default plot of current data
    dh: draw default histogram plot of current data
    dm: draw scatter matrix plot of current data
    h, ?: show help text, the menu
    lek: look for exact keywords in questions and answers
    m: show menu
    q: save data and quit the program
    s: show current item: question or answer
    sn: show next item: question or answer
    sp: show prior item: question or answer
    """
    user_cmd = ''
    saved_index = 0
    owner_reputation_df = pd.DataFrame()

    # Show prompt & wait for a cmd.
    print("======================\n")
    cmd_prompt = "Enter a command: q[uit] [m]enu  ...  [h]elp: "
    while user_cmd == "":  # Repeat the prompt if no i/p given.
        user_cmd = input(cmd_prompt)
    print("User entered this cmd: ", user_cmd)

    # Loop to handle user request.
    while user_cmd:
        if user_cmd.lower() == 'm':
            print(user_menu)
            # Clear user_cmd here to avoid infinite repetition of while loop.
            user_cmd = ''
        elif user_cmd.lower() == 'q':
            print("Quit the program.")
            log_msg = cf.log_file + ' - Quit, user req; Finish logging for ' + \
                CURRENT_FILE + '\n'
            cf.logger.warning(log_msg)
            raise SystemExit()
        elif user_cmd == '?' or user_cmd == 'h':
            print(user_menu)
            user_cmd = ''
        elif user_cmd.lower() == 's':
            user_cmd = ''
            if popular_qa_df.empty:
                print("WARN: dataframe empty or not found; try restarting.")
            else:
                print_short_record(popular_qa_df, saved_index)
        elif user_cmd.lower() == 'sn':  # show next item
            user_cmd = ''
            if popular_qa_df.empty:
                print("WARN: dataframe empty or not found; try restarting.")
            else:
                saved_index += 1
                print_short_record(popular_qa_df, saved_index)
        elif user_cmd.lower() == 'sp':  # show prior item
            user_cmd = ''
            if popular_qa_df.empty:
                print("WARN: dataframe empty or not found; try restarting.")
            else:
                saved_index -= 1
                print_short_record(popular_qa_df, saved_index)
        elif user_cmd.lower() == 'd':
            user_cmd = ''
            if popular_qa_df.empty:
                print("WARN: dataframe empty or not found; try restarting.")
            else:
                print("Drawing the default plot, with Score and HSTCount.")
                dr.draw_scatter_plot(
                    popular_qa_df,
                    'Score',
                    'HSTCount',
                    'Score',
                    'HSTCount')
        elif user_cmd.lower() == 'dh':
            user_cmd = ''
            if popular_qa_df.empty:
                print("WARN: dataframe empty or not found; try restarting.")
            else:
                print("Drawing the default histogram plot.")
                dr.draw_histogram_plot(popular_qa_df)
        elif user_cmd.lower() == 'dm':  # Scatter matrix plot
            user_cmd = ''
            if popular_qa_df.empty:
                print("WARN: dataframe empty or not found; try restarting.")
            else:
                print("Drawing the default scatter matrix plot.")
                dr.draw_scatter_matrix_plot(popular_qa_df)
        # drm: Draw Reputation matrix, mean, answers only; scatter.
        elif user_cmd.lower() == 'drm':
            user_cmd = ''
            owner_reputation_df = check_owner_reputation(
                all_ans_df, owner_reputation_df)
            #
            if owner_reputation_df.empty:
                print("WARN: owner reputation dataframe empty or not found.")
            else:
                print("NOTE: Drawing the owner reputation scatter matrix plot.")
                dr.draw_scatter_matrix_plot(
                    owner_reputation_df[['MeanScore', 'OwnerUserId']])
        # dsm: Draw q&a statistics matrix
        elif user_cmd.lower() == 'dsm':
            user_cmd = ''
            owner_reputation_df = check_owner_reputation(
                all_ans_df, owner_reputation_df)
            #
            qa_stats_df = build_stats(popular_qa_df, owner_reputation_df)
            #
            if qa_stats_df.empty:
                print("WARN: qa_stats_df empty or not found.")
            else:
                print("NOTE: Drawing the qa_stats_df scatter matrix plot.")
                dr.draw_scatter_matrix_plot(
                    qa_stats_df[['Score', 'BodyLength', 'OwnerRep', 'HSTCount']])
        # lek: Look for exact keywords in the Q&A df; now case sensitive.
        elif user_cmd.lower() == 'lek':
            user_cmd = 'lek'  # Force menu to always repeat the lek prompt.
            search_prompt = "\nType a search term; or press Enter for main menu: "
            search_term = input(search_prompt)
            if search_term == "":  # Return to main menu and ask for a cmd.
                user_cmd = 'm'
                continue
            print("User entered this search_term: ", search_term)
            log_msg = cf.log_file + ' - Search term entered by user: ' + \
                search_term + '\n'
            cf.logger.warning(log_msg)
            #
            columns_l = [
                'HSTCount',
                'Score',
                'Id',
                'ParentId',
                'Title',
                'Body',
                ]
                #TBD 'HiScoreTerms']
            qa_group_with_keyword_df = pd.DataFrame()  # Initlz at each call
            #
            # TBD Chg popular_qa_df to a df w/ more records, for dbg & initial
            # use.  Beware of memory & performance issues.
            qa_group_with_keyword_df = sga.select_keyword_recs(
                search_term, popular_qa_df, columns_l, opt_ns, progress_i)

            # Search term not found; show search prompt.
            if qa_group_with_keyword_df.empty:
                user_cmd = 'lek'
                print()
                continue

            #D # Save six columns to disk file.
            #D wr.write_part_df_to_csv(
                #D qa_group_with_keyword_df, DATADIR,
                #D 'qa_with_keyword.csv', columns_l, True, None)

            # Write only the Id column of df to disk, sorted by Id.
            # Sort it to match the ref file, so out-of-order data
            # does not cause test to fail.
            id_df = qa_group_with_keyword_df[['Id']].sort_values(['Id'])
            wr.write_part_df_to_csv(
                id_df, DATADIR,
                'qa_withkey_id.csv', ['Id'], True, None)


        else:
            print("Got bad cmd from user: ", user_cmd)
            print(user_menu)
            # Clear user_cmd here to avoid infinite repetition of while loop.
            user_cmd = ''
        # Show prompt & wait for a cmd.
        print("======================\n")
        while user_cmd == "":  # Repeat the prompt if no i/p given.
            user_cmd = input(cmd_prompt)
    print("#D End of the cmd interpretation loop; return.\n")

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
    qa_stats_df = qa_df[['Id', 'OwnerUserId', 'ParentId', 'Score', 'HSTCount']]
    # Add new column to df & initlz it.
    qa_stats_df = qa_stats_df.assign(BodyLength=qa_stats_df.Id)

    for index, row in qa_df.iterrows():
        ouid = row['OwnerUserId']
        try:
            owner_rep = round(
                or_df.loc[or_df['OwnerUserId'] == ouid, 'MeanScore'].iloc[0])
            # Add new column to df.
            qa_stats_df.loc[index, 'OwnerRep'] = owner_rep
        except IndexError:
            # TBD, Some answers in the data file were made by Owners
            # who are not yet in the reputation df.
            print(
                "build_stats(): ouid not in owner reputation dataframe;",
                " index,ouid: ",
                index,
                ouid)
            print("build_stats():data from the problem row in qa_df:\n", row)
            print()

        # Save length of body text of each answer.
        qa_stats_df.loc[index, 'BodyLength'] = len(row['Body'])

    qa_stats_df = qa_stats_df[['Id',
                               'ParentId',
                               'OwnerUserId',
                               'Score',
                               'BodyLength',
                               'OwnerRep',
                               'HSTCount']]

    stats_fname = DATADIR + 'qa_stats_by_dsm.csv'
    wr.save_prior_file('', stats_fname)
    qa_stats_df.to_csv(stats_fname)

    stats_fname = DATADIR + 'qa_stats_by_dsm.html'
    wr.save_prior_file('', stats_fname)
    qa_stats_df.to_html(stats_fname)

    return qa_stats_df


def check_owner_reputation(all_ans_df, owner_reputation_df):
    """Check for dataframe with reputation of each OwnerUserId.
    If not found, then calculate reputation of each OwnerUserId
    in the i/p data, based on Score of all answers they provided.
    Save the data to a disk file and use it when needed, so the
    calculation need not be done every time this program runs.
    """
    own_rep_file = DATADIR + 'owner_reputation.csv'
    # TBD Must chk i/p file & replace owner_rep*.csv if
    #  a different file was used.  OR, just build this file from Answers.csv
    #  which should have all answers & produce good reputation data.

    if not owner_reputation_df.empty:
        return owner_reputation_df

    if os.path.exists(own_rep_file):
        print("owner rep file, " + own_rep_file + ", found; read it.")
        owner_reputation_df = pd.read_csv(
            own_rep_file,
            encoding='latin-1',
            warn_bad_lines=False,
            error_bad_lines=False)
        return owner_reputation_df
    else:
        print(
            "owner rep file, " +
            own_rep_file +
            ", not found; build it.")
        print("This should be a one-time operation w/ data saved on disk.")
        owner_reputation_df = gd2_group_data(all_ans_df)
        owner_reputation_df.to_csv(own_rep_file)
    return owner_reputation_df


if __name__ == '__main__':
    main()
