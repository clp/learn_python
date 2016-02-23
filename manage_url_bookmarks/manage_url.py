#! /usr/bin/python

"""manage_url.py - Manage a list of URL bookmarks w/ name, title, 
and optional note fields.

TBD: For study group, project 2016_0215.
"""

"""
TBD, New features to consider for adding:
    Specify a current record & show it.
    scroll up & down rec list & change current rec.
        next and prior record cmds?
        page up & page down list of records cmds?
        up & down arrow keys?
    hilite curr rec in a list?  Or prefix w/ '=>'?
"""

"""Plan.
Read the input file.
    If i/p file not found in current dir, create it.
        TBD, Or ask user to confirm creation.
    TBD, Assign current record = first record.
TBD, Show current record and menu on Home screen.
Prompt for user command and wait.
Interpret user command when entered.
Perform requested action.
On 'Quit', save data to the file, then quit.
    TBD, save prior data file as a backup.
"""

import pickle
import sys

def PrintHelp():
    """
    Show usage and general help data.
    """
    usage_text = """\nUsage:
      manage_url
        Read data from this file in the current dir: url_bookmarks.pickle.
        Show the menu and wait for the user's command.
    """

    menu_help = """Menu commands:
      A: Add a new record.
      D: Delete a record; specify ID number; press 'y' to confirm.
      F: Find; specify a word to find in any record.
      H: Show help.
      S: Show all records.
      Q: Quit the program after saving data to the file.
      U: Update a record; specify ID number, then enter data at prompt.
    """

    print usage_text
    print menu_help


def GetFileData(infile):
    """
    Open an existing data file and read it.
    Create a new data file if no data file was found in current dir.
    """
    data = {}
    try:
        with open(infile, 'rb') as pf:
            (record_id, data) = pickle.load(pf)
        #D print "DBG, gfd, after load, data: ", data
        return (record_id, data)
    except (EOFError, IOError):
        print "WARN: File empty or not found in current dir: ", infile
        print "  Creating an empty data file.  Use Add to fill it with bookmarks."
        open(infile, 'wb').close()
    return


def GetMenu():
    menu = "\nPress the first letter of a command (ADFHSQU), then press Enter:\n"
    menu += "Commands: Add  Delete  Find  Help  Show  Quit  Update\n"
    return menu


def AddNewEntry(record_id, url_dict):
    """
    Ask user to enter data for a new record.
    Store it in the data structure.
    """
    name = raw_input("Enter the name of the entry> ")
    url = raw_input("Enter the URL> ")
    note = raw_input("Optional: Enter a one-line note> ")
    record_id += 1
    #D print "DBG, add, record_id: ", record_id
    url_dict[record_id] = [name, url, note]
    return record_id, url_dict


def SaveDataToFile(record_id, url_dict):
    """
    Convert to binary file w/ pickle & write to disk.
    """
    with open('url_bookmarks.pickle', 'wb') as pf:
        pickle.dump((record_id, url_dict), pf)


def GetCommand(menu, record_id, url_dict, infile):
    """
    Ask user to enter a character, then check if it is a valid
    command, and execute it or show an error msg.
    """
    while True:
        try:
            user_input = raw_input(menu)
        except KeyboardInterrupt:
            print
            return
        if user_input == 'q' or user_input == 'Q':
            SaveDataToFile(record_id, url_dict)
            print "Saving data and quitting program."
            sys.exit(0)  
        elif user_input == 'a' or user_input == 'A':
            (record_id, url_dict) = AddNewEntry(record_id, url_dict)
        elif user_input == 'h' or user_input == 'H':
            PrintHelp()
        elif user_input == 's' or user_input == 'S':
            ShowAllRecords(url_dict, infile)
        elif user_input == 'd' or user_input == 'D':
            if url_dict:
                DeleteCurrentRecord(url_dict)
            else:
                print "WARN: No bookmarks to delete; Add bookmarks first."
                GetCommand(menu, record_id, url_dict, infile)
        elif user_input == 'u' or user_input == 'U':
            if url_dict:
                UpdateCurrentRecord(url_dict)
            else:
                print "WARN: No bookmarks to update; Add bookmarks first."
                GetCommand(menu, record_id, url_dict, infile)
        elif user_input == 'f' or user_input == 'F':
            FindWord(url_dict)
        else:
            print "Invalid request, not implemented: ", user_input
    return


def CheckRecord(url_dict):
    """Ask user for record to check.
    Print a warning if a problem is found.
    Otherwise, show the record at the given key and return the key.
    """
    key = 0
    while True:
        try:
            key = int(raw_input("Enter the record number to use: "))
            break  # Exit the while loop when a key is entered & no exception is caused.
        except (KeyError, UnboundLocalError):
            print "\nWARN1: Problem with this record number: ["+ str(key) + "]" 
            return False
        except (ValueError):
            print "\nWARN2: Problem with your input; enter an integer."
            return False
    if key in url_dict:
        PrintRecord(url_dict, key)
        return key
    else:
        print "\nWARN3: Problem with this record number: ["+ str(key) + "]" 
        return False


def DeleteCurrentRecord(url_dict):
    """Check for a valid record at the given key.
    Ask user to confirm delete; & delete it if OK.
    """
    key = CheckRecord(url_dict)
    if not key:
        return

    if raw_input("Press 'y' to confirm delete: ") == 'y':
        try:
            url_dict.pop(key)
        except KeyError:
            "Record number error, try again."
    else:
        print "Record was not deleted."


def UpdateCurrentRecord(url_dict):
    """Check for a valid record at the given key.
    Ask user to enter data & confirm update; & update it if OK.
    """
    key = CheckRecord(url_dict)
    if not key:
        return

    print "\nEnter your updated bookmark: "
    upd_name = raw_input("  Name: ")
    upd_url  = raw_input("  URL : ")
    upd_note = raw_input("  Note: ")

    print "\nUpdated bookmark will be:"
    print "  Name: ", upd_name
    print "  URL : ", upd_url
    print "  Note: ", upd_note

    if raw_input("Press 'y' to confirm update: ") == 'y':
        try:
            url_dict[key][0] = upd_name
            url_dict[key][1] = upd_url
            url_dict[key][2] = upd_note
        except KeyError:
            "Record number error, try again."
    else:
        print "Record was not updated."


def ShowAllRecords(url_dict, infile):
    """Print all records to the screen, w/ key & fields.
    """
    bookmark_count = 0
    for key in url_dict.keys():
        bookmark_count += 1
        PrintRecord(url_dict, key)
    print "Found %d records in data file %s." % (bookmark_count, infile)


def PrintRecord(url_dict, key):
    """Print one record to the screen, w/ key & fields.
    """
    print "%2d" % key,
    print "Name: ", url_dict[key][0]
    print "   URL : ",  url_dict[key][1]
    print "   Note: ", url_dict[key][2]


def FindWord(url_dict):
    """Ask user for a word to find in the bookmark list.
    Search for it and return all records that have it.
    """
    bookmark_count = 0
    num_hits = 0
    word = raw_input("Enter word to find in bookmark list: ")
    for key,v in url_dict.items():
        bookmark_count += 1
        for s in v:
            if word in s:
                #D print "DBG, fw, hit, v: ", v
                num_hits += 1
                PrintRecord(url_dict, key)
    print "Found %d hits in %d records." % (num_hits, bookmark_count)


def main():
    """Get i/p data, print the menu, & wait for user command.
    """
    record_id = 0
    infile = "url_bookmarks.pickle"
    url_dict = {}

    try:
        (record_id, url_dict) = GetFileData(infile)
    except TypeError:
        print "WARN: No data in file?  Add bookmarks to the file to continue."
    menu = GetMenu()
    GetCommand(menu, record_id, url_dict, infile)

if __name__=='__main__':
    main()


