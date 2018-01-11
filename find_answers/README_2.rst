README_2

| Some details about the project and code.
| See README for basic data.


Project: **Find Good Answers**
----------------------------------------------


Workflow
=====================================

After installing the software into the 'working_dir',
follow these steps.

First Run
~~~~~~~~~~~~~~~~~~~~~~

  #. The software includes sample data files,
     and the fga program is hard-coded to use one pair of them.

     TBD, To use the program with a different data set,
     obtain the data,
     convert data to the required format if needed,
     place files in the <working_dir>/indir/ directory,
     and edit the program to use them.

  #. Run the fga program, which will create the main Q&A file
     data/popular_qa.csv; and may create several other files.

     The main Q&A file can be used by other tools.

Subsequent Runs
~~~~~~~~~~~~~~~~~~~~~~

  #. To study how different settings affect the output,
     edit the fga program; see the source code for details.

  #. Run the fga program, which creates a working data file.

  #. Optional: Run grade_each_answer.py to show each question and
     answer on screen;
     to prompt the user for a grade for the answer;
     and to save that grade with the record.

     If there are dozens or hundreds of answers in the data set,
     grading them can take a person many hours.
     This work need be done only once for a data set, and can be done
     over a period of time.
     The stored grades are recalled each time the program runs,
     and only ungraded answers are presented for grading by the user.

     One approach is to use a small subset of the data for analysis
     so they can be graded quickly.
     Maybe start with a subset of high priority questions,
     based on popularity:
     questions with many answers or questions and answers with
     high scores.

     TBD, After answers are graded,
     a tool can compare
     the evaluation by the program
     to your opinion of each answer.
     This is one way
     to evaluate how well the program finds good answers.
     There is no such tool in this project at this time.

  #. TBD, To find what program settings produce good results:
     review results; change the program; repeat tests.



Input data format
=====================================

TBD



Output data format, presentation
=====================================

TBD



Program: **fga_find_good_answers.py**
============================================

Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. The number of o/p records can be varied by changing
   the MAX_OWNERS variable.
   Original setting of MAX_OWNERS is 10.

  * When using the full Questions.csv and Answers.csv i/p files,
    with MAX_OWNERS set to 10, 
    the o/p file was 8749 lines long and was built in 36 sec
    on my system.

  * With MAX_OWNERS set to 50 for the same data set, 
    the o/p file was 22023 lines long and was built in 59 sec.

#. One way to use the data: run the grade_each_answer.py program
   which uses the popular_qa.csv file.  Read the question & answer
   and give the answer a grade based on its value.  That grade
   is saved and available to compare with grades from other
   analysis tools.





Module: **nltk_ex25.py**
================================================

Introduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The nltk_ex25 module is an experiment to learn about text processing
using the Natural Language Toolkit, NLTK.
It processes Q & A data from stackoverflow
(needs more than one answer for a question).

One way that it can identify useful answers is
based on the terms that they contain
compared to terms that are found in high-score answers.

The o/p is a csv file containing a question followed
by the selected answers.


Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At this time, the natural language processing routines
included in the nltk_ex25.py module
are called from the main fga*.py program.
Do not run the nltk*.py module separately.



Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. The number of high-score terms used for comparison can be varied
   by changing
   the MAX_HI_SCORE_TERMS variable in fga*py.




Program: **grade_each_answer.py**
================================================

Introduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The grade_each_answer program is a utility tool to add new fields
and their content to each record of the input file.

The o/p is a csv file containing two new fields for each i/p
record: Grade and Notes.


Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``python grade_each_answer.py``


The program reads the i/p file and
shows the first answer that is not yet graded,
with the related question.
It prompts for a command;
press 'm' to see the menu
and 'h' for help.

The menu has these choices::

    Menu choices to grade an answer:
    a: excellent value
    b: very good value
    c: good value
    d: fair value
    e: poor value
    f: no value
    i: ignore this item for now; leave its grade 'N' for none
    u: unknown value; skip it for now, evaluate it later
    .........................................................

    Other menu items:
    h, ?: show help text, the menu
    m: show menu
    q: save data and quit the program
    s: show question & answer

If you enter a grade (a,b,c,d,e,f),
it prompts for a comment and saves that text into the Notes field
for that record; pressing Enter terminates the note text.
The next ungraded answer is then shown.

Enter 'i' to ignore this answer for now.
Its grade remains set to 'N' (for no grade),
and it can be seen the next time the program is run.
The next ungraded answer is then shown.

Enter 'u' to mark this answer as 'Unknown value' for now.
Enter a comment if needed.
This answer will not be shown when the program is run and cannot be
easily changed.
Use this grade for answers whose value you cannot judge.
The next ungraded answer is then shown.

Enter 's' to show the current question & answer.
Use this command after looking at the menu or help,
to see the answer to be graded.

Enter 'q' to save data and quit the program.
The output goes to data/graded_popular_qa.csv.


Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. To change a grade or note,
   open the grading file with a tool that can read and write
   CSV data files
   (eg, a text editor or a spreadsheet),
   and make the change.
   Be careful not to corrupt the CSV format.

   If the file is large,
   you might not be able to easily edit it with a tool
   that brings the entire file into memory,
   and it might operate slowly.

   Suggestions to edit large files include LargeFile plugin for vim;
   the 'split' command to break a large file into smaller chunks,
   then concatenate them after editing;
   the 'grep', 'awk', and 'sed' commands.


