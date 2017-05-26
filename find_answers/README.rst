README

Project: **Find Good Answers**
----------------------------------------------

Introduction
=====================================

One goal of this project is to learn Python by building tools
that analyze text data.
The first data sources are 
questions and answers from stackoverflow.com about Python.
The software tries
to identify good but 'hidden' answers
to the questions.
Such answers have useful data but a low score.

It includes the following programs.

 * fga_find_good_answers.py (fga)
 * grade_each_answer.py
 * nltk_ex25.py


Workflow
=====================================

After installing the software into the 'working_dir',
follow these steps.

First Run
~~~~~~~~~~~~~~~~~~~~~~

  #. The software includes sample data files,
     and the program is hard-coded to use one pair of them.

     TBD, To use the program with a different data set,
     obtain the data,
     convert data to the required format if needed,
     place files in the <working_dir>/indir/ directory,
     and edit the program to use them.

  #. Run the fga program, which will create the main Q&A file
     outdir/q_with_a.csv; and may create several other files.

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

     TBD, At least one data file in the package includes
     graded answers.

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

Introduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fga_find_good_answers program
formats the input data from stackoverflow.com
into each question followed by the related answers.

TBD, It then analyzes the data to find good answers,
using statistical techniques or
natural language processing, NLP.


Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``python fga_find_good_answers.py``


Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. The number of o/p records can be varied by changing
   the num_owners variable.
   Original setting of num_owners is 10.

  * When using the full Questions.csv and Answers.csv i/p files,
    the o/p file was 8749 lines long and was built in 36 sec
    on my system.

  * With num_owners set to 50 for the same data set, 
    the o/p file was 22023 lines long and was built in 59 sec.

2. The main o/p data is saved at outdir/q_with_a.csv.

3. One way to use the data: run the grade_each_answer.py program
   which uses the q_with_a.csv file.  Read the question & answer
   and give the answer a grade based on its value.  That grade
   is saved and available to compare with grades from other
   analysis tools.

   * Another way to examine the data: open the file with
     LibreOffice Calc or other spreadsheet tool that can show
     the full content of each cell.




Program: **nltk_ex25.py**
================================================

Introduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The nltk_ex25 program is an experiment to learn about text processing
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

``python nltk_ex25.py``


Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. The number of high-score terms used for comparison can be varied
   by changing
   the num_hi_score_terms variable.
   Original setting of num_hi_score_terms is 22.

2. The program is now hard-coded to use a single,
   specific input file during initial debugging,
   outdir/pid_231767.csv.

3. The o/p data of answers that contain HiScoreTerms is saved
   at tmpdir/ans_with_hst.csv

4. Open the file with LibreOffice Calc
   or other spreadsheet tool to review the data.



Program: **grade_each_answer.py**
================================================

Introduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The grade_each_answer program is a utility tool to add new fields
and their content to each record of the input file
(q_with_a.csv).

The o/p is a csv file containing two new fields for each i/p
record: Grade and Notes.


Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``python grade_each_answer.py``


The program reads the i/p file and
shows the first answer that is not yet graded,
with its question.
It prompts for a command;
press 'm' to see the menu
and 'h' for help.

The menu has these choices::

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
    s: show question & answer

If you enter a grade (a,b,c,d,f),
it prompts for a comment and saves that text into the Notes field
for that record; pressing Enter terminates the note text.
The next ungraded answer is then shown.

Enter 'i' to ignore this answer for now.
Its grade remains set to 'N' (for no grade),
and it can be seen the next time the program is run.
The next ungraded answer is then shown.

Enter 'u' to mark this answer as 'Unknown value' for now.
Enter a comment if needed.
It will not be shown when the program is run and cannot be
easily changed.
Use this grade for answers whose value you cannot judge.
The next ungraded answer is then shown.

Enter 's' to show the current question & answer.
Use this command after looking at the menu or help,
to see the Q&A for grading.

Enter 'q' to save data and quit the program.
The output goes to outdir/graded_q_with_a.csv.


Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. If you finish handling all records in the i/p file,
   the program saves data and stops.
   If some answers were ignored and are graded 'N',
   they will be shown for grading when you next run the program.

#. To change a grade or note,
   open the grading file with a tool that can read and write
   CSV data files, and make the change.
   Be careful not to corrupt the CSV format.

   If the file is large,
   you might not be able to easily edit it with a tool
   that brings the entire file into memory,
   and it might operate slowly.

   Suggestions to edit large files include LargeFile plugin for vim;
   the 'split' command to break a large file into smaller chunks,
   then concatenate them after editing;
   the 'grep', 'awk', and 'sed' commands.

------------


FAQ
------------

**What is stackoverflow.com?**

SO is a question-and-answer web site.
Registered users can enter data and vote on questions and
answers,
so that higher-quality contributions might be identified.


**What is kaggle.com?**

Kaggle is a web site for learning about data science by using
documentation
and participating in competitions.
You can download data sets from the site.
One data set that is used in this project
is a collection of questions
and answers from stackoverflow about python.


**How to read the stackoverflow data?**

Use the Python pandas module, read_csv().

``ans_df = pd.read_csv('Answers.csv', encoding='latin-1', warn_bad_lines=False, error_bad_lines=False)``


**How to find answers with low scores that are high quality?**

That's one goal of this project.
One way might be to identify some unique properties of high score answers,
and find low score answers with the same or similar properties.


**What is the Natural Language Toolkit, NLTK?**

NLTK is a platform (code, documents, data sets, and more)
for building s/w to work with human language data.
For documentation, please visit nltk.org.

* https://nltk.org
* https://github.com/nltk/nltk


**What are some other useful sites and resources to check?**

* https://github.com/gleitz/howdoi
  A CLI tool that gets answers from stackoverflow.

* https://worksheets.codalab.org/


Fri2017_0526_12:47 
