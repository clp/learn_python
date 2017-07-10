README

| Basic data about the project and code.
| See README_2 for more details.

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

 * fga_find_good_answers.py (fga), main control program.
 * grade_each_answer.py, a user can read & grade answers, from A to F.
 * nltk_ex25.py, analyze text with natural language toolkit.




Program: **fga_find_good_answers.py**
============================================

Introduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fga_find_good_answers program
formats the input data from stackoverflow.com
into question-and-answer groups, Q&A groups:
one question followed by its related answers,
followed by the next Q&A group.

It then analyzes the data to find good answers,
using statistical techniques or
natural language processing, NLP.


Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start the program with one of these commands:
  * ``./fga_find_good_answers.sh``
  * ``python fga_find_good_answers.py``

Respond to the prompt with 'm' to see the menu.
Enter 'q' to quit the program.


Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. The main o/p data is saved at outdir/q_with_a.csv.

#. One way to examine the data: open the file with
   LibreOffice Calc or other spreadsheet tool that can show
   the full content of each cell.

#. The program writes some data to the log file, fl_fga.log.



Program: **nltk_ex25.py**
================================================

Introduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The nltk_ex25 program is an experiment to learn about text processing
using the Natural Language Toolkit, NLTK.
It processes Q & A data from stackoverflow
(and the use of NLTK by this program needs more than
one answer for a question).

One way that it can identify useful answers is
based on the terms that they contain
compared to terms that are found in high-score answers.

The o/p is a csv file containing a question followed
by the selected answers.


Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At this time, the natural language processing routines
included in the nltk_ex25.py program
are called from the main fga*.py program.
You need not run the nltk*.py program separately.

For debugging and development,
you can run this program directly, eg:

 * ``python nltk_ex25.py``


Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. The o/p data of answers that contain high score terms (hst) is saved
   at tmpdir/all_ans_with_hst.csv

#. The program writes some summary data to the screen,
   to help with debugging.

#. The program writes some data to the log file, fl_fga.log.


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

See README_2 for more details.


Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Consider opening the i/p csv data file in a separate
   terminal or window
   for easier viewing of the Q&A being graded.
   Eg, using the surf web browser:
     
      ``surf outdir/graded_q_with_a.csv``

#. If you finish handling all records in the i/p file,
   the program saves data and stops.
   If some answers were ignored and are graded 'N',
   they will be shown for grading when you next run the program.


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

``ans_df = pandas.read_csv('Answers.csv', encoding='latin-1', warn_bad_lines=False, error_bad_lines=False)``


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


Mon2017_0710_11:40 
