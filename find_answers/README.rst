README

Program: **fga_find_good_answers.py**

Introduction
------------

This project's goal is to learn Python by building tools 
that analyze text data to identify good quality answers
to technical questions.

The first release of the
fga_find_good_answers program uses pandas and numpy.
It processes data from stackoverflow.com
about the Python programming language,
and formats it for later analysis.

The o/p is a csv file containing questions followed
by the related answers.
Later analysis can be done with tools built
for natural language processing, NLP.


Usage
------------

``python fga_find_good_answers.py``


Notes
-----

1. The number of o/p records can be varied by changing
   the num_owners variable.
   Original setting of num_owners is 10.

  * When using the full Questions.csv and Answers.csv i/p files,
    the o/p file was 8749 lines long and was built in 36 sec
    on my system.

  * With num_owners set to 50 for the same data set, 
    the o/p file was 22023 lines long and was built in 59 sec.

2. The o/p data is saved at outdir/q_with_a.csv.

3. One way to use the data: run the grade_each_answer.py program
   which uses the q_with_a.csv file.  Read the question & answer
   and grade the answer.  That grade is saved and available to
   compare with grades from other analysis tools.

   * Another way to examine the data: open the file with
     LibreOffice Calc or other spreadsheet tool that can show
     the full content of each cell.

-------------



Program: **nltk_ex25.py**

Introduction
------------

The nltk_ex25 program is an experiment to learn about text processing
using the Natural Language Toolkit, NLTK.
It processes data for one question with several answers
from stackoverflow.com.
It identifies answers that might contain useful data,
based on the terms that they contain
(terms that are found in high-score answers).

The o/p is a csv file containing a question followed
by the selected answers.


Usage
------------

``python nltk_ex25.py``


Notes
-----

1. The number of high-score terms used for comparison can be varied
   by changing
   the num_hi_score_terms variable.
   Original setting of num_hi_score_terms is 22.

2. The program is now hard-coded to use a single,
   specific input file during initial debugging.
   That file is derived from the output
   of fga_find_good_answers.py,
   and is stored in the outdir/ directory of this repository.

3. The o/p data of Answers that contain HiScoreTerms is saved
   at tmpdir/ans_with_hst.csv

4. Open the file with LibreOffice Calc
   or other spreadsheet tool to review the data visually.

------------


Program: **grade_each_answer.py**

Introduction
------------

The grade_each_answer program is a utility tool to add new fields
and their content to each record of the input file,
which by default is q_with_a.csv.

The o/p is a csv file containing two new fields for each i/p
record: Grade and Notes.


Usage
------------

``python grade_each_answer.py``


Notes
-----

1. The program uses a simple menu at the command line.  Enter 'm'
   to show the menu; or 'h' to see help; 'q' to quit.

2. TBD

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


Tue2017_0404_15:44  
