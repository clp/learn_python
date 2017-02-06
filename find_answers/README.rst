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

3. One way to use the data: open the file with LibreOffice Calc
   or other spreadsheet tool that can show the full content of
   each cell.  Use that tool to mark records that are valuable,
   and worth more detailed analysis.


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


**What are some other useful sites and resources to check?**

* https://github.com/gleitz/howdoi
  A CLI tool that gets answers from stackoverflow.

* https://worksheets.codalab.org/


Sun2017_0205_23:32 
