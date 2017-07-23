README

Basic data about the project and code.

See README\_2 for more details.

Project: **Find Good Answers**
==============================

Introduction
------------

One goal of this project is to learn Python by building tools that analyze text data. The first data sources are questions and answers from stackoverflow.com about Python. The software tries to identify good but 'hidden' answers to the questions. Such answers have useful data but a low score.

It includes the following programs.

> -   fga\_find\_good\_answers.py (fga), main control program.
> -   nltk\_ex25.py, analyze text with natural language toolkit.
> -   grade\_each\_answer.py, a user can read & grade answers, from A to F.

Program: **fga\_find\_good\_answers.py**
----------------------------------------

### Introduction

The fga\_find\_good\_answers program formats the input data from stackoverflow.com into question-and-answer groups, Q&A groups: one question followed by its related answers, followed by the next Q&A group.

It then analyzes the data to find good answers, using statistical techniques or natural language processing, NLP.

### Usage

Start the program with one of these commands:  
-   `./fga_find_good_answers.sh`
-   `python fga_find_good_answers.py`

Respond to the prompt with 'm' to see the menu. Enter 'q' to quit the program.

### Notes

1.  The main o/p data is saved at outdir/q\_with\_a.csv.
2.  One way to examine the data: open the file with LibreOffice Calc or other spreadsheet tool that can show the full content of each cell.
3.  The program writes some data to the log file, fl\_fga.log.

Program: **nltk\_ex25.py**
--------------------------

### Introduction

The nltk\_ex25 program is an experiment to learn about text processing using the Natural Language Toolkit, NLTK. It processes Q & A data from stackoverflow (and the use of NLTK by this program needs more than one answer for a question).

One way that it can identify useful answers is based on the terms that they contain compared to terms that are found in high-score answers.

The o/p is a csv file containing a question followed by the selected answers.

### Usage

At this time, the natural language processing routines included in the nltk\_ex25.py program are called from the main fga\*.py program. You do not run the nltk\*.py program directly.

### Notes

1.  The o/p data of answers that contain high score terms (hst) is saved at tmpdir/all\_qa\_with\_hst.csv
2.  The program writes some summary data to the screen, to help with debugging.
3.  The program writes some data to the log file, fl\_fga.log.

Program: **grade\_each\_answer.py**
-----------------------------------

### Introduction

The grade\_each\_answer program is a utility tool to add new fields and their content to each record of its input file (q\_with\_a.csv).

The o/p is a csv file containing two new fields for each i/p record: Grade and Notes.

### Usage

`python grade_each_answer.py`

See README\_2 for more details.

### Notes

1.  Consider opening the i/p csv data file in a separate window for easier viewing of the Q&A being graded. Eg, using the surf web browser:

    > `surf outdir/graded_q_with_a.csv`

2.  If you finish handling all records in the i/p file, the program saves data and stops. If some answers were ignored and are graded 'N', they will be shown for grading when you next run the program.

------------------------------------------------------------------------

FAQ
===

**What is stackoverflow.com?**

SO is a question-and-answer web site. Registered users can enter data and vote on questions and answers, so that higher-quality contributions might be identified.

**What is kaggle.com?**

Kaggle is a web site for learning about data science by using documentation and participating in competitions. You can download data sets from the site. One data set that is used in this project is a collection of questions and answers from stackoverflow about python.

**How to read the stackoverflow data?**

Use the Python pandas module, read\_csv().

`ans_df = pandas.read_csv('Answers.csv', encoding='latin-1', warn_bad_lines=False, error_bad_lines=False)`

**How to find answers with low scores that are high quality?**

That's one goal of this project. One way might be to identify some unique properties of high score answers, and find low score answers with the same or similar properties.

**What is the Natural Language Toolkit, NLTK?**

NLTK is a platform (code, documents, data sets, and more) for building s/w to work with human language data. For documentation, please visit nltk.org.

-   <a href="https://nltk.org" class="uri" class="reference external">https://nltk.org</a>
-   <a href="https://github.com/nltk/nltk" class="uri" class="reference external">https://github.com/nltk/nltk</a>

**What are some other useful sites and resources to check?**

-   <a href="https://github.com/gleitz/howdoi" class="uri" class="reference external">https://github.com/gleitz/howdoi</a> A CLI tool that gets answers from stackoverflow.
-   <a href="https://worksheets.codalab.org/" class="uri" class="reference external">https://worksheets.codalab.org/</a>

Sat2017\_0715\_13:37


