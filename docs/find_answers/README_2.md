README\_2

Some details about the project and code.

See README for basic data.

Project: **Find Good Answers**
==============================

Workflow
--------

After installing the software into the 'working\_dir', follow these steps.

### First Run

> 1.  The software includes sample data files, and the fga program is hard-coded to use one pair of them.
>
>     TBD, To use the program with a different data set, obtain the data, convert data to the required format if needed, place files in the &lt;working\_dir&gt;/indir/ directory, and edit the program to use them.
>
> 2.  Run the fga program, which will create the main Q&A file outdir/q\_with\_a.csv; and may create several other files.
>
>     The main Q&A file can be used by other tools.
>
### Subsequent Runs

> 1.  To study how different settings affect the output, edit the fga program; see the source code for details.
>
> 2.  Run the fga program, which creates a working data file.
>
> 3.  Optional: Run grade\_each\_answer.py to show each question and answer on screen; to prompt the user for a grade for the answer; and to save that grade with the record.
>
>     If there are dozens or hundreds of answers in the data set, grading them can take a person many hours. This work need be done only once for a data set, and can be done over a period of time. The stored grades are recalled each time the program runs, and only ungraded answers are presented for grading by the user.
>
>     One approach is to use a small subset of the data for analysis so they can be graded quickly. Maybe start with a subset of high priority questions, based on popularity: questions with many answers or questions and answers with high scores.
>
>     TBD, After answers are graded, a tool can compare the evaluation by the program to your opinion of each answer. This is one way to evaluate how well the program finds good answers.
>
>     TBD, At least one data file in the package includes graded answers.
>
> 4.  TBD, To find what program settings produce good results: review results; change the program; repeat tests.
>
> 5.  TBD, To analyze the text run nltk\_ex25.py.
>
Input data format
-----------------

TBD

Output data format, presentation
--------------------------------

TBD

Program: **fga\_find\_good\_answers.py**
----------------------------------------

### Notes

1.  The number of o/p records can be varied by changing the num\_owners variable. Original setting of num\_owners is 10.

> -   When using the full Questions.csv and Answers.csv i/p files, the o/p file was 8749 lines long and was built in 36 sec on my system.
> -   With num\_owners set to 50 for the same data set, the o/p file was 22023 lines long and was built in 59 sec.

1.  One way to use the data: run the grade\_each\_answer.py program which uses the q\_with\_a.csv file. Read the question & answer and give the answer a grade based on its value. That grade is saved and available to compare with grades from other analysis tools.

Program: **nltk\_ex25.py**
--------------------------

### Introduction

The nltk\_ex25 program is an experiment to learn about text processing using the Natural Language Toolkit, NLTK. It processes Q & A data from stackoverflow (needs more than one answer for a question).

One way that it can identify useful answers is based on the terms that they contain compared to terms that are found in high-score answers.

The o/p is a csv file containing a question followed by the selected answers.

### Usage

At this time, the natural language processing routines included in the nltk\_ex25.py program are called from the main fga\*.py program. You need not run the nltk\*.py program separately.

For debugging and development, you can run this program directly, eg:

> -   `python nltk_ex25.py`

### Notes

1.  The number of high-score terms used for comparison can be varied by changing the num\_hi\_score\_terms variable. Original setting of num\_hi\_score\_terms is 22.
2.  The program is now hard-coded to use a single, specific input file during initial debugging, outdir/pid\_231767.csv. When run directly, it will use this data set by default. When called via fga\*.py, it will use the data provided by fga.

Program: **grade\_each\_answer.py**
-----------------------------------

### Introduction

The grade\_each\_answer program is a utility tool to add new fields and their content to each record of the input file (q\_with\_a.csv).

The o/p is a csv file containing two new fields for each i/p record: Grade and Notes.

### Usage

`python grade_each_answer.py`

The program reads the i/p file and shows the first answer that is not yet graded, with the related question. It prompts for a command; press 'm' to see the menu and 'h' for help.

The menu has these choices:

``` literal-block
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
```

If you enter a grade (a,b,c,d,e,f), it prompts for a comment and saves that text into the Notes field for that record; pressing Enter terminates the note text. The next ungraded answer is then shown.

Enter 'i' to ignore this answer for now. Its grade remains set to 'N' (for no grade), and it can be seen the next time the program is run. The next ungraded answer is then shown.

Enter 'u' to mark this answer as 'Unknown value' for now. Enter a comment if needed. This answer will not be shown when the program is run and cannot be easily changed. Use this grade for answers whose value you cannot judge. The next ungraded answer is then shown.

Enter 's' to show the current question & answer. Use this command after looking at the menu or help, to see the answer to be graded.

Enter 'q' to save data and quit the program. The output goes to outdir/graded\_q\_with\_a.csv.

### Notes

1.  To change a grade or note, open the grading file with a tool that can read and write CSV data files (eg, a text editor or a spreadsheet), and make the change. Be careful not to corrupt the CSV format.

    If the file is large, you might not be able to easily edit it with a tool that brings the entire file into memory, and it might operate slowly.

    Suggestions to edit large files include LargeFile plugin for vim; the 'split' command to break a large file into smaller chunks, then concatenate them after editing; the 'grep', 'awk', and 'sed' commands.

Tue2017\_0613\_20:22


