---
title: Find Answers
---

## Introduction

One goal of this project is to learn Python by building tools
that analyze text data. The first data sources are questions and
answers from stackoverflow.com about Python. The software tries
to identify good but 'hidden' answers to the questions. Such
answers have useful data but a low score.

The input data is analyzed using statistical techniques,
and using Natural Language Processing (NLP) techniques.
The Python Natural Language Toolkit (NLTK)
provides good access to NLP tools.

This site contains some output files that were built
from the software.

See the software for the find_answers project at
[github.](https://github.com/clp/learn_python/tree/master/find_answers)

## Some output tables and plots.

These sample files were made by processing two small input files
(q3_992.csv and a3_986.csv) with fga_find_good_answers.py,
unless otherwise noted.

* [popular_qa.html](popular_qa.html)

  Question followed by its Answers; then next Q&A.  
  Includes High Score Terms (hst)
  found for each record.


* [popular_qa_title_body.html](popular_qa_title_body.html)

  Each Question followed by its Answers; only shows ID, Title, and Body
  columns.

* [qa_stats.html](qa_stats.html)

  Numeric data only;
  it does not include text of Questions or Answers.

  A table of each Question followed by its Answers;
  shows ID, Owner, ParentId,
  Score, Body Length, Owner Reputation, and hstCount
  (high score terms count) columns.

* [search_result_full.html](search_result_full.html)

  Output of the search functionality.

  The subset of the input data that contains only Questions
  and Answers with the keyword,
  and their related Q's and A's.

  A table of each Question followed by its Answers;
  shows
  hstCount
  (high score terms count)
  Score, ID, Title, Body
  columns.

  The answers under each question are sorted by hstCount,
  then Score, then Id.

  The questions on the page are sorted by hstCount
  for the answers; thus the first Q has the A with the highest
  hstCount for all A's on the page.

  Since only Q's have Titles, it is easy to see where each
  Q&A group starts (with text in the Title column), while all
  the related A's have NaN in the title column.

* [scatter matrix plot, a3data, pdf](scat_mat_plot_4x4_a3data.pdf)
* [scatter matrix plot, a3data, png](scat_mat_plot_4x4_a3data.png)

  One plot in two file formats.

  A 4x4 matrix of scatter plots (and histograms) of
  Score, Body Length, Owner Reputation, and hstCount data.

