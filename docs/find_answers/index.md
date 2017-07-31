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
and using Natural Language Programming (NLP) techniques.
The Python Natural Language Toolkit (NLTK)
provides good access to NLP tools.

This site contains some data files that were built
from the software.

See the software for the find_answers project at
[github.](https://github.com/clp/learn_python/tree/master/find_answers)

## Some data files.

* [all_qa_with_hst.html](all_qa_with_hst.html)

  Question followed by its Answers; then next Q&A.  
  Includes High Score Terms (hst)
  found for each record.


* [all_qa_title_body.html](all_qa_title_body.html)

  Each Question followed by its Answers; only shows ID, Title, and Body
  columns.

* [qa_stats.html](qa_stats.html)

  A table of each Question followed by its Answers;
  shows ID, Owner, ParentId,
  Score, Body Length, Owner Reputation, and hstCount
  (high score terms count) columns.

* [scatter matrix plot, a3data, pdf](scat_mat_plot_4x4_a3data.pdf)
* [scatter matrix plot, a3data, png](scat_mat_plot_4x4_a3data.png)

  A 4x4 matrix of scatter plots (and histograms) of
  Score, Body Length, Owner Reputation, and hstCount data.

