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

  Question followed by its Answers; then next Q&A.  Includes High Score Terms
  found for each record.


* [all_qa_title_body.html](all_qa_title_body.html)

  Each Question followed by its Answers; only shows ID, Title, and Body
  columns.
