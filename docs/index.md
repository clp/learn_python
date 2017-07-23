---
title: Some data files   Sun2017_0723_10:08 
---

<ul class="list pa0">
  {% for fi in site.data %}
  <li class="mv2">
    <a href="{{ site.url }}{{ fi.url }}" class="db pv1 link blue hover-mid-gray">
      <time class="fr silver ttu">{{ fi.date | date_to_string }} </time>
      {{ fi.title }}
    </a>
  </li>
  {% endfor %}
</ul>

* [all_qa_with_hst.html](/data/all_qa_with_hst.html)

  Question followed by its Answers; then next Q&A.  Includes High Score Terms
  found for each record.


* [q_with_a.csv](/data/q_with_a.csv)

  All Questions followed by all Answers.
