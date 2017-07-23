---
title: Some data files   Sun2017_0723_11:35 
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

# learn_python
Notes, data, tips, code, resources for learning Python.

See the project source at
[github](https://github.com/clp/learn_python).
See the
[project web site here](https://clp.github.io/learn_python).

See the
[Wiki](https://github.com/clp/learn_python/wiki)
for some details.

This repository contains several projects.
To learn about the
find_answers
project,
see its
[source at github here](https://github.com/clp/learn_python/tree/master/find_answers);
and see its
[web site here](https://clp.github.io/learn_python).
See some docs & links
[here](https://clp.github.io/learn_python//find_answers/find_answers.md).

-----

Experimental link to fga dir:
[fga](https://clp.github.io/learn_python//find_answers/find_answers.md).
<br />
[fga relative link](/find_answers/find_answers.md).
<br />
[fga README.md relative link](/find_answers/README.md).
