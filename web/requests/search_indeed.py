#!/usr/bin/python
# -*- coding: utf-8-*- 

'''search_indeed.py - Show jobs found at indeed.com.

Some search parameters are hard-coded: query string,
location, limit (number of records to show).
'''

# Example of using the requests module to get data;
# and using regex to parse the HTML response page.

import re
import requests
import time

site = 'indeed.com'
query = 'python+entry'
location = '94063'
limit = '10'  # Number of records to retrieve; maybe max is 100.

url = 'http://' + site + '/jobs?q=' + query + '&l=' + location + '&limit=' + limit

resp = requests.get(url)

lines = resp.content.split("\n")

# Example Jobmap field, Sun2016_0417
# jobmap[0]= {jk:'abdd6bbcacf2a8e0',efccid: 'de7f1e9cd569189f',srcid:'02b7985ab68935b2',cmpid:'007d012feca84599',num:'0',srcname:'Springpath',cmp:'Springpath',cmpesc:'Springpath',cmplnk:'/q-Springpath-l-94063-jobs.html',loc:'Sunnyvale, CA',country:'US',zip:'',city:'Sunnyvale',title:'Entry Level Software QA Engineer',locid:'62f8f620fe0679b7',rd:'BvNCMOi6ebTuqTtC1vLQYw'};


now = time.ctime()

print "Jobs from %s; query: %s; location: %s; %s\n" % (site, query, location, now)

for j in lines:
    jobmap = r"^jobmap.*jk:'(.*?)'.*srcname:'(.*?)'.*loc:'(.*?)'.*title:'(.*?)'.*$"
    regex = re.compile(jobmap)
    so = regex.search(j)
    if (so):
        jobkey = so.group(1)
        srcname = so.group(2)
        loc = so.group(3)
        title = so.group(4)
        print title
        print "    %s :: %s :: %s " % (srcname, loc, jobkey)
        print
