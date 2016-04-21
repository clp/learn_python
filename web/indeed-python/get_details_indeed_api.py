# -*- coding: utf-8-*-

'''
get_details_indeed_api.py - Retrieve details of specific jobs
from indeed.com, based on the job key.

User must provide a PUBLISHER_NUMBER and one or more job keys.

Get each job key by running the search_indeed_api.py program.
'''

from indeed import IndeedClient
import time

site = 'indeed.com'

client = IndeedClient('PUBLISHER_NUMBER')

# Job keys copied from the response to the search program:
jk_l = ["e8930d8d162c4b70", "6bb8f41ea97bd6f8"]

job_response = client.jobs(jobkeys = (jk_l ))

# Example job response, Mon2016_0418.
#   Entire response is a dictionary.
#   The key 'results' has value of a list of dicts, eg:
#   {u'version': 2, u'results': [{u'formattedRelativeTime': u'30+ days ago', ...
#
# Each dict in the list is one job record, as shown here:
# {u'formattedRelativeTime': u'30+ days ago', u'city': u'San Francisco', u'date': u'Tue, 01 Mar 2016 02:31:38 GMT', u'latitude': 37.774727, u'url': u'http://www.indeed.com/rc/clk?jk=e8930d8d162c4b70&atk=', u'jobtitle': u'Jr Data Engineer', u'company': u'Komodo Health', u'formattedLocationFull': u'San Francisco, CA', u'longitude': -122.41758, u'onmousedown': u"indeed_clk(this, '');", u'snippet': u'Focused on using the best of what lies at the forefront of technology and data science to address answer complex, real-world problems in the Healthcare and Life Science space. BS or MS in Computer Science or a related field. Develop tools to monitor, debug and analyze data pipelines. We are looking for a Data Engineer to help build out our data infrastructure using the best tools available so that...', u'source': u'Komodo Health', u'state': u'CA', u'sponsored': False, u'country': u'US', u'formattedLocation': u'San Francisco, CA', u'jobkey': u'e8930d8d162c4b70', u'recommendations': [], u'expired': False, u'indeedApply': True}

# TBD, after the list of dicts of job data, there are a few more k:v data


jobs_l = job_response['results']

keys_l = ['jobtitle', 'company', 'city', 'date', 'snippet', 'jobkey']

now = time.ctime()

print "Job details from %s using the API; %s\n" % (site, now)

for i,j in enumerate(jk_l):
    for k in keys_l:
        print job_response['results'][i][k]
    print

