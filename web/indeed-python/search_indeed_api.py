# -*- coding: utf-8-*-

'''
search_indeed_api.py - Retrieve some job records from
indeed.com, based on the query parameters.

User must provide a valid indeed.com PUBLISHER_NUMBER and some
parameters.  See this page to create an account (it was free and
quick for me):
  http://www.indeed.com/publisher/

Use a job key to find more data by running the
get_details_indeed_api.py program.
'''


from indeed import IndeedClient

client = IndeedClient('PUBLISHER_NUMBER')

params = {
    'q' : "python entry",
    'l' : "94063",
    'userip' : "1.2.3.4",
    'useragent' : "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_2)"
}

search_response = client.search(**params)

# Example search response, Mon2016_0418_22:51.
# Entire response is a dictionary.
# The dict key 'results' has a list of dicts for its value.
# Each dict in the list is one job record, as partly shown here:
'''

{u'formattedRelativeTime': u'30+ days ago',
u'city': u'Redwood City', u'date': u'Wed, 03 Feb 2016 02:33:06 GMT',
...
u'url': u'http://www.indeed.com/viewjob?jk=52d202269106777a&qd=...
...
u'jobtitle': u'Senior Cloud Operations/ Software Engineer',
u'company': u'Phoenix 2.0', u'onmousedown': u"indeed_clk(this, '4903');",
...}, 

'''
# After the list of dicts of job data, there are a few more 
# key:value data.


jobs_l = search_response['results']

keys_l = ['jobtitle', 'company', 'city', 'date', 'snippet', 'jobkey']

for i,j in enumerate(jobs_l):
    for k in keys_l:
        print search_response['results'][i][k]
    print

