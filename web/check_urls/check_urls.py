#!/usr/bin/env python3
# vim: ft=python fileencoding=utf-8 sts=4 sw=4 et:

#
# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with it.  If not, see <http://www.gnu.org/licenses/>.

"""Query http URLs found in an asciidoc file."""

import os
import re
from socket import timeout
import sys
import urllib.request
import urllib.error

infile = '/tmp/readme.asciidoc'
wait = 3  # Timeout for web query.

def check_urls(infile, wait):
    """Extract http URLs from a file and query each address.

    Goal: Find broken URLs & non-responsive sites.

    Read i/p file, w/ links in asciidoc form:
        http://path/to/page[name_of_anchor]

    Extract each http URL using a very simple regex.
    The full URL must be on a single line.

    Send a request to each URL and print the responses.
    """

    test_lines = [
        "* http://clp.github.io/learn_python/find_answers/[find_ans] (fga)",
        "* https://github.com/clp/learn_python/[learn_python] (Home)",
    ]

    # Read i/p file & build list of strings; or use test data if no file.
    try:
        with open(infile, 'r') as f:
            lines = f.readlines()
    except IOError as e:
        print("ERR ({}; use test data)".format(e))
        lines = test_lines

    # Read data & build list of http urls to check.
    total_urls = 0
    urls = []
    for line in lines:
        s = re.search(r'(http.*?)\[', line)
        if s:
            urls.append(s.group(1))
            total_urls += 1

    total_ok = 0
    total_err = 0
    print('Testing urls ... ', end='\n')
    for url in urls:
        request = urllib.request.Request(url, method='HEAD')
        try:
            response = urllib.request.urlopen(request, timeout=wait)
            if response.status == 200:
                print(url, ' : OK')
                total_ok += 1
            else:
                print('ERR',  url, ' : ERROR: {}'.format(response.status))
                total_err += 1
        except (urllib.error.HTTPError, urllib.error.URLError) as error:
            print("ERR Could not retrieve URL {}: {}".format(url, error))
            total_err += 1
        except timeout:
            print('ERR socket timed out - URL %s', url)
            total_err += 1
    print('Total URLs {}, OK {}, ERR {}'.format(total_urls, total_ok, total_err))


def main():
    check_urls(infile, wait)


if __name__ == '__main__':
    main()
