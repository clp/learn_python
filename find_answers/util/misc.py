from bs4 import BeautifulSoup


def calc_progress(input):
    if input < 11:
        progress_i = 1
    else:
        progress_i = max(10, int(round(input / 10)))
    return progress_i


def strip_html(raw_qa_s, lxml):
    """Remove html tags from the i/p string.
    """

    qa_text = BeautifulSoup(raw_qa_s, lxml).get_text()
    return qa_text 

