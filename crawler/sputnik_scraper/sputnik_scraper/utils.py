import re
from datetime import datetime
from bs4 import BeautifulSoup, Tag
from time import gmtime, strftime


def now():
    """
    Returns
    -------
    Current time : str
        eg: 2018-11-22 13:35:23
    """
    return strftime("%Y-%m-%d-%W %H:%M:%S", gmtime())


def get_soup(url, headers=None):
    """
    Arguments
    ---------
    url : str
        Web page url
    headers : dict
        Headers for requests. If None, use Mozilla/5.0 as default user-agent

    Returns
    -------
    soup : bs4.BeautifulSoup
        Soup format web page
    """

    if headers is None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'}
    r = requests.get(url, headers=headers)
    html = r.text
    page = BeautifulSoup(html, 'lxml')
    return page


doublespace_pattern = re.compile('\s+')
lineseparator_pattern = re.compile('\n+')


def normalize_text(text):
    text = text.replace('\t', ' ')
    text = text.replace('\r', ' ')
    text = lineseparator_pattern.sub('\n', text)
    text = doublespace_pattern.sub(' ', text)
    return text.strip()
