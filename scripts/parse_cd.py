#!/usr/bin/env python3

'''
Usage:
    parse_cd.py [--timeout=<n>]

Options:
    --timeout=<n>    Maximum time in seconds to run for [default: inf].
'''

import os
import sys
from functools import reduce
_upper_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
if _upper_dir not in sys.path:
    sys.path.append(_upper_dir)

import chdb
import config
import yamwapi
import snippet_parser
from utils import *

import docopt
import requests

import cProfile
import functools
import glob
import itertools
import logging
import multiprocessing
import pstats
import re
import shutil
import tempfile
import time
import traceback
import types
import urllib.request, urllib.parse, urllib.error
import pickle as pickle

import nltk
import mwapi
import mwparserfromhell
from mwparserfromhell.nodes.text import Text
import pandas as pd

cfg = config.get_localized_config()
WIKIPEDIA_BASE_URL = 'https://' + cfg.wikipedia_domain
WIKIPEDIA_WIKI_URL = WIKIPEDIA_BASE_URL + '/wiki/'
WIKIPEDIA_API_URL = WIKIPEDIA_BASE_URL + '/w/api.php'

MAX_EXCEPTIONS_PER_SUBPROCESS = 5

DATA_TRUNCATED_WARNING_RE = re.compile(
    'Data truncated for column .* at row (\d+)')

CITATION_NEEDED_TEMPLATE = "{{Citation needed|reason=Added by Citation Detective|date=%s}}" % pd.datetime.now().strftime("%B %Y")
CITATION_NEEDED_TEMPLATE = mwparserfromhell.parse(CITATION_NEEDED_TEMPLATE).filter_templates()[0]

#logger = logging.getLogger('parse_live')
#setup_logger_to_stderr(logger)

def section_name_to_anchor(section):
    # See Sanitizer::escapeId
    # https://doc.wikimedia.org/mediawiki-core/master/php/html/classSanitizer.html#ae091dfff62f13c9c1e0d2e503b0cab49
    section = section.replace(' ', '_')
    # urllib.quote interacts really weirdly with unicode in Python2:
    # https://bugs.python.org/issue23885
    section = urllib.parse.quote(e(section), safe = e(''))
    section = section.replace('%3A', ':')
    section = section.replace('%', '.')
    return section

def query_revids(revids):
    session = mwapi.Session('https://en.wikipedia.org/', 'citationdetective', formatversion=2)

    response_docs = session.post(action='query',
                                prop='revisions',
                                rvprop='ids|content',
                                revids='|'.join(map(str, revids)),
                                format='json',
                                utf8='',
                                rvslots='*',
                                continuation=True)

    for doc in response_docs:
        for page in doc['query']['pages']:
            pageid = page['pageid']
            revid = page['revisions'][0]['revid']
            if 'title' not in page:
                continue
            title = d(page['title'])
            text = page['revisions'][0]['slots']['main']['content']
            if not text:
                continue
            text = d(text)
            yield (revid, pageid, title, text)

self = types.SimpleNamespace() # Per-process state

def initializer(backdir):
    self.backdir = backdir

    self.wiki = yamwapi.MediaWikiAPI(WIKIPEDIA_API_URL, cfg.user_agent)
    self.parser = snippet_parser.create_snippet_parser(self.wiki, cfg, 'cd')
    self.exception_count = 0
    """
    if cfg.profile:
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        # Undocumented :( https://stackoverflow.com/questions/24717468
        multiprocessing.util.Finalize(None, finalizer, exitpriority=16)
    """
def finalizer():
    self.profiler.disable()
    profile_path = os.path.join(self.backdir, 'profile-%s' % os.getpid())
    pstats.Stats(self.profiler).dump_stats(profile_path)
    stats_path = os.path.join(self.backdir, 'stats-%s' % os.getpid())
    with open(stats_path, 'wb') as stats_f:
        pickle.dump(self.parser.stats, stats_f)

def _add_citation_needed_tpl(statement, section):
    match = False
    for text in section.filter_text():
        # a text node may contains elements from multiple sentences 
        tokens = nltk.tokenize.sent_tokenize(text.value)
        if not tokens:
            continue
        for i, token in enumerate(tokens):
            if token.strip() == statement[:len(token.strip())].strip():
                statement = statement[len(token.strip()):].strip()
                if not statement:
                    match = True
                    # match first token
                    if i == 0:
                        if len(tokens) == 1:
                            section.insert_after(text, CITATION_NEEDED_TEMPLATE)
                        else:
                            section.insert_before(text, Text(token))
                            section.insert_before(text, CITATION_NEEDED_TEMPLATE)
                            for k in range(i+1, len(tokens)):
                                section.insert_after(text, Text(tokens[k]))
                            section.remove(text)
                    # match last token
                    elif i == len(tokens)-1:
                        section.insert_after(text, CITATION_NEEDED_TEMPLATE)
                    # match token in between
                    else:
                        for k in range(0, i+1): 
                            section.insert_before(text, Text(tokens[k]))
                        section.insert_before(text, CITATION_NEEDED_TEMPLATE)
                        for k in range(i+1, len(tokens)):
                            section.insert_after(text, Text(tokens[k]))
                        section.remove(text)
                    return match
    return match

def add_citation_needed_tpl(wikitext, df):
    cn_count = 0
    wikicode = mwparserfromhell.parse(wikitext)
    sections = wikicode.get_sections(levels=[2], include_lead=True)
    for section in sections:
        headings = section.filter_headings()
        if not headings:
            if 'main_section' in df.section.unique():
                for statement in df[df.section == 'main_section']['statement'].values:
                    if _add_citation_needed_tpl(statement, section):
                        cn_count += 1
            continue

        section_name = headings[0].title.strip().lower()
        if section_name in df.section.unique():
            for statement in df[df.section == section_name]['statement'].values:
                if _add_citation_needed_tpl(statement, section):
                    cn_count += 1
    return wikicode, cn_count

def process(df):
    _tpl_added = 0
    _num_snippet = 0

    rows = []
    results = query_revids(df.rev_id.unique())
    for revid, pageid, title, wikitext in results:
        #print(pageid, title)
        wikicode, cn = add_citation_needed_tpl(wikitext, df[df.rev_id == revid])
        _tpl_added += cn
        
        url = WIKIPEDIA_WIKI_URL + title.replace(' ', '_')
        snippets_rows = []
        snippets = self.parser.extract(wikicode)
        _num_snippet += len(snippets)
        
        for sec, snips in snippets:
            sec = section_name_to_anchor(sec)
            for sni in snips:
                id = mkid(title + sni)
                row = (id, sni, sec, pageid)
                snippets_rows.append(row)

        if snippets_rows:
            article_row = (pageid, url, title)
            rows.append({'article': article_row, 'snippets': snippets_rows})
    
    def insert(cursor, r):
        cursor.execute('''
            INSERT INTO articles VALUES(%s, %s, %s)''', r['article'])
        cursor.executemany('''
            INSERT IGNORE INTO snippets VALUES(%s, %s, %s, %s)''',
            r['snippets'])

        # We can't allow data to be truncated for HTML snippets, as that can
        # completely break the UI, so we detect truncation warnings and get rid
        # of the corresponding data.
        warnings = cursor.execute('SHOW WARNINGS')
        truncated_snippets = []
        for _, _, message in cursor.fetchall():
            m = DATA_TRUNCATED_WARNING_RE.match(message)
            if m is None:
                # Not a truncation, ignore (it's already logged)
                continue
            # MySQL warnings index rows starting at 1
            idx = int(m.groups()[0]) - 1
            truncated_snippets.append((r['snippets'][idx][0],))
        if len(truncated_snippets) < len(r['snippets']):
            cursor.executemany('''
                DELETE FROM snippets WHERE id = %s''', truncated_snippets)
        else:
            # Every single snippet was truncated, remove the article itself
            cursor.execute('''DELETE FROM articles WHERE page_id = %s''',
                (r['article'][0],))
    # Open a short-lived connection to try to avoid the limit of 20 per user:
    # https://phabricator.wikimedia.org/T216170
    db = chdb.init_scratch_db()
    for r in rows:
        db.execute_with_retry(insert, r)
    
    return _tpl_added, _num_snippet

def parse_live(df, timeout):
    tpl_added = 0
    snippet_count = 0

    backdir = tempfile.mkdtemp(prefix = 'citationhunt_parse_live_')
    initializer(backdir)

    # Make sure we query the API 32 pageids at a time
    batch_size = 32
    revids_list = list(df.rev_id.unique())
    for i in range(0, len(revids_list), batch_size):
        _tpl_added, _num_snippet = process(df[df.rev_id.isin(revids_list[i:i+batch_size])])

        tpl_added += _tpl_added
        snippet_count += _num_snippet

    print("number of {{cn}} added: ", tpl_added)
    print("number of snippets found: ", snippet_count)
    return 0

def query_citation_detective(cn_score, size):
    db = chdb.init_cd_db()
    cursor = db.cursor()
    cursor.execute( # limit number of rows for testing
        '''SELECT statement, section, rev_id FROM statements WHERE score > %s
         ORDER BY rev_id, section LIMIT %s''', (cn_score, size))
    return pd.DataFrame(cursor.fetchall(), columns=['statement', 'section', 'rev_id'])

if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)
    timeout = float(arguments['--timeout'])
    if timeout == float('inf'):
        timeout = None
    start = time.time()
    df = query_citation_detective(0.5, 10000)
    print("number of article: ", df.rev_id.nunique())
    print("number of statements: ", df.shape[0])
    print('query cd done in %d seconds.' % (time.time() - start))
    
    ret = parse_live(df, timeout)
    print('all done in %d seconds.' % (time.time() - start))
    sys.exit(ret)
