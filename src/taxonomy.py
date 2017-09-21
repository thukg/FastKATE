#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Fang Zhang <thuzhf@gmail.com>

import sys,os,json,gzip,math,time,datetime,random,copy
import functools,itertools,requests,pickle,configparser
import argparse,logging,uuid,shutil,collections
from urllib.parse import urlparse, parse_qs
from collections import defaultdict as dd
import multiprocessing as mp
import numpy as np
import regex as re
import string
re.DEFAULT_VERSION = re.VERSION1

from wikidump_parser.utils.logger import simple_logger
logger = simple_logger(__name__, 'wikidump_parser/log')

from nltk.stem import WordNetLemmatizer
lemmatize = WordNetLemmatizer().lemmatize
lemmatize('')

def rand_str(n):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(n))

class Worker(mp.Process):
    def __init__(self, handler_func, queue):
        super().__init__()
        self.handler_func = handler_func
        self.queue = queue

    def run(self):
        for params in iter(self.queue.get, None):
            self.handler_func(*params)

def parse_k_lines_for_pages(lines, outfile, line_no):
    start_str = 'INSERT INTO `page` VALUES '
    # arbitrary_str = r"'.*?(?<!(?<!\\)(?:\\\\)*\\)'"
    arbitrary_str = r"'(?:\\\\|\\'|.)*?'"
    page_regex_str = r"\(([+-]?\d+),([+-]?\d+),({0}),{0},[+-]?\d+,[+-]?\d+,[+-]?\d+,[0-9\.]+,{0},(?:{0}|NULL),[+-]?\d+,[+-]?\d+,(?:{0}|NULL),(?:{0}|NULL)\)".format(arbitrary_str)
    page_regex = re.compile(page_regex_str)

    results = []
    for line in lines:
        if line.startswith(start_str):
            for i in page_regex.findall(line):
                try:
                    page_id = eval(i[0])
                    page_namespace = eval(i[1])
                    page_title = eval(i[2])
                    if page_namespace == 0 or page_namespace == 14:
                        results.append((page_id, page_namespace, page_title))
                except:
                    logger.info(i)
                    sys.exit()
    with open(outfile, 'w', errors='surrogateescape') as f:
        for r in results:
            f.write('{}\t{}\t{}\n'.format(r[0], r[1], r[2]))
    logger.info('DONE: {}'.format(line_no))

def extract_all_page_basic_info(pages_infile, pages_outfile, category_infile, category_outfile, all_pages_outfile):
    extract_page_basic_info(pages_infile, pages_outfile)
    extract_category_basic_info(category_infile, category_outfile)
    pages_ids = {}
    logger.info('loading pages_outfile...')
    with open(pages_outfile, errors='surrogateescape') as f:
        for line in f:
            page_id, page_namespace, page_title = line.strip('\n').split('\t')
            if page_namespace == '14':
                page_title = 'Category:{}'.format(page_title)
            pages_ids[page_id] = {'page_namespace': page_namespace, 'page_title': page_title}
    logger.info('loading category_outfile...')
    with open(category_outfile, errors='surrogateescape') as f:
        for line in f:
            page_id, page_namespace, page_title = line.strip('\n').split('\t')
            page_title = 'Category:{}'.format(page_title)
            pages_ids[page_id] = {'page_namespace': page_namespace, 'page_title': page_title}
    logger.info('merging into all_pages_outfile...')
    with open(all_pages_outfile, 'w', errors='surrogateescape') as f:
        for i in pages_ids:
            f.write('{}\t{}\t{}\n'.format(i, pages_ids[i]['page_namespace'], pages_ids[i]['page_title']))
    return pages_ids

def extract_page_basic_info(pages_infile, pages_outfile):
    line_no = 0
    lines = []

    workers = []
    queue = mp.SimpleQueue()
    for i in range(mp.cpu_count()):
        workers.append(Worker(parse_k_lines_for_pages, queue))
    for w in workers:
        w.start()

    tmp_dir = '/tmp/wikipage/'
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    with open(pages_infile, errors='surrogateescape') as f:
        for line in f:
            # if line.startswith(start_str):
            lines.append(line)
            line_no += 1
            if line_no % 5 == 0:
                outfile = '{}{}.txt'.format(tmp_dir, line_no)
                queue.put((lines, outfile, line_no))
                lines = []
        if lines:
            outfile = '{}{}.txt'.format(tmp_dir, line_no)
            queue.put((lines, outfile, line_no))
            lines = []
        for _ in workers:
            queue.put(None)
    for w in workers:
        w.join()

    with open(pages_outfile, 'w', errors='surrogateescape') as fout:
        for filename in os.listdir(tmp_dir):
            full_path = os.path.join(tmp_dir, filename)
            with open(full_path, errors='surrogateescape') as fin:
                fout.write(fin.read())

def parse_k_lines_for_categorylinks(lines, outfile, line_no):
    start_str = 'INSERT INTO `categorylinks` VALUES '
    # arbitrary_str = r"'.*?(?<!(?<!\\)(?:\\\\)*\\)'"
    arbitrary_str = r"'(?:\\\\|\\'|.)*?'"
    cl_regex_str = r"\(([+-]?\d+),({0}),{0},'.*?',{0},{0},('.*?')\)".format(arbitrary_str)
    cl_regex = re.compile(cl_regex_str)

    results = []
    for line in lines:
        if line.startswith(start_str):
            for i in cl_regex.findall(line):
                try:
                    cl_from = eval(i[0])
                    cl_to = eval(i[1])
                    cl_type = eval(i[2])
                    if cl_type == 'page' or cl_type == 'subcat':
                        results.append((cl_from, cl_to, cl_type))
                except:
                    logger.info(i)
                    sys.exit()
    with open(outfile, 'w', errors='surrogateescape') as f:
        for r in results:
            f.write('{}\t{}\t{}\n'.format(r[0], r[1], r[2]))
    if line_no % 100 == 0:
        logger.info('DONE: {}'.format(line_no))

def extract_categorylinks_basic_info(categorylinks_infile, categorylinks_outfile):
    line_no = 0
    lines = []

    workers = []
    queue = mp.SimpleQueue()
    for i in range(mp.cpu_count()):
        workers.append(Worker(parse_k_lines_for_categorylinks, queue))
    for w in workers:
        w.start()

    tmp_dir = '/tmp/wiki_categorylinks/'
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    with open(categorylinks_infile, errors='surrogateescape') as f:
        for line in f:
            # if line.startswith(start_str):
            lines.append(line)
            line_no += 1
            if line_no % 1 == 0:
                outfile = '{}{}.txt'.format(tmp_dir, line_no)
                queue.put((lines, outfile, line_no))
                lines = []
        if lines:
            outfile = '{}{}.txt'.format(tmp_dir, line_no)
            queue.put((lines, outfile, line_no))
            lines = []
        for _ in workers:
            queue.put(None)
    for w in workers:
        w.join()

    with open(categorylinks_outfile, 'w', errors='surrogateescape') as fout:
        for filename in os.listdir(tmp_dir):
            full_path = os.path.join(tmp_dir, filename)
            with open(full_path, errors='surrogateescape') as fin:
                fout.write(fin.read())

def parse_k_lines_for_category(lines, outfile, line_no):
    start_str = 'INSERT INTO `category` VALUES '
    # arbitrary_str = r"'.*?(?<!(?<!\\)(?:\\\\)*\\)'"
    arbitrary_str = r"'(?:\\\\|\\'|.)*?'"
    c_regex_str = r"\(([+-]?\d+),({0}),[+-]?\d+,[+-]?\d+,[+-]?\d+\)".format(arbitrary_str)
    c_regex = re.compile(c_regex_str)

    results = []
    for line in lines:
        if line.startswith(start_str):
            for i in c_regex.findall(line):
                try:
                    cat_id = eval(i[0])
                    cat_title = eval(i[1])
                    results.append((cat_id, 14, cat_title)) # 14 stands for category
                except:
                    logger.info(i)
                    sys.exit()
    with open(outfile, 'w', errors='surrogateescape') as f:
        for r in results:
            f.write('{}\t{}\t{}\n'.format(r[0], r[1], r[2]))
    logger.info('DONE: {}'.format(line_no))

def extract_category_basic_info(category_infile, category_outfile):
    line_no = 0
    lines = []

    workers = []
    queue = mp.SimpleQueue()
    for i in range(mp.cpu_count()):
        workers.append(Worker(parse_k_lines_for_category, queue))
    for w in workers:
        w.start()

    tmp_dir = '/tmp/thuzhf/taxonomy/'
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    with open(category_infile, errors='surrogateescape') as f:
        for line in f:
            # if line.startswith(start_str):
            lines.append(line)
            line_no += 1
            if line_no % 1 == 0:
                outfile = '{}{}.txt'.format(tmp_dir, line_no)
                queue.put((lines, outfile, line_no))
                lines = []
        if lines:
            outfile = '{}{}.txt'.format(tmp_dir, line_no)
            queue.put((lines, outfile, line_no))
            lines = []
        for _ in workers:
            queue.put(None)
    for w in workers:
        w.join()

    with open(category_outfile, 'w', errors='surrogateescape') as fout:
        for filename in os.listdir(tmp_dir):
            full_path = os.path.join(tmp_dir, filename)
            with open(full_path, errors='surrogateescape') as fin:
                fout.write(fin.read())

def construct_pages_taxonomy(all_pages_outfile_as_infile, categorylinks_outfile_as_infile, taxonomy_lemmatized_outfile):
    pages_ids = {}
    pages_titles = {}
    logger.info('loading all_pages_outfile_as_infile...')
    with open(all_pages_outfile_as_infile, errors='surrogateescape') as f:
        for line in f:
            page_id, page_namespace, page_title = line.strip('\n').split('\t')
            pages_ids[page_id] = {'page_namespace': page_namespace, 'page_title': page_title}
            pages_titles[page_title] = {'page_namespace': page_namespace, 'page_id': page_id, 'subcats': [], 'subpages': []}

    logger.info('loading categorylinks_outfile_as_infile...')
    with open(categorylinks_outfile_as_infile, errors='surrogateescape') as f:
        for line in f:
            cl_from, cl_to, cl_type = line.strip('\n').split('\t')
            cl_to = 'Category:{}'.format(cl_to)
            if cl_to not in pages_titles:
                logger.info('Category missing: {}'.format(cl_to))
            elif cl_from not in pages_ids:
                # logger.info('Page/Category missing: {}'.format(cl_from))
                continue
            else:
                if cl_type == 'page':
                    pages_titles[cl_to]['subpages'].append(pages_ids[cl_from]['page_title'])
                elif cl_type == 'subcat':
                    pages_titles[cl_to]['subcats'].append(pages_ids[cl_from]['page_title'])

    pages_titles_lemmatized = {}
    logger.info('lemmatize all wiki titles...')
    for t in pages_titles:
        tl = t[9:].lower().replace('-', '_') # 9 == len('Category:')
        tl = normalize_name_for_querying_vector_model(tl)

        tmp = pages_titles[t]
        if tl not in pages_titles_lemmatized:
            pages_titles_lemmatized[tl] = {'page_id': set(), 'subcats': set(), 'subpages': set()}
        pages_titles_lemmatized[tl]['page_id'].add(int(tmp['page_id']))
        for sc in tmp['subcats']:
            sc = normalize_name_for_querying_vector_model(sc[9:].lower().replace('-', '_'))
            pages_titles_lemmatized[tl]['subcats'].add(sc)
        for sp in tmp['subpages']:
            sp = normalize_name_for_querying_vector_model(sp.lower().replace('-', '_'))
            pages_titles_lemmatized[tl]['subpages'].add(sp)
    logger.info('pickling into {}...'.format(taxonomy_lemmatized_outfile))
    with open(taxonomy_lemmatized_outfile, 'wb') as f:
        pickle.dump(pages_titles_lemmatized, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('timestamp', help='the timestamp of wikipedia dumps that you have downloaded')
    parser.add_argument('wikidata_dir', help='the directory of wikipedia dumps that you have downloaded')
    parser.add_argument('job', help='pages_basic_info/categorylinks/pages_taxonomy')
    args = parser.parse_args()

    timestamp = args.timestamp
    pages_infile = os.path.join(args.wikidata_dir, 'enwiki-{}-pages.sql'.format(timestamp))
    pages_outfile = os.path.join(args.wikidata_dir, 'enwiki-{}-pages-outfile'.format(timestamp))
    categorylinks_infile = os.path.join(args.wikidata_dir, 'enwiki-{}-categorylinks.sql'.format(timestamp))
    categorylinks_outfile = os.path.join(args.wikidata_dir, 'enwiki-{}-categorylinks-outfile'.format(timestamp))
    category_infile = os.path.join(args.wikidata_dir, 'enwiki-{}-category.sql'.format(timestamp))
    category_outfile = os.path.join(args.wikidata_dir, 'enwiki-{}-category-outfile'.format(timestamp))
    all_pages_outfile = os.path.join(args.wikidata_dir, 'enwiki-{}-page-and-category-outfile'.format(timestamp))
    taxonomy_lemmatized_outfile = os.path.join(args.wikidata_dir, 'taxonomy_lemmatized.pkl')

    if args.job == 'pages_basic_info':
        extract_all_page_basic_info(pages_infile, pages_outfile, category_infile, category_outfile, all_pages_outfile)
    elif args.job == 'categorylinks':
        extract_categorylinks_basic_info(categorylinks_infile, categorylinks_outfile)
    elif args.job == 'pages_taxonomy':
        if not os.path.isfile(all_pages_outfile) or not os.path.getsize(all_pages_outfile):
            extract_all_page_basic_info(pages_infile, pages_outfile, category_infile, category_outfile, all_pages_outfile)
        if not os.path.isfile(categorylinks_outfile) or not os.path.getsize(categorylinks_outfile):
            extract_categorylinks_basic_info(categorylinks_infile, categorylinks_outfile)
        construct_pages_taxonomy(all_pages_outfile, categorylinks_outfile, taxonomy_lemmatized_outfile)


if __name__ == '__main__':
    start_t = time.time()
    main()
    end_t = time.time()
    t = end_t - start_t
    logger.info('Time elapsed: {:.4f} minutes'.format(t / 60.))