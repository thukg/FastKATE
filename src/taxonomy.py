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
re.DEFAULT_VERSION = re.VERSION1

from FastKATE.utils.logger import simple_logger
logger = simple_logger(__name__, 'FastKATE/log')

from nltk.stem import WordNetLemmatizer
lemmatize = WordNetLemmatizer().lemmatize
lemmatize('')

from FastKATE.src.api import normalize_name_for_querying_vector_model

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

def main():
    timestamp = '20170901' # CHANGE THIS AS NEEDED
    parser = argparse.ArgumentParser()
    parser.add_argument('--pages_infile', default='./wikidata/enwiki-{}-pages.sql'.format(timestamp))
    parser.add_argument('--pages_outfile', default='./wikidata/enwiki-{}-pages-outfile'.format(timestamp))
    parser.add_argument('--categorylinks_infile', default='./wikidata/enwiki-{}-categorylinks.sql'.format(timestamp))
    parser.add_argument('--categorylinks_outfile', default='./wikidata/enwiki-{}-categorylinks-outfile'.format(timestamp))
    parser.add_argument('category_infile', default='./wikidata/enwiki-{0}-category.sql'.format(timestamp))
    parser.add_argument('category_outfile', default='./wikidata/enwiki-{0}-category-outfile'.format(timestamp))
    parser.add_argument('page_and_category_outfile', default='./wikidata/enwiki-{0}-page-and-category-outfile'.format(timestamp))
    parser.add_argument('taxonomy_outfile', default='./wikidata/taxonomy.pkl')
    parser.add_argument('taxonomy_lemmatized_outfile', default='./wikidata/taxonomy_lemmatized.pkl')
    args = parser.parse_args()

    pages_infile = args.pages_infile
    pages_outfile = args.pages_outfile
    categorylinks_infile = args.categorylinks_infile
    categorylinks_outfile = args.categorylinks_outfile
    category_infile = args.category_infile
    category_outfile = args.category_outfile
    page_and_category_outfile = args.page_and_category_outfile
    taxonomy_outfile = args.taxonomy_outfile
    taxonomy_lowercase_outfile = args.taxonomy_lowercase_outfile


    # pages
    if 1:
        line_no = 0
        lines = []

        workers = []
        queue = mp.SimpleQueue()
        for i in range(mp.cpu_count()):
            workers.append(Worker(parse_k_lines_for_pages, queue))
        for w in workers:
            w.start()

        tmp_dir = '/tmp/thuzhf/taxonomy/'
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

    # category links
    if 1:
        line_no = 0
        lines = []

        workers = []
        queue = mp.SimpleQueue()
        for i in range(mp.cpu_count()):
            workers.append(Worker(parse_k_lines_for_categorylinks, queue))
        for w in workers:
            w.start()

        tmp_dir = '/tmp/thuzhf/taxonomy/'
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

    # category (We do this because we find that page.sql doesn't contain all categories.)
    if 1:
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

    pages_ids = {}
    pages_titles = {}
    # merge page&category
    if 1:
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
        logger.info('merging into page_and_category_outfile...')
        with open(page_and_category_outfile, 'w', errors='surrogateescape') as f:
            for i in pages_ids:
                f.write('{}\t{}\t{}\n'.format(i, pages_ids[i]['page_namespace'], pages_ids[i]['page_title']))

    if 1:
        logger.info('loading page_and_category_outfile...')
        with open(page_and_category_outfile, errors='surrogateescape') as f:
            for line in f:
                page_id, page_namespace, page_title = line.strip('\n').split('\t')
                pages_ids[page_id] = {'page_namespace': page_namespace, 'page_title': page_title}
                pages_titles[page_title] = {'page_namespace': page_namespace, 'page_id': page_id, 'subcats': [], 'subpages': []}

        logger.info('loading categorylinks_outfile...')
        with open(categorylinks_outfile, errors='surrogateescape') as f:
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
        
        logger.info('pickling into {}...'.format(taxonomy_outfile))
        with open(taxonomy_outfile, 'wb') as f:
            pickle.dump(pages_titles, f)

    pages_titles_lemmatized = {}
    if 1:
        logger.info('loading pickled file {}...'.format(taxonomy_outfile))
        with open(taxonomy_outfile, 'rb') as f:
            pages_titles = pickle.load(f)
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
        logger.info('pickling into {}...'.format(taxonomy_lowercase_outfile))
        with open(taxonomy_lowercase_outfile, 'wb') as f:
            pickle.dump(pages_titles_lemmatized, f)


if __name__ == '__main__':
    start_t = time.time()
    main()
    end_t = time.time()
    t = end_t - start_t
    logger.info('Time elapsed: {:.4f} minutes'.format(t / 60.))