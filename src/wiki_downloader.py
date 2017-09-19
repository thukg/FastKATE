#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys,os,json,gzip,math,time,datetime,random,copy
import functools,itertools,requests,pickle,configparser
import argparse,logging,uuid,shutil,collections
from urllib.parse import urlparse, parse_qs
from collections import defaultdict as dd
import multiprocessing as mp
import numpy as np
import regex as re
re.DEFAULT_VERSION = re.VERSION1
from subprocess import call

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('time_stamp', help='such as 20170901; you can only choose timestamps that are listed at https://dumps.wikimedia.org/enwiki/')
    parser.add_argument('out_dir', help='output directory')
    parser.add_argument('job', help='all_titles/pages_articles/pages_articles_multistream/pages/categorylinks/category/all', action='append')
    args = parser.parse_args()
    time_stamp = args.time_stamp
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    if 'all' in args.job:
        pending_jobs = ['all_titles', 'pages_articles', 'pages_articles_multistream', 'pages', 'categorylinks', 'category']
    else:
        pending_jobs = args.job

    mirrors = [
        "https://dumps.wikimedia.org/enwiki/", 
        "http://ftp.acc.umu.se/mirror/wikimedia.org/dumps/enwiki/", 
        "http://wikipedia.c3sl.ufpr.br/enwiki/", 
        "http://dumps.wikimedia.your.org/enwiki/"
    ]

    all_titles = "{0}/enwiki-{0}-all-titles-in-ns0.gz".format(time_stamp)
    pages_articles = "{0}/enwiki-{0}-pages-articles.xml.bz2".format(time_stamp)
    pages_articles_multistream = "{0}/enwiki-{0}-pages-articles-multistream.xml.bz2".format(time_stamp)
    pages = "{0}/enwiki-{0}-pages.sql.gz".format(time_stamp)
    categorylinks = "{0}/enwiki-{0}-categorylinks.sql.gz".format(time_stamp)
    category = "{0}/enwiki-{0}-category.sql.gz".format(time_stamp)

    latter_urls_for_different_jobs = {'all_titles': all_titles, 'pages_articles': pages_articles, 'pages_articles_multistream': pages_articles_multistream, 'pages': pages, 'categorylinks': categorylinks, 'category': category}

    for j in pending_jobs:
        latter_url = latter_urls_for_different_jobs[j]
        basename = os.path.basename(latter_url)
        if os.path.isfile(os.path.join(out_dir, basename)):
            continue
        all_urls = []
        for base_url in mirrors:
            tmp_url = os.path.join(base_url, latter_url)
            all_urls.append(tmp_url)
        params = ['aria2c', '-j {}'.format(len(mirrors) * 2), '-s {}'.format(len(mirrors) * 2), '-x 2', '-d {}'.format(out_dir)]
        params.extend(all_urls)
        call(params)

if __name__ == '__main__':
    start_t = time.time()
    main()
    end_t = time.time()
    t = end_t - start_t
    print('Time elapsed: {:.4f} minutes'.format(t / 60.))