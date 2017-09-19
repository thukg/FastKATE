#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Fang Zhang <thuzhf@gmail.com>

import sys,os,json,gzip,math,time,datetime,random,copy
import functools,itertools,requests,pickle,configparser
import argparse,logging,uuid,shutil
from collections import defaultdict as dd
import multiprocessing as mp
import numpy as np
import regex as re
re.DEFAULT_VERSION = re.VERSION1
import gensim

from FastKATE.utils.multiprocess import multiprocess_dir, multiprocess_file
from FastKATE.utils.logger import simple_logger
logger = simple_logger(__name__, 'FastKATE/log/')

from nltk.stem import WordNetLemmatizer
lemmatize = WordNetLemmatizer().lemmatize
lemmatize('')
from spacy.en import English
nlp = English()


# global variables are defined here
config = configparser.ConfigParser()
config_file = 'FastKATE/config/word2vec.cfg'
config.read(config_file)
section = config['global']

SIZE = eval(section['SIZE'])
WINDOW = eval(section['WINDOW'])
MIN_COUNT = eval(section['MIN_COUNT'])
WORKERS = eval(section['WORKERS'])
SAMPLE = eval(section['SAMPLE'])

MIN_WORD_LENGTH = eval(section['MIN_WORD_LENGTH'])
MIN_SENTENCE_LENGTH = eval(section['MIN_SENTENCE_LENGTH'])
FORBIDDEN_POS = eval(section['FORBIDDEN_POS'])
# FORBIDDEN_TAGS = eval(section['FORBIDDEN_TAGS'])
POSSIBLE_FORBIDDEN_TAGS = eval(section['POSSIBLE_FORBIDDEN_TAGS'])
MUST_TAGS = eval(section['MUST_TAGS'])

PHRASE_LENGTHS = eval(section['PHRASE_LENGTHS'])
MIN_PHRASE_LENGTHS, MAX_PHRASE_LENGTHS = min(PHRASE_LENGTHS), max(PHRASE_LENGTHS)
USE_FILTERED_WIKI_TITLES = eval(section['USE_FILTERED_WIKI_TITLES'])
if USE_FILTERED_WIKI_TITLES:
    _FILTERED = '_onFilteredWikiTitles'
else:
    _FILTERED = ''
logger.info('configs have been read from {}'.format(config_file))

def transform_to_noun_phrases_list(doc):
    np = []
    tokens = nlp(doc, entity=False)
    for i in tokens.noun_chunks:
        tmp = []
        for j in i:
            if not j.is_stop:
                tmp.append(lemmatize(j.lower_))
        np.append('_'.join(tmp))
    return np

def lower_and_remove_non_alphanumeric_wiki_title(title): # title should be a non-empty string
    title = title.lower()
    title2 = title.strip()
    if 'regex' not in lower_and_remove_non_alphanumeric_wiki_title.__dict__:
        lower_and_remove_non_alphanumeric_wiki_title.regex = re.compile(r'^(?!-)([a-z0-9-]+_?)+$')
    if title2 and lower_and_remove_non_alphanumeric_wiki_title.regex.match(title2):
        return title
    else:
        return ''

def lemmatize_and_remove_wiki_title_consisting_of_short_words(title, min_word_length=MIN_WORD_LENGTH):
    title2 = title.strip()
    tmp = [lemmatize(i) for i in title2.split('_')]
    valid = False
    for word in tmp:
        if len(word) >= min_word_length:
            valid = True
            break
    if title2 and valid:
        return '{}\n'.format('_'.join(tmp))
    else:
        return ''

def remove_wiki_title_by_pos_and_tag_rules(title, forbidden_pos=FORBIDDEN_POS, forbidden_tags=POSSIBLE_FORBIDDEN_TAGS, must_tags=MUST_TAGS):
    title2 = title.strip().replace('_', ' ')
    tokens = nlp(title2, tag=True, parse=False, entity=False)
    skip_line = False
    if must_tags:
        exist_one_must_tag = False
    else:
        exist_one_must_tag = True
    for t in tokens:
        if forbidden_pos and t.pos_ in forbidden_pos:
            skip_line = True
            break
        if forbidden_tags and t.tag_ in forbidden_tags:
            skip_line = True
            break
        if must_tags and t.tag_ in must_tags:
            exist_one_must_tag = True
    if title2 and not skip_line and exist_one_must_tag:
        return title
    else:
        return ''

# remove words with forbidden POSs, and make remaining words lowercase and lemmatized
def clean_raw_text_file(infile, outfile, ignored_line_start='<', ignore_existed_file=False, forbidden_pos=FORBIDDEN_POS):
    if ignore_existed_file and os.path.isfile(outfile):
        return
    logger.info('processing file: {}'.format(infile))
    with open(infile) as fin, open(outfile, 'w') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                fout.write('\n')
                continue
            if ignored_line_start != None and line.startswith(ignored_line_start):
                continue
            tokens = nlp(line, tag=True, parse=False, entity=False)
            lemmatized_lower_tokens = []
            for t in tokens:
                if t.is_stop:
                    continue
                if t.pos_ not in forbidden_pos:
                    lemmatized_lower_tokens.append(lemmatize(t.lower_))
            if lemmatized_lower_tokens:
                fout.write('{}\n'.format(' '.join(lemmatized_lower_tokens)))
    logger.info('written into file: {}'.format(outfile))

# remove words with forbidden POSs, and make remaining words lowercase and lemmatized
def clean_raw_text_file_and_add_newline_after_each_document(infile, outfile, ignore_existed_file=False, forbidden_pos=FORBIDDEN_POS):
    if ignore_existed_file and os.path.isfile(outfile):
        return
    logger.info('processing file: {}'.format(infile))
    with open(infile) as fin, open(outfile, 'w') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                # fout.write('\n')
                continue
            if line.startswith('<doc id="'):
                continue
            if line.startswith('</doc>'):
                fout.write('\n')
                continue
            tokens = nlp(line, tag=True, parse=False, entity=False)
            lemmatized_lower_tokens = []
            for t in tokens:
                if t.is_stop:
                    continue
                if t.pos_ not in forbidden_pos:
                    lemmatized_lower_tokens.append(lemmatize(t.lower_))
            if lemmatized_lower_tokens:
                fout.write('{}\n'.format(' '.join(lemmatized_lower_tokens)))
    logger.info('written into file: {}'.format(outfile))


def generate_sentences_of_noun_phrases(infile, outfile):
    logger.info('processing file: {}...'.format(infile))
    with open(infile) as fin, open(outfile, 'w') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                fout.write('\n')
                continue
            if line.startswith('<doc id=') or line.startswith('</doc>'):
                continue
            np_list = transform_to_noun_phrases_list(line)
            if np_list:
                fout.write('{}\n'.format(' '.join(np_list)))
    logger.info('has written into file: {}'.format(outfile))

def generate_sentences_of_given_phrases(given_phrases, infile, outfile, phrase_lengths):
    logger.info('processing file: {}...'.format(infile))
    with open(infile) as fin, open(outfile, 'w') as fout:
        for line in fin:
            word_list = line.split()
            if not word_list:
                fout.write('\n')
                continue
            phrase_list = generate_phrase_list_from_word_list(word_list, given_phrases, phrase_lengths)
            if phrase_list:
                fout.write('{}\n'.format(' '.join(phrase_list)))
    logger.info('has written into file: {}'.format(outfile))

def generate_phrase_list_from_word_list(word_list, given_phrases, phrase_lengths=PHRASE_LENGTHS):
    current_word_index = 0
    phrase_lengths = sorted(phrase_lengths, reverse=True)
    ret = []
    while current_word_index < len(word_list):
        found_new_phrase = False
        for phrase_length in phrase_lengths:
            if current_word_index + phrase_length <= len(word_list):
                new_phrase = '_'.join(word_list[current_word_index: current_word_index + phrase_length])
                if new_phrase in given_phrases:
                    found_new_phrase = True
                    ret.append(new_phrase)
                    current_word_index += phrase_length
                    break
        if not found_new_phrase:
            current_word_index += 1
    return ret

def generate_sentences_of_noun_phrases_at_dir_level(src_dir, dst_dir, chunksize=1, multiprocess=True):
    logger.info('generate_sentences_of_noun_phrases_at_dir_level...')
    multiprocess_dir(src_dir, dst_dir, generate_sentences_of_noun_phrases, chunksize=chunksize, multiprocess=multiprocess)

def generate_sentences_of_given_phrases_at_dir_level(src_dir, dst_dir, given_phrases_file, phrase_lengths=PHRASE_LENGTHS, chunksize=1, multiprocess=True):
    given_phrases = set()
    logger.info('loading {}...'.format(given_phrases_file))
    with open(given_phrases_file) as fin:
        for line in fin:
            given_phrases.add(line.strip())
    logger.info('has loaded {}'.format(given_phrases_file))
    multiprocess_dir(src_dir, dst_dir, generate_sentences_of_given_phrases, phrase_lengths, chunksize=chunksize, multiprocess=multiprocess, necessary_resources=given_phrases)

def train_word2vec_on_text(sentences_dir, model_path, size=SIZE, window=WINDOW, min_count=MIN_COUNT, workers=WORKERS, sample=SAMPLE):
    model = gensim.models.Word2Vec(None, size=size, window=window, min_count=min_count, workers=workers, sample=sample)
    logger.info('building vocabulary from: {}...'.format(sentences_dir))
    model.build_vocab(sentence_generator(sentences_dir))
    model.save(model_path)
    logger.info('training...')
    model.train(sentence_generator(sentences_dir))
    logger.info('saving model to: {}...'.format(model_path))
    model.save(model_path)
    logger.info('done')

def sentence_generator(sentences_dir):
    for dirpath, dirnames, filenames in os.walk(sentences_dir):
        for fname in filenames:
            full_path = os.path.join(dirpath, fname)
            with open(full_path) as fin:
                for line in fin:
                    yield line.split()

def extract_wiki_categories(category_file, outfile, encoding='latin1'):
    categories = set()
    with open(category_file, encoding=encoding) as f:
        lines = f.readlines()
    start_str = 'INSERT INTO `category` VALUES '
    category_regex = re.compile(r"(?<=\(\d+,').*?(?=',\d+,\d+,\d+\))")
    for line in lines:
        if line.startswith(start_str):
            for i in category_regex.findall(line):
                tmp = i.split('_')
                for j in range(len(tmp)):
                    tmp[j] = lemmatize(tmp[j].lower())
                tmp = '_'.join(tmp)
                categories.add(tmp)
    categories = sorted(categories)
    with open(outfile, 'w') as f:
        for i in categories:
            f.write('{}\n'.format(i))

# text rank data in task1
def prepare_wiki_article_for_textrank(infile, outfile_titles, outfile_contents):
    logger.info('infile: {}'.format(infile))
    with open(infile) as fin, open(outfile_titles, 'w') as fout_titles, open(outfile_contents, 'w') as fout_contents:
        current_paper = []
        first_line = True
        valid = None
        for line in fin:
            tmp = line.split()
            if tmp:
                if first_line:
                    if len(tmp) == 1:
                        valid = True
                        fout_titles.write(line)
                    else:
                        valid = False
                    first_line = False
                if valid:
                    for i in tmp:
                        current_paper.append(i)
                else:
                    continue
            else:
                first_line = True
                if current_paper:
                    # tmp = sorted(current_paper.items(), key=lambda x: x[1], reverse=True)
                    # tmp2 = []
                    # for k, v in tmp:
                        # tmp2.append(k)
                        # tmp2.append(str(v))
                    tmp2 = ' '.join(current_paper)
                    fout_contents.write('{}\n'.format(tmp2))
                    current_paper = []
    logger.info('DONE. infile: {}'.format(infile))

def prepare_wiki_article_for_textrank_MP(in_dir, out_dir_titles, out_dir_contents):
    os.makedirs(out_dir_titles, exist_ok=True)
    os.makedirs(out_dir_contents, exist_ok=True)
    params = []
    for i in os.listdir(in_dir):
        in_dir_i = os.path.join(in_dir, i)
        out_dir_i_titles = os.path.join(out_dir_titles, i)
        out_dir_i_contents = os.path.join(out_dir_contents, i)
        os.makedirs(out_dir_i_titles, exist_ok=True)
        os.makedirs(out_dir_i_contents, exist_ok=True)
        for j in os.listdir(in_dir_i):
            infile = os.path.join(in_dir_i, j)
            outfile_titles = os.path.join(out_dir_i_titles, j)
            outfile_contents = os.path.join(out_dir_i_contents, j)
            params.append((infile, outfile_titles, outfile_contents))
    p = mp.Pool()
    p.starmap(prepare_wiki_article_for_textrank, params)

# lda data in task1
def prepare_wiki_article_for_lda(infile, outfile_titles, outfile_contents):
    logger.info('infile: {}'.format(infile))
    with open(infile) as fin, open(outfile_titles, 'w') as fout_titles, open(outfile_contents, 'w') as fout_contents:
        current_paper = dd(int)
        first_line = True
        valid = None
        for line in fin:
            tmp = line.split()
            if tmp:
                if first_line:
                    if len(tmp) == 1:
                        valid = True
                        fout_titles.write(line)
                    else:
                        valid = False
                    first_line = False
                if valid:
                    for i in tmp:
                        current_paper[i] += 1
                else:
                    continue
            else:
                first_line = True
                if current_paper:
                    tmp = sorted(current_paper.items(), key=lambda x: x[1], reverse=True)
                    tmp2 = []
                    for k, v in tmp:
                        tmp2.append(k)
                        tmp2.append(str(v))
                    tmp2 = ' '.join(tmp2)
                    fout_contents.write('{}\n'.format(tmp2))
                    current_paper = dd(int)
    logger.info('DONE. infile: {}'.format(infile))

def prepare_wiki_article_for_lda_MP(in_dir, out_dir_titles, out_dir_contents):
    os.makedirs(out_dir_titles, exist_ok=True)
    os.makedirs(out_dir_contents, exist_ok=True)
    params = []
    for i in os.listdir(in_dir):
        in_dir_i = os.path.join(in_dir, i)
        out_dir_i_titles = os.path.join(out_dir_titles, i)
        out_dir_i_contents = os.path.join(out_dir_contents, i)
        os.makedirs(out_dir_i_titles, exist_ok=True)
        os.makedirs(out_dir_i_contents, exist_ok=True)
        for j in os.listdir(in_dir_i):
            infile = os.path.join(in_dir_i, j)
            outfile_titles = os.path.join(out_dir_i_titles, j)
            outfile_contents = os.path.join(out_dir_i_contents, j)
            params.append((infile, outfile_titles, outfile_contents))
    p = mp.Pool()
    p.starmap(prepare_wiki_article_for_lda, params)

def merge_wiki_article_for_lda(in_dir_titles, in_dir_contents, outfile_titles, outfile_contents):
    infiles = []
    for i in os.listdir(in_dir_titles):
        dir_i_titles = os.path.join(in_dir_titles, i)
        for j in os.listdir(dir_i_titles):
            infile_titles = os.path.join(dir_i_titles, j)
            infile_contents = os.path.join(in_dir_contents, i, j)
            infiles.append((infile_titles, infile_contents))
    titles = dd(list)
    for infile_titles, infile_contents in infiles:
        with open(infile_titles) as ft, open(infile_contents) as fc:
            ft_lines = ft.readlines()
            fc_lines = fc.readlines()
        for i in range(len(ft_lines)):
            ft_line = ft_lines[i].strip()
            fc_line = fc_lines[i].strip()
            titles[ft_line].append(fc_line)
    with open(outfile_titles, 'w') as ot, open(outfile_contents, 'w') as oc:
        for k, v in titles.items():
            v = ' '.join(v)
            ot.write('{}\n'.format(k))
            oc.write('{}\n'.format(v))

def extract_all_articles_of_given_titles(title_file, all_titles_file, all_contents_file, outfile):
    titles = []
    with open(title_file) as fin:
        for line in fin:
            titles.append(line.strip())
    all_articles = {}
    with open(all_titles_file) as f1, open(all_contents_file) as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
    for i in range(len(lines1)):
        all_articles[lines1[i].strip()] = lines2[i]
    with open(outfile, 'w') as fout:
        for t in titles:
            if t in all_articles:
                fout.write(all_articles[t])

def concatenate_files(filenames, outfile):
    with open(outfile, 'w') as fout:
        for fname in filenames:
            with open(fname) as fin:
                for line in fin:
                    fout.write(line)

def main():
    timestamp = '20170901' # the timestamp of wikidump data that you downloaded; ***CHANGE THIS AS NEEDED***
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--wiki_document_dir', default='./wikidata/wikidump/', help='the directory of preprocessed (by WikiExtractor) wikidump data')
    parser.add_argument('--wiki_document_cleaned_dir', default='./wikidata/wikidump_cleaned/')
    parser.add_argument('--category_file', default='./wikidata/enwiki-{}-category.sql'.format(timestamp), help='wiki category file downloaded using wiki_downloader.py')
    parser.add_argument('--category_titles_file', default='./wikidata/enwiki-{}-category-titles'.format(timestamp))
    parser.add_argument('--page_titles_file', default='./wikidata/enwiki-{}-all-titles-in-ns0'.format(timestamp))
    parser.add_argument('--wiki_titles_raw', default='./wikidata/enwiki-{}-pages-categories-titles'.format(timestamp))
    parser.add_argument('--wiki_titles_alphanumeric', default='./wikidata/enwiki-{}-pages-categories-titles-alphanumeric'.format(timestamp), help='this output file will contain all candidate topics (in the form of phrases)')
    parser.add_argument('--wiki_sentences_dir', default='./wikidata/wikidump_title_sentences/')
    parser.add_argument('--wiki_text_model_path', default='./wikidata/wiki_embeddings.model')

    args = parser.parse_args()

    # Wikipedia text
    if 1: # process raw text wiki files
        multiprocess_dir(wiki_document_dir, wiki_document_cleaned_dir, clean_raw_text_file)

    if 1: # extract all category titles
        logger.info('category_file...')
        extract_wiki_categories(category_file, category_titles_file)

    if 1: # concatenate two files of page titles and category titles
        concatenate_files([args.category_titles_file, args.page_titles_file], args.wiki_titles_raw)

    if 1: # process wiki titles
        chunksize = 1000
        logger.info('lower_and_remove_non_alphanumeric_wiki_title...')
        multiprocess_file(wiki_titles_raw, wiki_titles_alphanumeric, lower_and_remove_non_alphanumeric_wiki_title, chunksize, skip_head_line=False)
        logger.info('DONE.')
    if 1: # generate sentences of given phrases
        generate_sentences_of_given_phrases_at_dir_level(wiki_document_cleaned_dir, wiki_sentences_dir, wiki_titles_alphanumeric)
    if 1: # train_word2vec_on_wiki_text
        train_word2vec_on_text(wiki_sentences_dir, wiki_text_model_path)

# def main():
#     parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     # wiki_titles_raw = '/run/user/1024/enwiki-20160601-all-titles-in-ns0'
#     wiki_titles_raw = '/home/thuzhf/data/wikipedia/20161201/enwiki-20161201-titles-categories'
#     wiki_titles_alphanumeric = '/home/thuzhf/data/wikipedia/20161201/enwiki-20161201-titles-categories_alphanumeric'
#     wiki_titles_nonShort = '/home/thuzhf/data/wikipedia/20161201/enwiki-20161201-titles-categories_alphanumeric_nonShort'
#     wiki_titles_filteredByPosRules = '/home/thuzhf/data/wikipedia/20161201/enwiki-20161201-titles-categories_alphanumeric_nonShort_filteredByPosRules'
#     wiki_titles_filteredByPosAndTagRules = '/home/thuzhf/data/wikipedia/20161201/enwiki-20161201-titles-categories_alphanumeric_nonShort_filteredByPosAndTagRules'
#     if USE_FILTERED_WIKI_TITLES:
#         given_phrases_file = wiki_titles_filteredByPosAndTagRules
#     else:
#         given_phrases_file = wiki_titles_filteredByPosRules

#     wiki_document_dir = '/home/thuzhf/data/wikipedia/wikidump_20160601_document'
#     wiki_document_cleaned_dir = '/home/thuzhf/data/wikipedia/wikidump_20160601_document_cleaned'
#     wiki_document_cleaned_dir_new = '/home/thuzhf/data/wikipedia/wikidump_20160601_document_cleaned_new'
#     wiki_sentences_dir = '/home/thuzhf/data/wikipedia/wikidump_20161201_document_cleaned_{}to{}_words{}'.format(MIN_PHRASE_LENGTHS, MAX_PHRASE_LENGTHS, _FILTERED)
#     wiki_sentences_dir_new = '/home/thuzhf/data/wikipedia/wikidump_20161201_document_cleaned_new_{}to{}_words{}'.format(MIN_PHRASE_LENGTHS, MAX_PHRASE_LENGTHS, _FILTERED)
#     wiki_noun_phrases_sentences_dir = '/home/thuzhf/data/wikipedia/wikidump_20160601_document_noun_phrases_sentences'

#     wiki_text_model_path = '/home/thuzhf/work/KEG/FastKATE/model/wiki_titles_on_wikipedia_text/wiki_text_20161201_{}to{}{}_{}d.model'.format(MIN_PHRASE_LENGTHS, MAX_PHRASE_LENGTHS, _FILTERED, SIZE)
#     wiki_text_noun_phrases_model_path = '/home/thuzhf/work/KEG/FastKATE/model/noun_phrases_on_wiki_text/wiki_text_noun_phrases.model'

#     category_file = '/home/thuzhf/data/wikipedia/20161201/enwiki-20161201-category.sql'
#     category_titles_file = '/home/thuzhf/data/wikipedia/20161201/enwiki-20161201-category_titles'

#     wiki_sentences_dir_new_titles = '/home/thuzhf/data/wikipedia/wikidump_20161201_document_cleaned_new_titles_{}to{}_words{}'.format(MIN_PHRASE_LENGTHS, MAX_PHRASE_LENGTHS, _FILTERED)
#     wiki_sentences_dir_new_contents = '/home/thuzhf/data/wikipedia/wikidump_20161201_document_cleaned_new_contents_{}to{}_words{}'.format(MIN_PHRASE_LENGTHS, MAX_PHRASE_LENGTHS, _FILTERED)
#     wiki_sentences_dir_new_titles_for_textrank = '/home/thuzhf/data/wikipedia/wikidump_20161201_document_cleaned_new_titles_for_textrank_{}to{}_words{}'.format(MIN_PHRASE_LENGTHS, MAX_PHRASE_LENGTHS, _FILTERED)
#     wiki_sentences_dir_new_contents_for_textrank = '/home/thuzhf/data/wikipedia/wikidump_20161201_document_cleaned_new_contents_for_textrank_{}to{}_words{}'.format(MIN_PHRASE_LENGTHS, MAX_PHRASE_LENGTHS, _FILTERED)

#     # Wikipedia text
#     if 1: # process raw text wiki files
#         multiprocess_dir(wiki_document_dir, wiki_document_cleaned_dir, clean_raw_text_file)

#     if 1: # extract all category titles
#         logger.info('category_file...')
#         extract_wiki_categories(category_file, category_titles_file)

#     if 1: # process wiki titles
#         chunksize = 1000
#         logger.info('lower_and_remove_non_alphanumeric_wiki_title...')
#         multiprocess_file(wiki_titles_raw, wiki_titles_alphanumeric, lower_and_remove_non_alphanumeric_wiki_title, chunksize, skip_head_line=True)
#         # logger.info('lemmatize_and_remove_wiki_title_consisting_of_short_words...')
#         # multiprocess_file(wiki_titles_alphanumeric, wiki_titles_nonShort, lemmatize_and_remove_wiki_title_consisting_of_short_words, chunksize)
#         # logger.info('remove_wiki_title_by_*pos*_rules...')
#         # func = functools.partial(remove_wiki_title_by_pos_and_tag_rules, forbidden_tags=None)
#         # multiprocess_file(wiki_titles_nonShort, wiki_titles_filteredByPosRules, func, chunksize)
#         # logger.info('remove_wiki_title_by_*pos_and_tag*_rules...')
#         # func = functools.partial(remove_wiki_title_by_pos_and_tag_rules, forbidden_pos=None)
#         # multiprocess_file(wiki_titles_filteredByPosRules, wiki_titles_filteredByPosAndTagRules, remove_wiki_title_by_pos_and_tag_rules, chunksize)
#         logger.info('DONE.')
#     if 1: # generate sentences of given phrases
#         generate_sentences_of_given_phrases_at_dir_level(wiki_document_cleaned_dir, wiki_sentences_dir, wiki_titles_alphanumeric)
#     if 1: # train_word2vec_on_wiki_text
#         train_word2vec_on_text(wiki_sentences_dir, wiki_text_model_path)
#     if 0: # generate sentences of noun phrases
#         generate_sentences_of_noun_phrases_at_dir_level(wiki_document_dir, wiki_noun_phrases_sentences_dir)
#     if 0: # train_word2vec_on_wiki_text using noun phrases
#         train_word2vec_on_text(wiki_noun_phrases_sentences_dir, wiki_text_noun_phrases_model_path)

#     if 0: # transform AI papers' titles and abstracts into topics sentences
#         AI_titles_and_abstracts_dir = '/home/thuzhf/work/paper_crawler/data/AI'
#         AI_titles_and_abstracts_dir_cleaned = '/home/thuzhf/work/paper_crawler/data/AI_cleaned'
#         AI_titles_and_abstracts_sentences_dir = '/home/thuzhf/work/paper_crawler/data/AI_processed_sentences'
#         multiprocess_dir(AI_titles_and_abstracts_dir, AI_titles_and_abstracts_dir_cleaned, clean_raw_text_file)
#         generate_sentences_of_given_phrases_at_dir_level(AI_titles_and_abstracts_dir_cleaned, AI_titles_and_abstracts_sentences_dir, wiki_titles_alphanumeric)
#     if 0:
#         infile = '/home/thuzhf/work/paper_crawler/data/selected_papers.txt'
#         outfile = '/home/thuzhf/work/paper_crawler/data/selected_papers_cleaned.txt'
#         clean_raw_text_file(infile, outfile)
#     if 0:
#         in_dir = '/home/thuzhf/work/paper_crawler/data/selected_papers_cleaned'
#         out_dir = '/home/thuzhf/work/paper_crawler/data/selected_papers_topics_sentences'
#         generate_sentences_of_given_phrases_at_dir_level(in_dir, out_dir, wiki_titles_alphanumeric)

#     # # KDD text
#     # if 0:
#     #     kdd_pdf_text_cleaned_dir = '/home/thuzhf/data/papers/kdd/txt_cleaned'
#     #     kdd_pdf_text_cleaned_lemmatized_dir = '/home/thuzhf/data/papers/kdd/txt_cleaned_lemmatized'
#     #     kdd_papers_sentences_dir = '/home/thuzhf/data/papers/kdd/txt_cleaned_lemmatized_{}to{}_words{}'.format(MIN_PHRASE_LENGTHS, MAX_PHRASE_LENGTHS, _FILTERED)

#     #     kdd_papers_text_model_path = '/home/thuzhf/work/KEG/FastKATE/model/wiki_titles_on_kdd_text/kdd_papers_text_{}to{}{}_{}d.model'.format(MIN_PHRASE_LENGTHS, MAX_PHRASE_LENGTHS, _FILTERED, SIZE)

#     #     if 0: # process raw text paper files (converted from PDFs)
#     #         multiprocess_dir(kdd_pdf_text_cleaned_dir, kdd_pdf_text_cleaned_lemmatized_dir, clean_raw_text_file)
#     #     if 0: # generate sentences of given phrases
#     #         generate_sentences_of_given_phrases_at_dir_level(kdd_pdf_text_cleaned_lemmatized_dir, kdd_papers_sentences_dir, given_phrases_file, multiprocess=False)
#     #     if 0: # train_word2vec_on_kdd_text
#     #         train_word2vec_on_text(kdd_papers_sentences_dir, kdd_papers_text_model_path)


#     if 0:
#         multiprocess_dir(wiki_document_dir, wiki_document_cleaned_dir_new, clean_raw_text_file_and_add_newline_after_each_document)
#     if 0:
#         generate_sentences_of_given_phrases_at_dir_level(wiki_document_cleaned_dir_new, wiki_sentences_dir_new, wiki_titles_alphanumeric)
#     if 0:
#         prepare_wiki_article_for_lda_MP(wiki_sentences_dir_new, wiki_sentences_dir_new_titles, wiki_sentences_dir_new_contents)
#     if 0:
#         prepare_wiki_article_for_textrank_MP(wiki_sentences_dir_new, wiki_sentences_dir_new_titles_for_textrank, wiki_sentences_dir_new_contents_for_textrank)



#     merged_wiki_titles = '/home/thuzhf/data/wikipedia/merged_wiki_titles.txt'
#     merged_wiki_contents = '/home/thuzhf/data/wikipedia/merged_wiki_contents.txt'
#     merged_wiki_titles_for_textrank = '/home/thuzhf/data/wikipedia/merged_wiki_titles_for_textrank.txt'
#     merged_wiki_contents_for_textrank = '/home/thuzhf/data/wikipedia/merged_wiki_contents_for_textrank.txt'
#     if 0:
#         merge_wiki_article_for_lda(wiki_sentences_dir_new_titles, wiki_sentences_dir_new_contents, merged_wiki_titles, merged_wiki_contents)
#     if 0:
#         merge_wiki_article_for_lda(wiki_sentences_dir_new_titles_for_textrank, wiki_sentences_dir_new_contents_for_textrank, merged_wiki_titles_for_textrank, merged_wiki_contents_for_textrank)
#     if 0:
#         parser.add_argument('--title_file')
#         parser.add_argument('--outfile')
#         args = parser.parse_args()
#         extract_all_articles_of_given_titles(args.title_file, merged_wiki_titles_for_textrank, merged_wiki_contents_for_textrank, args.outfile)

if __name__ == '__main__':
    start_t = time.time()
    main()
    end_t = time.time()
    t = end_t - start_t
    print('Time elapsed: {:.4f} minutes'.format(t / 60.))