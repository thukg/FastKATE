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
logger.setLevel(logging.INFO)
from gensim.models import Word2Vec

from flask import Flask,request,jsonify
app = Flask(__name__)

from nltk.stem import WordNetLemmatizer
lemmatize = WordNetLemmatizer().lemmatize
lemmatize('')

taxonomy = None
w2v_model = None
msfos_data = None
acm_ccs_data = None

@app.route('/topics')
def topics():
    start_t = time.time()

    area_name = request.args.get('area', 'artificial_intelligence')
    area_name = normalize_name_for_querying_vector_model(area_name)
    context = request.args.get('context', 'computer_science')
    context = normalize_name_for_querying_vector_model(context)
    k = request.args.get('k', 10, int)
    depth = request.args.get('depth', 3, int)
    weighting_mode = request.args.get('weighting_mode', 0, int)
    use_category_structure = request.args.get('use_category_structure', 1, int)
    use_all_next_depth_subcats = request.args.get('use_all_next_depth_subcats', 0, int)
    use_msfos = request.args.get('use_msfos', 1, int)
    use_acm_ccs = request.args.get('use_acm_ccs', 1, int)
    # use_mcg = request.args.get('use_mcg', 0, int)
    # logger.debug('area_name: {}, k: {}'.format(area_name, k))

    ranked_scores = []
    if not use_category_structure:
        try:
            if context:
                ranked_scores = w2v_model.most_similar(positive=[area_name, context], topn=k)
            else:
                ranked_scores = w2v_model.most_similar(positive=[area_name], topn=k)
        except Exception as e:
            logger.debug(e)
    # elif not use_msfos and not use_mcg:
    else:
        if not use_all_next_depth_subcats:
            subcats = subcats_not_more_than_depth(area_name, depth, taxonomy)
            if use_msfos:
                subcats_from_msfos = subcats_not_more_than_depth_from_msfos(area_name, depth, msfos_data)
                for i in range(len(subcats_from_msfos)):
                    subcats[i].update(subcats_from_msfos[i])
            if use_acm_ccs:
                subcats_from_acm_ccs = subcats_not_more_than_depth_from_acm_ccs(area_name, depth, acm_ccs_data)
                for i in range(len(subcats_from_acm_ccs)):
                    subcats[i].update(subcats_from_acm_ccs[i])
        else: # deprecated
            subcats = get_subcats(area_name, taxonomy)
        logger.debug('subcats: {}'.format(subcats))

        weights = []
        for d in range(len(subcats)):
            weight_for_depth_d = weight_for_depth(d, weighting_mode)
            if weight_for_depth_d <= 0:
                break
            weights.append(weight_for_depth_d)
        logger.debug('weights: {}'.format(weights))

        for d in range(len(subcats)): # normalize_name_for_querying_vector_model and convert subcats[d] (from set) to list
            tmpcats = subcats[d]
            subcats[d] = []
            for c in tmpcats:
                subcats[d].append(normalize_name_for_querying_vector_model(c))
        logger.debug('subcats_normalized: {}'.format(subcats))

        scores = {}
        for d in range(len(subcats) - 1): # excluding root
            tmpcats = subcats[d + 1]
            for c in tmpcats:
                tmp_score = 0
                for d in range(len(weights)):
                    tmpcats2 = subcats[d]
                    for c2 in tmpcats2:
                        try:
                            if context:
                                tmp_score += w2v_model.n_similarity([c], [c2, context]) * weights[d]
                            else:
                                tmp_score += w2v_model.n_similarity([c], [c2]) * weights[d]
                        except Exception as e:
                            logger.debug(e)
                scores[c] = tmp_score
        ranked_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    # return jsonify(ranked_scores)
    end_t = time.time()
    t = end_t - start_t
    r = {'area': area_name, 'result': ranked_scores, 'time': t}
    return jsonify(r)

def weight_for_depth(d, weighting_mode):
    if weighting_mode == 1:
        return math.exp(4 - d)
    else: # this means 'root takes all weight'
        if d == 0:
            return 1
        else:
            return 0

def normalize_name_for_querying_vector_model(name):
    tmp = name.split('_')
    for i in range(len(tmp)):
        tmp[i] = lemmatize(tmp[i])
    name = '_'.join(tmp)
    return name

def load_pkl_file(infile):
    if not os.path.isfile(infile) or not os.path.getsize(infile):
        return None
    logger.info('loading data from {}...'.format(infile))
    with open(infile, 'rb') as f:
        data = pickle.load(f)
    logger.info('data loaded.')
    return data

def subcats_not_more_than_depth_from_msfos(area, depth, msfos_data):
    subcats = [set([area])]
    for i in range(depth):
        tmpcats = set()
        for j in subcats[-1]:
            if j in msfos_data:
                # tmp = [i[0] for i in msfos_data[j]['children']]
                # tmpcats.update(tmp)
                tmpcats.update(msfos_data[j])
        subcats.append(tmpcats)
    return subcats

def subcats_not_more_than_depth_from_acm_ccs(area, depth, data):
    subcats = [set([area])]
    for i in range(depth):
        tmpcats = set()
        for j in subcats[-1]:
            if j in data:
                tmpcats.update(data[j])
        subcats.append(tmpcats)
    return subcats

def subcats_not_more_than_depth(area, depth, taxonomy):
    subcats = [set([area])]
    for i in range(depth):
        tmpcats = set()
        for j in subcats[-1]:
            if j in taxonomy:
                tmpcats.update(taxonomy[j]['subcats'])
        subcats.append(tmpcats)
    return subcats

def subcats_and_subpages_not_more_than_depth(area, depth, taxonomy):
    subcats = [set([area])]
    subpages = []
    for i in range(depth):
        tmpcats = set()
        tmppages = set()
        for j in subcats[-1]:
            tmpcats.update(taxonomy[j]['subcats'])
            tmppages.update(taxonomy[j]['subpages'])
        subcats.append(tmpcats)
        subpages.append(tmppages)
    return subcats, subpages

def load_vector_model(vector_model):
    logger.info('loading vector model from {}...'.format(vector_model))
    w2v_model = Word2Vec.load(vector_model)
    logger.info('model loaded.')
    return w2v_model

def get_subcats(area, taxonomy):
    hierachy = [set(["area_of_computer_science"])]
    for i in range(10):
        tmpcats = set()
        for j in hierachy[-1]:
            if j in taxonomy:
                tmpcats.update(taxonomy[j]['subcats'])
        hierachy.append(tmpcats)

    #drop the duplicates
    for i in range(10):
        for j in range(i):
            hierachy[i] -= hierachy[j]

    subcats = [set([area])]
    for i in range(len(hierachy)):
        if area in hierachy[i]:
            break
    for j in range(i+1, len(hierachy)):
        subcats.append(hierachy[j])

    subcats2 = [set([area])]
    for i in range(len(subcats)-1):
        tmp = subcats[i+1]
        for j in range(i+1):
            tmp -= subcats[j]
        subcats2.append(tmp)
    return subcats2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--taxonomy_infile', default='./wikidata/taxonomy_lemmatized.pkl')
    parser.add_argument('--vector_model_infile', default='./wikidata/wiki_embeddings.model')
    parser.add_argument('--msfos_infile', default='', help='not used for the moment')
    parser.add_argument('--acm_ccs_infile', default='', help='not used for the moment')
    args = parser.parse_args()

    global taxonomy
    global w2v_model
    global msfos_data
    global acm_ccs_data
    taxonomy = load_pkl_file(args.taxonomy_infile)
    w2v_model = load_vector_model(args.vector_model_infile)
    msfos_data = load_pkl_file(args.msfos_infile)
    acm_ccs_data = load_pkl_file(args.acm_ccs_infile)

    app.run(host='0.0.0.0', port=15400, threaded=True)
    # app.run(host='127.0.0.1', port=15401, threaded=True)

if __name__ == '__main__':
    start_t = time.time()
    main()
    end_t = time.time()
    t = end_t - start_t
    print('Time elapsed: {:.4f} minutes'.format(t / 60.))