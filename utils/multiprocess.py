#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Fang Zhang <thuzhf@gmail.com>

import os, time
import multiprocessing as mp
from FastKATE.utils.multiprocessor import MPRoutine

def multiprocess_dir(src_dir, dst_dir, function, *args, chunksize=1, recursive=True, dst_file_suffix = '', multiprocess=True, necessary_resources=None):
    # if dst_dir == '', function will not write to file
    params = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(src_dir, followlinks=True):
            dirpath_dst = dirpath.replace(src_dir, dst_dir, 1)
            if dst_dir:
                os.makedirs(dirpath_dst, exist_ok=True)
            for fname in filenames:
                fname_src = os.path.join(dirpath, fname)
                fname_dst = os.path.join(dirpath_dst, fname + dst_file_suffix)
                params.append((fname_src, fname_dst, *args))
    else:
        filenames = next(os.walk(path, followlinks=True))[2]
        if dst_dir:
            os.makedirs(dst_dir, exist_ok=True)
        for fname in filenames:
            fname_src = os.path.join(src_dir, fname)
            fname_dst = os.path.join(dst_dir, fname + dst_file_suffix)
            params.append((fname_src, fname_dst, *args))
    if multiprocess:
        # p = mp.Pool()
        # return p.starmap(function, params, chunksize)
        with MPRoutine(function, necessary_resources=necessary_resources) as mprt:
            return mprt.results(params)
    else:
        ans = []
        for p in params:
            ans.append(function(*p))
        return ans

def multiprocess_file(infile, outfile, function, chunksize=1, skip_head_line=False, allow_duplicate=False):
    p = mp.Pool()
    if not allow_duplicate:
        unduplicated_lines = set()
    with open(infile) as fin, open(outfile, 'w') as fout:
        if skip_head_line:
            next(fin)
        for processed_line in p.imap_unordered(function, fin, chunksize):
            if allow_duplicate:
                fout.write(processed_line)
            elif not processed_line in unduplicated_lines:
                fout.write(processed_line)
                unduplicated_lines.add(processed_line)

def main():
    pass

if __name__ == '__main__':
    start_t = time.time()
    main()
    end_t = time.time()
    t = end_t - start_t
    print('Time elapsed: {:.4f} minutes'.format(t / 60.))