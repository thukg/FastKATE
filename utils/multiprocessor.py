#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Fang Zhang <thuzhf@gmail.com>

import sys,os,json,gzip,math,time,datetime,random,copy
import functools,itertools,requests,pickle,configparser
import argparse,logging,uuid,shutil,collections
from urllib.parse import urlparse, parse_qs
from collections import defaultdict as dd
import multiprocessing as mp
from threading import Thread
import traceback

THREADS_NUM_MAX = 3500

class Worker(mp.Process):
    """ 
    1.The worker is in the same process with its caller.
    2.The worker's run() method will be invoked in a new seperate process once it's start() method being called.
    3.So the worker will only be allocated new memory for its referenced resources in its run() method's process.
    4.You shouldn't put heavily time/cpu/memory consuming code in __init__(); put them in run().
    """
    def __init__(self, job_params_queue, results_queue, necessary_resources, handler_func):
        super().__init__()
        self.job_params_queue = job_params_queue # job_params_queue can be mp.SimpleQueue or mp.Queue
        self.results_queue = results_queue
        self.necessary_resources = necessary_resources
        self.handler_func = handler_func

    def run(self):
        # every worker won't stop running until it gets a 'None' signal, so if you have k workers, 
        # you should send k 'None' signals in the job_params_queue in order to close them all correctly.
        if self.necessary_resources:
            for job_params in iter(self.job_params_queue.get, None):
                ans = self.handler_func(self.necessary_resources, *job_params)
                self.results_queue.put(ans)
        else:
            for job_params in iter(self.job_params_queue.get, None):
                ans = self.handler_func(*job_params)
                self.results_queue.put(ans)

        # this means the worker has been closed (unnecessary because every result maps to a different input)
        # self.results_queue.put(None)

class MPRoutine(object):
    """
    1.This is a wrapper class for common steps of calling class:Worker
    2.If has necessary_resources, the parameters of handler_func should be of the form: handler_func(necessary_resources, *params);
      otherwise the parameters of handler_func should be of the form: handler_func(*params), where params are input to self.put().
    """
    def __init__(self, handler_func, necessary_resources=None, num_workers=mp.cpu_count()):
        super(MPRoutine, self).__init__()
        self.necessary_resources = necessary_resources
        self.handler_func = handler_func
        self.num_workers = num_workers
        self.workers = []
        self.job_params_queue = mp.SimpleQueue()
        self.results_queue = mp.SimpleQueue() # You can also use seperate queues for seperate workers
        self.num_total_jobs = 0
        self.num_done_jobs = 0
        # necessary_resources = ...
        # handler_func = ...
        for i in range(self.num_workers):
            self.workers.append(Worker(self.job_params_queue, self.results_queue, self.necessary_resources, self.handler_func))
        for w in self.workers:
            w.start()

    def _put(self, params): # should be non-block; otherwise might result in deadlock
        # self.job_params_queue.put(params)
        Thread(target=self.job_params_queue.put, args=(params, )).start() # using Thread() instead of Process()
        self.num_total_jobs += 1

    def _get(self): # get one single result
        if self.num_done_jobs < self.num_total_jobs:
            r = self.results_queue.get()
            self.num_done_jobs += 1
            return r
        else:
            return None

    def _has_some_results(self):
        return not self.results_queue.empty()

    def _get_all_remaining_results(self):
        while self.num_done_jobs < self.num_total_jobs:
            r = self.results_queue.get()
            self.num_done_jobs += 1
            yield r

    def results(self, job_params): # return a list, but not a generator
        num_jobs = len(job_params)
        job_params = iter(job_params)
        ans = []
        next_job_index = 0
        num_results_got = 0
        while num_results_got < num_jobs:
            if self._has_some_results() or next_job_index - num_results_got >= THREADS_NUM_MAX:
                r = self._get()
                ans.append(r)
                num_results_got += 1
            else:
                if next_job_index < num_jobs:
                    # self._put(job_params_list[next_job_index])
                    self._put(next(job_params))
                    next_job_index += 1
                else:
                    for r in self._get_all_remaining_results():
                        ans.append(r)
                        num_results_got += 1
        return ans

    def close(self):
        for i in range(self.num_workers):
            self._put(None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        if exc_type:
            traceback.print_exc()
            sys.exit(-1)

def main():
    pass

if __name__ == '__main__':
    start_t = time.time()
    main()
    end_t = time.time()
    t = end_t - start_t
    print('Time elapsed: {:.4f} minutes'.format(t / 60.))