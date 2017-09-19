#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging, os

def simple_logger(name=__name__, log_dir='../log', level=logging.DEBUG, use_console=True, use_file=True, fmt='%(asctime)s - %(name)s - %(levelname)s: %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if use_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter(fmt)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    if use_file:
        log_path = os.path.join(log_dir, '{}.log'.format(name))
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        formatter = logging.Formatter(fmt)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger