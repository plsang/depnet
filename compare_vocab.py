"""
Compare two vocabulary
"""

import os
import json
import argparse
import h5py
import numpy as np
import cv2
from random import shuffle, seed

import logging
from datetime import datetime

logger = logging.getLogger(__name__)
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--vocab1', default= '/net/per610a/export/das11f/plsang/codes/clcv/resources/data/Microsoft_COCO/mscoco2014_train_mydepsv4vocab_freq.json')
    parser.add_argument('--vocab2', default= '/net/per610a/export/das11f/plsang/codes/clcv/resources/data/Microsoft_COCO_20160302/mscoco2014_train_mydepsv4vocab_freq.json')

    args = parser.parse_args()
    start = datetime.now()
    
    print(args)
    
    logger.info('Load input file: %s', args.vocab1)
    v1 = json.load(open(args.vocab1))
    dv1 = [(v[1], v[2]) for v in v1]
    
    logger.info('Load input file: %s', args.vocab2)
    v2 = json.load(open(args.vocab2))
    dv2 = [(v[1], v[2]) for v in v2]
    
    top_k = 1000
    top_diff1 = [(i,v) for i,v in enumerate(dv1) if v not in dv2 and 'be/V' in v ]
    top_diff2 = [(i,v) for i,v in enumerate(dv2) if v not in dv1]
    print(1)       