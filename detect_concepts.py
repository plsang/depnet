"""
Load and read images from COCO json file 
Save as h5 format
"""

import os
import json
import argparse
import h5py
import numpy as np
import cv2
from random import shuffle, seed
import pprint

import logging
from datetime import datetime

logger = logging.getLogger(__name__)
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--coco_data_root', default='/net/per610a/export/das11f/plsang/codes/clcv/resources/data/Microsoft_COCO', help='path to coco data root')
    #parser.add_argument('--input_file', default='data/Microsoft_COCO/mscoco2014_val_depnet_mydepsv4fc8.h5', help='Jsonl contains a list of video and its paths')
    parser.add_argument('--input_file', default='/net/per920a/export/das14a/satoh-lab/plsang/trecvidmed/feature/mydeps/depnet_fc8.h5', help='Jsonl contains a list of video and its paths')
    parser.add_argument('--vocab_file_task1', default='mscoco2014_train_myconceptsv3vocab.json', help='saving a copy of the vocabulary that was used for training')
    parser.add_argument('--vocab_file_task2', default='mscoco2014_train_mydepsv4vocab.json', help='saving a copy of the vocabulary that was used for training')
    parser.add_argument('--data', default='data', help='saving a copy of the vocabulary that was used for training')
    parser.add_argument('--test_ind', default=0, type=int, help='saving a copy of the vocabulary that was used for training')
    parser.add_argument('--test_vid', default=None, type=int, help='saving a copy of the vocabulary that was used for training')
    parser.add_argument('--concat', type=bool, default=False, help='saving a copy of the vocabulary that was used for training')
    
    args = parser.parse_args()
    start = datetime.now()
    
    #print(args)
    
    #logger.info('Load input file: %s', args.input_file)
    f = h5py.File(args.input_file, "r")
    
    #logger.info('Load vocab file: %s', args.vocab_file_task1)
    vocab1 = json.load(open(os.path.join(args.coco_data_root, args.vocab_file_task1)))
    
    #logger.info('Load vocab file: %s', args.vocab_file_task2)
    vocab2 = json.load(open(os.path.join(args.coco_data_root, args.vocab_file_task2)))
    
    labels1 = np.array(vocab1)
    labels2 = np.array(vocab2)
    
    if args.test_vid:
        ii = [i for (i,v) in enumerate(f['index']) if v == args.test_vid][0]
        ind = args.test_vid
    else:
        ii = args.test_ind
        ind = f['index'][ii]

    
    if args.concat:
        print('------ Concepts ------ ')
        feats = f[args.data][ii][0:1000]
        feats = feats/np.linalg.norm(feats, ord=2)
        indices = (-feats).argsort(axis=None)
        predicts = labels1[indices]
        pairs = [(vocab1[k], feats[k]) for k in indices[:10]]
        print(ind, pairs, '\n')

        print('------ Deps ------ ')
        feats = f[args.data][ii][1000:]
        feats = feats/np.linalg.norm(feats, ord=2)
        indices = (-feats).argsort(axis=None)
        predicts = labels2[indices]
        pairs = [(vocab2[k], feats[k]) for k in indices[:20]]
        pprint.pprint(pairs)
    else:
        print('------ Concepts/Deps ------ ')
        feats = f[args.data][ii]
        feats = feats/np.linalg.norm(feats, ord=2)
        indices = (-feats).argsort(axis=None)
        predicts = labels2[indices]
        pairs = [(vocab2[k], feats[k]) for k in indices[:20]]
        pprint.pprint(pairs)