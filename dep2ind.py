


import os
import json
import argparse
import h5py
import numpy as np

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def quote_str(str):
    return '"' + str + '"'
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--coco_data_root', default='/net/per610a/export/das11f/plsang/codes/clcv/resources/data/Microsoft_COCO')
    parser.add_argument('--concept_vocab', default='mscoco2014_train_myconceptsv3vocab.json')
    parser.add_argument('--dep_vocab', default='mscoco2014_train_mydepsv4vocab.json')
    parser.add_argument('--output_file', default='data/mscoco2014_train_depind.h5')
    
    args = parser.parse_args()
    start = datetime.now()
    
    
    
    logger.info('loading concept vocab: %s', args.concept_vocab)
    c = json.load(open(os.path.join(args.coco_data_root, args.concept_vocab)))
    logger.info('loading dependency vocab: %s', args.dep_vocab)
    d = json.load(open(os.path.join(args.coco_data_root, args.dep_vocab)))
    
    f = h5py.File(args.output_file, 'w')
    dep_indinces = f.create_dataset('ind', (2, len(d)), dtype='int')
    
    for i in range(len(d)):
        dep = json.loads(d[i])
        head = dep[1]
        dependent = dep[2]

        dep_indinces[0][i] = c.index(quote_str(head))
        dep_indinces[1][i] = c.index(quote_str(dependent))
        

    f.close()      
    logger.info('Time: %s', datetime.now() - start)
