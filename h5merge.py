import os
import json
import argparse
import h5py
import numpy as np

import logging
from datetime import datetime

logger = logging.getLogger(__name__)
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_file1')
    parser.add_argument('input_file2')
    parser.add_argument('output_file')
    parser.add_argument('--data', type=str, default='data')
    parser.add_argument('--index', type=str, default='index')

    args = parser.parse_args()
    start = datetime.now()    
    
    f1 = h5py.File(args.input_file1, 'r')
    f2 = h5py.File(args.input_file2, 'r')
    
    assert(args.data in f1 and args.data in f2)
    assert(args.index in f1 and args.index in f2)
    assert(f1[args.data].shape[0] == f1[args.index].shape[0])
    assert(f2[args.data].shape[0] == f2[args.index].shape[0])
    assert(f1[args.data].shape[1] == f2[args.data].shape[1])
    
    m1 = f1[args.data].shape[0]
    m2 = f2[args.data].shape[0]
    m = m1 + m2
    n = f1[args.data].shape[1]
    
    f = h5py.File(args.output_file, 'w')
    
    data = f.create_dataset(args.data, (m, n), dtype='float32')
    index = f.create_dataset(args.index, (m,), dtype='int32')
    
    logger.info('Copying from file 1: %s', args.input_file1)
    
    for i in range(f1[args.data].shape[0]):
        data[i] = f1[args.data][i]
        index[i] = f1[args.index][i]
        if i % 1000 == 0: 
            logger.info('%d processed', i)

    logger.info('Copying from file 2: %s', args.input_file2)
    for i in range(f2[args.data].shape[0]):
        data[i + m1] = f2[args.data][i]
        index[i + m1] = f2[args.index][i]
        if i % 1000 == 0: 
            logger.info('%d processed', i)
        
    f1.close()      
    f2.close()     
    f.close()      
    
    logger.info('Time: %s', datetime.now() - start)
