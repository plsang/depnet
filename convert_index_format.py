"""
Convert index format from string to number
"""

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
    parser.add_argument('--input_h5', type=str, default='/home/plsang/works/clcv/resources/data-20160708/Microsoft_COCO/mscoco2014_dev1_captions_exconceptsv3.h5', help='Input h5')
    
    parser.add_argument('--output_h5', type=str, default='/home/plsang/works/clcv/resources/data-20160708/Microsoft_COCO/mscoco2014_dev1_captions_exconceptsv3_depnet.h5', help='Output h5')
    
    args = parser.parse_args()
    start = datetime.now()
    
    fi = h5py.File(args.input_h5, "r")
    fo = h5py.File(args.output_h5, "w")
    
    fi.copy('data', fo)
    fi.copy('vocab', fo)
    
    new_index = np.array(fi['index']).astype(int)
    
    fo.create_dataset('index', data=new_index) 
    
    fi.close()
    fo.close()
    