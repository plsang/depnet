"""
Load and read images from COCO json file 
Save as h5 format
"""

import os
import json
import argparse
import h5py
import numpy as np
from scipy.misc import imread, imresize
from random import shuffle, seed

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def load_image(params):
    data = json.load(open(params['input_json'], 'r'))
    
    # sort images by image id (acsending)
    img_dict = {}
    for img in data['images']:
        img_id = img['id']
        img_file = img['file_name']
        img_dict[img_id] = img_file
    
    sorted_imgs = sorted(img_dict.items(), key=lambda x: x[0])
    assert(len(sorted_imgs) == len(data['images']))
    num_images = len(sorted_imgs)
    
    # shuffle images
    index_shuffle = range(num_images)
    seed(params['seed'])      # fix the random seed
    shuffle(index_shuffle) 
    
    # create output h5 file
    
    img_size = params['images_size']
    f = h5py.File(params['output_h5'], 'w')
    images = f.create_dataset('images', (num_images, 3, img_size, img_size), dtype='uint8')
    indexes = f.create_dataset('index', (num_images,), dtype='int')
    f.create_dataset('index_shuffle', (num_images,), dtype='int', data=index_shuffle)
    
    for i, shuffled_i in enumerate(index_shuffle):
        #for (img_id, img_file) in sorted_imgs:
        img_id = sorted_imgs[shuffled_i][0]
        img_file = sorted_imgs[shuffled_i][1]
        
        I = imread(os.path.join(params['images_root'], img_file))
        try:
            I_rsz = imresize(I, (img_size, img_size))
        except:
            logger.info(' image not readable: %s. Generate random data', img_file)
            # generate random data
            I_rsz = np.random.randint(np.iinfo(np.uint8).max, size=(img_size, img_size, 3))

        # handle grayscale input images
        if len(I_rsz.shape) == 2:
            I_rsz = I_rsz[:,:,np.newaxis]
            I_rsz = np.concatenate((I_rsz, I_rsz, I_rsz), axis=2)
        
        # swap order of axes from (w, h, c) -> (c, w, h)
        I_rsz = I_rsz.transpose(2, 0, 1)
        
        images[i] = I_rsz
        indexes[i] = img_id
        if i % 1000 == 0:
            logger.info('processing %d/%d (%.2f%% done)' % (i, num_images, i*100.0/num_images))
    
    f.close()       

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_json', required=True, help='COCO json file e.g., captions_val2014.json ')
    parser.add_argument('--output_h5', required=True, help='Output h5 file')
    parser.add_argument('--images_root', default='', help='Location of COCO image directory')
    parser.add_argument('--images_size', default=227, type=int, help='Location of COCO image directory')
    parser.add_argument('--seed', default=123, type=int, help='Random seed')
    
    args = parser.parse_args()
    params = vars(args) # convert to dictionary
    print json.dumps(params, indent = 2)
    
    start = datetime.now()
    load_image(params)
    logger.info('Wrote to %s', params['output_h5'])    
    logger.info('Time: %s', datetime.now() - start)
