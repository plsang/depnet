"""
Load and read images from COCO json file 
Save as h5 format
"""

import os
import json
import argparse
import h5py
import numpy as np
#from scipy.misc import imread, imresize
import cv2
from random import shuffle, seed

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def upsample_image(im, sz):
    """
    this function is copied from Gupta's code
    Note: Caffe also supports function to resize an image in stretch mode, while this function does not stretch the image.
    """
    h = im.shape[0]
    w = im.shape[1]
    s = np.float(max(h, w))
    I_out = np.zeros((sz, sz, 3), dtype = np.float)
    I = cv2.resize(im, None, None, fx = np.float(sz)/s, fy = np.float(sz)/s, interpolation=cv2.INTER_LINEAR)
    SZ = I.shape
    I_out[0:I.shape[0], 0:I.shape[1],:] = I
    return I_out, I, SZ


def load_image(params):
    
    im_mean = np.array([[[103.939, 116.779, 123.68]]]) # in BGR order
    
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
    images = f.create_dataset('images', (num_images, 3, img_size, img_size), dtype='int16')
    indexes = []
    f.create_dataset('index_shuffle', (num_images,), dtype='int', data=index_shuffle)
    
    for i, shuffled_i in enumerate(index_shuffle):
        #for (img_id, img_file) in sorted_imgs:
        img_id = sorted_imgs[shuffled_i][0]
        img_file = sorted_imgs[shuffled_i][1]
            
        try:
            # Scipy funciton read image in RGB order
            # I = imread(os.path.join(params['images_root'], img_file))
            # I_rsz = imresize(I, (img_size, img_size))
            
            # note that opencv read image in BGR order
            im = cv2.imread(os.path.join(params['images_root'], img_file))
            im = im.astype(np.float32, copy=True)
            im -= im_mean
            im_rsz = upsample_image(im, img_size)[0]
            
        except:
            logger.info(' image not readable: %s. Generate random data', img_file)
            # generate random data
            im_rsz = np.random.randint(np.iinfo(np.uint8).max, size=(img_size, img_size, 3))
            im_rsz -= im_mean

        # handle grayscale input images
        if len(im_rsz.shape) == 2:
            im_rsz = I_rsz[:,:,np.newaxis]
            im_rsz = np.concatenate((im, im, im), axis=2)
        
        # swap order of axes from (w, h, c) -> (c, w, h)
        im_rsz = np.transpose(im_rsz, axes = (2, 0, 1))
        
        images[i] = im_rsz
        indexes.append(str(img_id))
        if i % 1000 == 0:
            logger.info('processing %d/%d (%.2f%% done)' % (i, num_images, i*100.0/num_images))
    
    f.close()
    logger.info('Wrote image data to %s', params['output_h5']) 
    
    json.dump(indexes, open(params['output_json'], 'w'))
    logger.info('Wrote indexes to %s', params['output_json'])    

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_json', required=True, help='COCO json file e.g., captions_val2014.json ')
    parser.add_argument('--output_h5', required=True, help='Output h5 file')
    parser.add_argument('--output_json', required=True, help='Output JSON file to store index')
    parser.add_argument('--images_root', default='', help='Location of COCO image directory')
    parser.add_argument('--images_size', default=224, type=int, help='Location of COCO image directory')
    parser.add_argument('--seed', default=123, type=int, help='Random seed')
    
    args = parser.parse_args()
    params = vars(args) # convert to dictionary
    print json.dumps(params, indent = 2)
    
    start = datetime.now()
    load_image(params)
    logger.info('Time: %s', datetime.now() - start)
