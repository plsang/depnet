"""
Load and read videos from a VTT json file 
Save as h5 format
"""

import os
import json
import argparse
import h5py
import numpy as np
import glob
import cv2
from random import shuffle, seed
from scipy.misc import imread, imresize

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def read_img(image_file, image_height, image_width):
    im = imread(image_file)
    h = im.shape[0]
    w = im.shape[1]

    if h != image_height or w != image_width:
        logger.info(' resizing image: %s', image_file)	
        im = imresize(im, (image_height, image_width), 'bicubic') 
   
    return im
                    
def load_flow(params):
    
    data = json.load(open(params['input_json'], 'r'))
    
    sorted_videos = sorted(data['videos'], key=lambda x: x['id'])
    assert(len(sorted_videos) == len(data['videos']))
    num_videos = len(sorted_videos)
    
    image_width = params['image_width']
    image_height = params['image_height']
    f = h5py.File(params['output_h5'], 'w')
    indexes = []
    
    for iv, sorted_video in enumerate(sorted_videos):
        #for (img_id, img_file) in sorted_imgs:
        vid = sorted_video['id']  # integer video id
        video_id = sorted_video['video_id']
        indexes.append(str(vid))
        
        frame_dir_x = os.path.join(params['image_root'], video_id, params['flowx_dir'])
        frame_dir_y = os.path.join(params['image_root'], video_id, params['flowy_dir'])
        
        image_files_x = glob.glob(frame_dir_x + '/*.jpg')
        image_files_y = glob.glob(frame_dir_y + '/*.jpg')
        image_files = zip(image_files_x, image_files_y)
        
        assert(len(image_files_x) == len(image_files_y))
        num_images = 2*len(image_files_x)
        
        images = f.create_dataset(str(vid), (num_images, image_height, image_width), dtype='uint8')
        
        for i, image_file in enumerate(image_files):
            try:
                # opencv read image in BGR order
                # im = cv2.imread(image_file)
                # scipy read image in RGB order
                assert(os.path.basename(image_file[0]) == os.path.basename(image_file[1]))
                im_x = read_img(image_file[0], image_height, image_width)
                im_y = read_img(image_file[1], image_height, image_width)
            except:
                logger.info(' image not readable: %s. Generate random data', image_file)
                # generate random data
                im_x = np.random.randint(np.iinfo(np.uint8).max, size=(image_height, image_width))
                im_y = np.random.randint(np.iinfo(np.uint8).max, size=(image_height, image_width))

            
            images[2*i] = im_x.astype(np.uint8, copy=True)
            images[2*i+1] = im_y.astype(np.uint8, copy=True)
        
        logger.info('processed %d/%d (%.2f%% done) (%d images)' % (iv+1, num_videos, (iv+1)*100.0/num_videos, num_images))
    
    f.close()
    logger.info('Wrote image data to %s', params['output_h5']) 
    
    json.dump(indexes, open(params['output_json'], 'w'))
    logger.info('Wrote indexes to %s', params['output_json'])    

    
def load_rgb(params):
    
    data = json.load(open(params['input_json'], 'r'))
    
    sorted_videos = sorted(data['videos'], key=lambda x: x['id'])
    assert(len(sorted_videos) == len(data['videos']))
    num_videos = len(sorted_videos)
    
    image_width = params['image_width']
    image_height = params['image_height']
    f = h5py.File(params['output_h5'], 'w')
    indexes = []
    
    for iv, sorted_video in enumerate(sorted_videos):
        #for (img_id, img_file) in sorted_imgs:
        vid = sorted_video['id']  # integer video id
        video_id = sorted_video['video_id']
        indexes.append(str(vid))
        
        frame_dir = os.path.join(params['image_root'], video_id, params['rgb_dir'])
        image_files = glob.glob(frame_dir + '/*.jpg')

        image_files = image_files[::params['frame_sample']]
        num_images = len(image_files)
        images = f.create_dataset(str(vid), (num_images, 3, image_height, image_width), dtype='uint8')
        
        
        for i, image_file in enumerate(image_files):
            try:
                # opencv read image in BGR order
                # im = cv2.imread(image_file)
                # scipy read image in RGB order
                im = read_img(image_file, image_height, image_width)

            except:
                logger.info(' image not readable: %s. Generate random data', image_file)
                # generate random data
                im = np.random.randint(np.iinfo(np.uint8).max, size=(image_height, image_width, 3))
                
            # handle grayscale input images
            if len(im.shape) == 2:
                logger.info(' gray scale image: %s ', image_file)
                # im_rsz = I_rsz[:,:,np.newaxis]
                # im_rsz = np.concatenate((im, im, im), axis=2)
                im = np.tile(im[:,:,np.newaxis], (1,1,3))    
                
            # swap order of axes from (w, h, c) -> (c, w, h)
            im = np.transpose(im, axes = (2, 0, 1))
            
            images[i] = im.astype(np.uint8, copy=True)
        
        logger.info('processed %d/%d (%.2f%% done) (%d images)' % (iv+1, num_videos, (iv+1)*100.0/num_videos, num_images))
    
    f.close()
    logger.info('Wrote image data to %s', params['output_h5']) 
    
    json.dump(indexes, open(params['output_json'], 'w'))
    logger.info('Wrote indexes to %s', params['output_json'])    


########################    
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_json', required=True, help='COCO json file e.g., captions_val2014.json ')
    parser.add_argument('--output_h5', required=True, help='Output h5 file')
    parser.add_argument('--output_json', required=True, help='Output JSON file to store index')
    parser.add_argument('--image_root', default='', help='Location of image directory (rbg/flow images)')
    parser.add_argument('--image_height', default=240, type=int, help='Image width')
    parser.add_argument('--image_width', default=320, type=int, help='Image height')
    parser.add_argument('--type', type=str, default='rgb', choices=['rgb', 'flow'])
    parser.add_argument('--rgb_dir', type=str, default='rgb')
    parser.add_argument('--flowx_dir', type=str, default='flow_x')
    parser.add_argument('--flowy_dir', type=str, default='flow_y')
    parser.add_argument('--frame_sample', type=int, default=8, help='Select 1 every `frame_sample` interval')
    parser.add_argument('--seed', default=123, type=int, help='Random seed')
    
    args = parser.parse_args()
    params = vars(args) # convert to dictionary
    logger.info('Input params: %s', json.dumps(params, indent = 2))
    
    start = datetime.now()
    if args.type == 'rgb':
        load_rgb(params)
    else:
        load_flow(params)
        
    logger.info('Time: %s', datetime.now() - start)
