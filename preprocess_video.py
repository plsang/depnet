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

import logging
from datetime import datetime
from preprocess_image import upsample_image

logger = logging.getLogger(__name__)

def load_image(img_path, img_size):
    
    im_mean = np.array([[[ 103.939, 116.779, 123.68]]]) # in BGR order
        
    try:
        # note that opencv read image in BGR order
        im = cv2.imread(img_path)
        im = im.astype(np.float32, copy=True)
        im -= im_mean
        im_rsz = upsample_image(im, img_size)[0]
            
    except:
        logger.info(' image not readable: %s. Generate random data', img_path)
        # generate random data
        im_rsz = np.random.randint(np.iinfo(np.uint8).max, size=(img_size, img_size, 3))
        im_rsz -= im_mean

    # handle grayscale input images
    if len(im_rsz.shape) == 2:
        im_rsz = I_rsz[:,:,np.newaxis]
        im_rsz = np.concatenate((im, im, im), axis=2)
        
    # swap order of axes from (w, h, c) -> (c, w, h)
    im_rsz = np.transpose(im_rsz, axes = (2, 0, 1))
    
    return im_rsz
    
    
def preprocess_video(output_file, video_kf_dir, img_size=224, step=2):
    
    img_files = []
    for img_file in os.listdir(video_kf_dir):
        if img_file.endswith(".jpg"):
            img_files.append(img_file)
    img_files = sorted(img_files)
    
    sampling_range = range(0, len(img_files), step)
    num_images = len(sampling_range)
    
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    f = h5py.File(output_file, 'w')
    data = f.create_dataset('data', (num_images, 3, img_size, img_size), dtype='uint8')
    index = f.create_dataset('index', (num_images,), dtype='int')
    
    for (ii, ss) in enumerate(sampling_range):
        img_file = img_files[ss]
        img_path = os.path.join(video_kf_dir, img_file)
        im_rsz = load_image(img_path, img_size)
        
        data[ii] = im_rsz
        index[ii] = ii
        
    f.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_file', default='/net/per610a/export/das11f/plsang/trecvidmed/metadata/med_videos.jsonl', help='Jsonl contains a list of video and its paths')
    parser.add_argument('--video_dir', default='/net/per610a/export/das11f/plsang/trecvidmed/keyframes', help='LDC dir')
    parser.add_argument('--output_dir', default='/net/per610a/export/das11f/plsang/trecvidmed/preprocessed', help='Output directory')
    parser.add_argument('--img_size', default=565, type=int, help='224 for VGG, 565 for MSMIL')
    parser.add_argument('--step', default=2, type=int, help='Sampling step for extracting frames')
    
    args = parser.parse_args()
    start = datetime.now()
    
    logger.info('Load input file: %s', args.input_file)
    with open(args.input_file) as f:
        videos = [json.loads(line) for line in f]
    
    for ii in range(0, len(videos)):
        video = videos[ii]
        video_id = video['video_id']
        video_loc = video['location']
        video_kf_dir = os.path.join(args.video_dir, video_loc)
        
        logger.info('[%d/%d] Processing video %s', ii, len(videos), video_loc)    
        output_file = os.path.join(args.output_dir, video_loc + '.h5')
        if os.path.exists(output_file):
            logger.info('File existed: %s', output_file)
            continue    
            
        preprocess_video(output_file, video_kf_dir, img_size=args.img_size, step=args.step)
        
    logger.info('Wrote to %s', params['output_h5'])    
    logger.info('Time: %s', datetime.now() - start)
