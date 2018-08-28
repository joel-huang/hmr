#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 23:49:01 2018

@author: joel
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import cv2
import src.config
import numpy as np
from absl import flags
import tensorflow as tf
from src.util import image
from src.util import openpose
from src.RunModel import RunModel
from src.util import renderer as rnd

sys.path.append('/home/joel/Desktop/keras-yolo3')
try:
    from yolo import YOLO
except:
    print("cant import yolo")

flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_string('json_path', None,
                    'If specified, uses the openpose output to crop the image.')


def preprocess_image(img, json_path=None):
    
    # remove alpha channel
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if json_path is None:
        # check if higher of two sides is same as config size
        if np.max(img.shape[:2]) != config.img_size:
            print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = openpose.get_bbox(json_path)
    crop, proc_param = image.scale_and_crop(img, scale, center,
                                               config.img_size)
    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)
    return crop, proc_param, img

def get_render(img, proc_param, joints, verts, cam):
    """
    Renders the result in original image coordinate frame.
    """
    cam_for_render, vert_shifted, joints_orig = rnd.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])
    rend_img_overlay = renderer(
        vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
    return rend_img_overlay


config = flags.FLAGS
config(sys.argv)

# Using pre-trained model, change this to use your own.
config.load_path = src.config.PRETRAINED_MODEL
config.batch_size = 1
sess = tf.Session()
model = RunModel(config, sess=sess)
renderer = rnd.SMPLRenderer(face_path=config.smpl_face_path)
cam = cv2.VideoCapture(0)
yolo = YOLO()

while True:
    _, frame = cam.read()
    boxes, scores, classes = yolo.forwardpass(frame)
    yolo.close_session()
    
    # get highest scoring human bounding box
#    top_index = np.argmax(scores)
#    for index in xrange(len(scores)):
#        if index == top_index and classes[index] 
        
    input_img, proc_param, img = preprocess_image(frame)
    cv2.imshow('crop',input_img)
    # Add batch dimension: 1 x D x D x 3
    #input_img = np.expand_dims(input_img, 0)
    #joints, verts, cams, joints3d, theta = model.predict(
    #        input_img, get_theta=True)
    #overlay = get_render(frame, proc_param, joints[0], verts[0], cams[0])
    #cv2.imshow('x',overlay)
    if cv2.waitKey(1) == 27: 
        break  # esc to quit
cv2.destroyAllWindows()
#
#
