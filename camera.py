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
from PIL import Image
import tensorflow as tf
from src.util import image
from src.util import openpose
from src.RunModel import RunModel
from src.util import renderer as rnd

sys.path.append('/yolov3')
try:
    from yolo import YOLO
except:
    print("cant import yolo")

flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_string('json_path', None,
                    'If specified, uses the openpose output to crop the image.')

HUMAN_CONFIDENCE_THRESHOLD = 0.9

def preprocess_image(img, bbox=None):
    
    # remove alpha channel
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if bbox is None:
        if np.max(img.shape[:2]) != config.img_size:
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2])/2).astype(int)
        center = center[::-1]

    else:
        t, l, b, r = [float(coord) for coord in bbox]
	min_pt = np.array([l, t])
	max_pt = np.array([r, b])
        person_height = np.linalg.norm(max_pt - min_pt)
        scale = 150. / person_height
        center = (min_pt + max_pt) / 2.
        print("scale,cetr")
        print(scale,center)

    crop, proc_param = image.scale_and_crop(img, scale, center,
                                               config.img_size)

    abnormal = crop
    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)
    return abnormal, crop, proc_param, img

def get_render(img, proc_param, joints, verts, cam):
    """
    Renders the result in original image coordinate frame.
    """
    cam_for_render, vert_shifted, joints_orig = rnd.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])
    rend_img_overlay = renderer(
        vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
    return rend_img_overlay

if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)

    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL
    config.batch_size = 1
    renderer = rnd.SMPLRenderer(face_path=config.smpl_face_path)
    cam = cv2.VideoCapture(0)

    yolo_sess = tf.Session()
    yolo = YOLO(yolo_sess=yolo_sess)

    hmr_graph = tf.Graph()
    hmr_sess = tf.Session(graph=hmr_graph)
    with hmr_graph.as_default():
        model = RunModel(config, sess=hmr_sess)

    while True:

        _, frame = cam.read()

        with yolo_sess.as_default():
            pil_frame = Image.fromarray(frame.astype('uint8'))
            print(pil_frame.size)
            overlay, boxes, scores, classes = yolo.detect_image(pil_frame)
<<<<<<< HEAD
#	    import ipdb;ipdb.set_trace()

=======

        try:
>>>>>>> e4b7d5c3dff9603b5125c687a508166f197299a6
            # all indices where humans are found
            human_indices = np.where(classes == 0)
            # find max score index of the humans
            winner_index = np.argmax([scores[human_indices]])
<<<<<<< HEAD
            bbox = boxes[winner_index] if scores[winner_index] >= HUMAN_CONFIDENCE_THRESHOLD else None

            if bbox is None:
                print("Sorry, no humans")
                continue
=======
            bbox = boxes[winner_index]

        except:
            print("Sorry, no humans")
>>>>>>> e4b7d5c3dff9603b5125c687a508166f197299a6

        with hmr_sess.as_default():
            abnormal, input_img, proc_param, img = preprocess_image(frame, bbox)
            # Add batch dimension: 1 x D x D x 3
            input_img = np.expand_dims(input_img, 0)
            joints, verts, cams, joints3d, theta = model.predict(
                                   input_img, get_theta=True)
            overlay = get_render(frame, proc_param, joints[0], verts[0], cams[0])
        cv2.imshow('x',overlay)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit

