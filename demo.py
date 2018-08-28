"""
Demo of HMR.

Note that HMR requires the bounding box of the person in the image. 
The best performance is obtained when max length of the person in
the image is roughly 150px. 

When only the image path is supplied, it assumes that the image is
centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out
the bbox and the right scale factor.

Sample usage:

# On images on a tightly cropped image around the person
python -m demo --img_path data/im1963.jpg
python -m demo --img_path data/coco1.png

# On images, with openpose output
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel

flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')


def visualize(img, proc_param, joints, verts, cam):
    """
    Renders the result in original image coordinate frame.
    """
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    # Render results
    skel_img = vis_util.draw_skeleton(img, joints_orig)
    rend_img_overlay = renderer(
        vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
    rend_img = renderer(
        vert_shifted, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp1 = renderer.rotated(
        vert_shifted, 60, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp2 = renderer.rotated(
        vert_shifted, -60, cam=cam_for_render, img_size=img.shape[:2])
    
    print((skel_img.shape, rend_img_overlay.shape, rend_img.shape,
           rend_img_vp1.shape, rend_img_vp2.shape))

    import matplotlib.pyplot as plt
    # plt.ion()
    plt.figure(1)
    plt.clf()
    plt.subplot(231)
    plt.imshow(img)
    plt.title('input')
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(skel_img)
    plt.title('joint projection')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(rend_img_overlay)
    plt.title('3D Mesh overlay')
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(rend_img)
    plt.title('3D mesh')
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(rend_img_vp1)
    plt.title('diff vp')
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(rend_img_vp2)
    plt.title('diff vp')
    plt.axis('off')
    plt.draw()
    plt.show()
    # import ipdb
    # ipdb.set_trace()
    
def visualize_all(imgs, proc_params, all_joints, all_verts, all_cams):
    """
    Renders the result in original image coordinate frame.
    """
    for index in range(len(imgs)):
        cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
            proc_params[index], all_verts[index][0], all_cams[index][0],
            all_joints[index][0], img_size=imgs[index].shape[:2])

        if index == 0:
            skel_img = vis_util.draw_skeleton(imgs[index], joints_orig)
            rend_img_overlay = renderer(
                vert_shifted, cam=cam_for_render, img=imgs[index], do_alpha=True)
            rend_img = renderer(
                vert_shifted, cam=cam_for_render, img_size=imgs[index].shape[:2])
            rend_img_vp1 = renderer.rotated(
                vert_shifted, 60, cam=cam_for_render, img_size=imgs[index].shape[:2])
            rend_img_vp2 = renderer.rotated(
                vert_shifted, -60, cam=cam_for_render, img_size=imgs[index].shape[:2])
        else:
            skel_img = vis_util.draw_skeleton(skel_img, joints_orig)
            rend_img_overlay = renderer(
                vert_shifted, cam=cam_for_render, img=skel_img, do_alpha=True)
            rend_img = renderer(
                vert_shifted, cam=cam_for_render, img_size=imgs[index].shape[:2])
            rend_img_vp1 = renderer.rotated(
                vert_shifted, 60, cam=cam_for_render, img_size=imgs[index].shape[:2])
            rend_img_vp2 = renderer.rotated(
                vert_shifted, -60, cam=cam_for_render, img_size=imgs[index].shape[:2])

    import matplotlib.pyplot as plt
    # plt.ion()
    plt.figure(1)
    plt.clf()
    plt.subplot(231)
    plt.imshow(imgs[0])
    plt.title('input')
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(skel_img)
    plt.title('joint projection')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(rend_img_overlay)
    plt.title('3D Mesh overlay')
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(rend_img)
    plt.title('3D mesh')
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(rend_img_vp1)
    plt.title('diff vp')
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(rend_img_vp2)
    plt.title('diff vp')
    plt.axis('off')
    plt.draw()
    plt.show()
    # import ipdb
    # ipdb.set_trace()



def preprocess_image(img_path, json_path=None):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if json_path is None:
        if np.max(img.shape[:2]) != config.img_size:
            print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(json_path)

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img

def preprocess_multiple(img_path, json_path=None):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if json_path is None:
        raise Exception("No JSON.")
        
    else:
        scales_centers = op_util.get_bbox_all(json_path)

    crops_params_imgs = []

    for scale, center in scales_centers:
        crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)

        # Normalize image to [-1, 1]
        crop = 2 * ((crop / 255.) - 0.5)

        crops_params_imgs.append((crop, proc_param, img))
        
    return crops_params_imgs
    


def main(img_path, json_path=None):
    sess = tf.Session()
    model = RunModel(config, sess=sess)

    input_img, proc_param, img = preprocess_image(img_path, json_path)
    # Add batch dimension: 1 x D x D x 3
    input_img = np.expand_dims(input_img, 0)

    joints, verts, cams, joints3d, theta = model.predict(
        input_img, get_theta=True)

    visualize(img, proc_param, joints[0], verts[0], cams[0])


def multiple(img_path, json_path=None):
    sess = tf.Session()
    model = RunModel(config, sess=sess)

    all_joints = []
    all_verts = []
    all_cams = []

    #array of [input_img, proc_param, img]
    img_params = preprocess_multiple(img_path, json_path)
    
    for input_img, proc_param, img in img_params:
        # Add batch dimension: 1 x D x D x 3
        input_img = np.expand_dims(input_img, 0)

        joints, verts, cams, joints3d, theta = model.predict(
            input_img, get_theta=True)

        all_joints.append(joints)
        all_verts.append(verts)
        all_cams.append(cams)
        
    imgs = [tup[2] for tup in img_params]
    proc_params = [tup[1] for tup in img_params]
    visualize_all(imgs, proc_params, all_joints, all_verts, all_cams)

if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    main(config.img_path, config.json_path)
    #multiple(config.img_path, config.json_path)