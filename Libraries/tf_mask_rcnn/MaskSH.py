import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from tf_mask_rcnn import utils
import tf_mask_rcnn.model as modellib
import tf_mask_rcnn.coco as coco
import tensorflow as tf

CLASSES = ['BG','spraybottle', 'screwdriver', 'torch', 'cloth', 'cutter', 
                            'pliers', 'brush', 'torch_handle', 'guard', 'ladder', 'closed_ladder', 
                                'guard-support', 'robot', 'technician', 'diverter' ]

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = len(CLASSES)

class MaskSH:

    def __init__(self, checkpoint_path, sess=None, device=None):
        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess
    
        self.class_list = CLASSES   
        with tf.device(device):
            self.latest_ckp = tf.train.latest_checkpoint(checkpoint_path) 
            self.model = modellib.MaskRCNN(mode="inference", config=InferenceConfig(), sess=self.sess)
            self.saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())
            self.saver.restore(self.sess, self.latest_ckp)

    def detect(self, images):
        result = self.model.shDetect(images, verbose=1)
        output = {}
        r  = result[0]
        for i in range(len(r['class_ids'])):
            output[self.class_list[r['class_ids'][i]]] = r['scores'][i]
        return output, result
    