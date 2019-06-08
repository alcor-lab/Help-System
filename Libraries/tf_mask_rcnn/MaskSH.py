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
from tf_mask_rcnn import visualize
import tf_mask_rcnn.coco as coco
import tensorflow as tf

class MaskSH:

    def __init__(self, checkpoint_path):
        self.s = tf.Session()
        self.class_list  = ['BG','spraybottle', 'screwdriver', 'torch', 'cloth', 'cutter', 'pliers', 'brush', 'torch_handle', 'guard', 'ladder', 'closed_ladder', 'guard-support', 'robot', 'technician', 'diverter' ]

        class InferenceConfig(coco.CocoConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            NUM_CLASSES = len(self.class_list)

        print(checkpoint_path)
        self.latest_ckp = tf.train.latest_checkpoint(checkpoint_path)
        self.config = InferenceConfig()
        self.model = modellib.MaskRCNN(mode="inference", config=self.config)
        self.saver = tf.train.Saver()
        self.s.run(tf.global_variables_initializer())
        self.saver.restore(self.s, self.latest_ckp)

    def detect(self, image):
        result = self.model.shDetect(image, self.s, verbose=1)
        output = {}
        r  = result[0]
        for i in range(len(r['class_ids'])):
            output[self.class_list[r['class_ids'][i]]] = r['scores'][i]
        return output, result
