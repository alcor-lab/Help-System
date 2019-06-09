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
    IMAGES_PER_GPU = 2
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

    def wheresWaldo(self, r, class_label_gt):
        rois = r['rois']
        class_ids = r['class_ids']
        class_label = [self.class_list[i] for i in class_ids]
        masks = r['masks']
        masks = np.transpose(masks, (2,0,1))

        person_index = class_label_gt.index('technician')
        diverter_index = class_label_gt.index('diverter')
        ladder_index = class_label_gt.index('ladder')
        if person_index not in class_ids.tolist():
            return "No person"
        if diverter_index not in class_ids.tolist():
            return "No diverter"

        mask_person = np.zeros((480,640))
        mask_person = np.where(masks[class_label.index('technician'), :, :]==True,50, mask_person)

        diverter_bb = rois[class_label.index('diverter')]
        x1, x2, y1, y2 = diverter_bb
        mask_diverter = np.zeros((480,640))
        mask_diverter[x1:y1, x2:y2] = 100

        mask_diverter_extended = np.zeros((480,640))
        mask_diverter_extended[:, :y2] = 100

        diverter_person = mask_diverter + mask_person
        extended_diverter_person = mask_diverter_extended + mask_person

        num_p = len(np.where(mask_person>0)[0])

        s_ed_p = len(np.where(extended_diverter_person>100)[0]) * 100
        s_d_p = len(np.where(diverter_person>100)[0]) * 100

        if ladder_index in class_ids.tolist():
            ladder_bb = rois[class_label.index('ladder')]
            x1, x2, y1, y2 = ladder_bb
            ladder_bb = np.zeros((480,640))
            ladder_bb[x1:y1, x2:y2] = 75
            ladder_person = ladder_bb + mask_person
            s_l_p = len(np.where(ladder_person>100)[0]) * 100
            if s_l_p > 1 and s_d_p >15:
                return "Technician on the Ladder"
        if s_d_p > 15 and s_ed_p > 70:
            return "Technician on the Ladder"
        if s_ed_p < 15:
            return "Technician next to Guard"
        if s_d_p < 15 and s_ed_p > 70:
            return "Technician under Diverter"
        else:
            return [s_ed_p, s_d_p, s_l_p]

    def detect(self, images):
        result = self.model.shDetect(images, verbose=1)
        output = {}
        r  = result[0]
        for i in range(len(r['class_ids'])):
            output[self.class_list[r['class_ids'][i]]] = r['scores'][i]

        #Technician Location
        output['location'] = self.wheresWaldo(result[1], self.class_list)

        return output, result
    