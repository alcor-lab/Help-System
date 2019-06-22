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

CLASSES = ['BG','spray_bottle', 'screwdriver', 'torch', 'cloth', 'cutter', 
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

        self.shape = (1080, int(1080*4/3))
        self.class_list = CLASSES   
        with tf.device(device):
            self.latest_ckp = tf.train.latest_checkpoint(checkpoint_path) 
            self.model = modellib.MaskRCNN(mode="inference", config=InferenceConfig(), sess=self.sess)
            self.saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())
            self.saver.restore(self.sess, self.latest_ckp)

    def wheresWaldo(self, r, class_label_gt, person_index):
        rois = r['rois']
        class_ids = r['class_ids']
        class_label = [self.class_list[i] for i in class_ids]
        masks = r['masks']
        masks = np.transpose(masks, (2,0,1))

        diverter_index = class_label_gt.index('diverter')
        ladder_index = class_label_gt.index('ladder')
              
        

        mask_person = np.zeros(self.shape)
        mask_person = np.where(masks[person_index, :, :]==True,50, mask_person)

        diverter_bb = rois[class_label.index('diverter')]
        x1, x2, y1, y2 = diverter_bb
        mask_diverter = np.zeros(self.shape)
        mask_diverter[x1:y1, x2:y2] = 100

        mask_diverter_extended = np.zeros(self.shape)
        mask_diverter_extended[:, :y2] = 100

        diverter_person = mask_diverter + mask_person
        extended_diverter_person = mask_diverter_extended + mask_person

        num_p = len(np.where(mask_person>0)[0])

        s_ed_p = len(np.where(extended_diverter_person>100)[0]) * 100
        s_d_p = len(np.where(diverter_person>100)[0]) * 100

        if ladder_index in class_ids.tolist():
            ladder_bb = rois[class_label.index('ladder')]
            x1, x2, y1, y2 = ladder_bb
            ladder_bb = np.zeros(self.shape)
            ladder_bb[x1:y1, x2:y2] = 75
            ladder_person = ladder_bb + mask_person
            s_l_p = len(np.where(ladder_person>100)[0]) * 100
            if s_l_p > 1 and s_d_p >15:
                return "Technician on the Ladder"
        if s_d_p > 15 and s_ed_p > 70:
            return "Technician on the Ladder"
        if s_ed_p < 15:
            return "Technician next to Guard"
        '''
        if s_d_p < 15 and s_ed_p > 70:
            return "unsure_location"
        else:
            return [s_ed_p, s_d_p, s_l_p]
        '''
        return 'The Location is NOT sure (maybe under_diverter case)'

    def where_index(self, list_to_inspect, element):
        ret = []
        for index in range(len(list_to_inspect)):
            if list_to_inspect[index] == element:
                ret.append(index)
        return ret


    def detect(self, images):
        result = self.model.shDetect(images, verbose=1)
        output = {}
        r  = result[0]
        for i in range(len(r['class_ids'])):
            output[self.class_list[r['class_ids'][i]]] = r['scores'][i]
        

        #Technician Location
        r = result[1]
        class_ids = r['class_ids']
        class_label = [self.class_list[i] for i in class_ids]

        person_index = self.class_list.index('technician')
        diverter_index = self.class_list.index('diverter')
        ladder_index = self.class_list.index('ladder')
        
        if person_index not in class_ids.tolist():
            output['location'] = "No person"
        elif diverter_index not in class_ids.tolist():
            output['location'] = "No diverter"
        elif class_ids.tolist().count(person_index) == 1:
            index = class_ids.tolist().index(person_index)
            output['location'] = self.wheresWaldo(result[1], self.class_list, index)
        elif class_ids.tolist().count(person_index) > 1:
            person_indices = self.where_index(class_ids.tolist(), person_index)
            locations = []
            right_location = 'unsure_location'
            valid_locations = ["Technician on the Ladder",  "Technician next to Guard"]
            for index in person_indices:
                locations.append(self.wheresWaldo(result[1], self.class_list, index))
            for location in locations:
                if location in valid_locations:
                    right_location = location
            output['location'] = location

        return output, result
    