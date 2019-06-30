import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image as pi
from tf_mask_rcnn import utils
import tf_mask_rcnn.model as modellib
import tf_mask_rcnn.coco as coco
import tensorflow as tf

import time

import cv2

from sklearn.externals import joblib

CLASSES = ['BG','spray_bottle', 'screwdriver', 'torch', 'cloth', 'cutter', 
                            'pliers', 'brush', 'torch_handle', 'guard', 'ladder', 'closed_ladder', 
                                'guard-support', 'robot', 'technician', 'diverter' ]


MU_MULT=3.5
PIXEL_THRESHOLD = 200

PKL_PATH = os.path.dirname(os.path.abspath(__file__))

def createTestImageNp(imV):
    pixelList = []
    im = pi.fromarray(imV)
    imHSV = im.convert('HSV')
    imHsvV = np.array(imHSV)
    
    reshapeDim = imV.shape[0] * imV.shape[1]
    imV = np.reshape(imV, (reshapeDim, 3))
    imHsvV = np.reshape(imHsvV, (reshapeDim, 3))
    data = np.concatenate((imV, imHsvV), axis=1)
    
    return data

def point_count(frame, model):
    testImageV = np.array(frame)
    #testImageV = testImageV[...,::-1]
    testImageShape = testImageV.shape[:2]
    # start = time.monotonic()
    processedImage = createTestImageNp(testImageV)
    # print("1", time.monotonic()-start)
    output = np.exp(model.score_samples(processedImage))
    # print("2", time.monotonic()-start)
    output1 = np.divide(output, np.max(output))
    
    mu = np.mean(output1)
    mask_gmm  = output1>MU_MULT*mu
    # indexTT = np.where(output1>3.5*mu)
#    auxImm = np.zeros((480*640))
#
#    auxImm[indexTT] = 1
#    auxImm2 = np.reshape(auxImm, (480,640))
#    idx = np.where(auxImm2==1)
    return mask_gmm #indexTT


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

        
        self.gmm_model = None
        with open(os.path.join(PKL_PATH, 'saved_model_technician.pkl'),'rb') as f:
            self.gmm_model = joblib.load(f)

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
                return "on_the_ladder"
        if s_d_p > 15 and s_ed_p > 70:
            return "on_the_ladder"
        if s_ed_p < 15:
            return "at_guard_support"
        '''
        if s_d_p < 15 and s_ed_p > 70:
            return "unsure_location"
        else:
            return [s_ed_p, s_d_p, s_l_p]
        '''
        return 'under_diverter'

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
        
        output['debug'] = None
        if person_index not in class_ids.tolist():
            output['location'] = "no_technician"
            output['debug'] = "no_person_at_all"
        if diverter_index not in class_ids.tolist():
            output['location'] = "no_diverter"
        
        elif class_ids.tolist().count(person_index) >= 1:
            start = time.monotonic()
            person_indices = self.where_index(class_ids.tolist(), person_index)
            masks = r['masks']
            masks = np.transpose(masks, (2,0,1))
            # imagePilHSV = pi.fromarray(images[1]).convert('HSV')
            # imageHSVflat = np.reshape(np.array(imagePilHSV), (-1,3))
            shape_gmm = (240,320)
            frame = np.array(pi.fromarray(images[1]).resize(shape_gmm, resample=pi.BILINEAR))
            jacket_mask = point_count(frame, self.gmm_model)
            # jacket_mask_res =  np.array(np.reshape(jacket_mask, shape_gmm[::-1])*255).astype(np.uint8)
            # cv2.imshow('test gmm', jacket_mask_res)
            # cv2.waitKey(200)
            # print("TIME GMM:", time.monotonic() - start)
            technician_id = -1
            for p_elem in person_indices:
                person_mask = 1.0 * masks[p_elem, :, :]
                person_mask = np.array(pi.fromarray(person_mask).resize(shape_gmm, resample=pi.NEAREST))
                # person_mask =  np.reshape(person_mask, (480,640))
                person_mask = person_mask.flatten()
                num_pixels = np.sum(person_mask * jacket_mask, axis=-1)
                if num_pixels > PIXEL_THRESHOLD:
                    technician_id = p_elem
                # maschera = masks[p_elem, :, :].flatten()
                # #aux_zero = np.zeros(imageHSV.shape)
                # valid_pixels = imageHSVflat[maschera,...]
                # h = valid_pixels[:,0]
                # s = valid_pixels[:,1]
                # v = valid_pixels[:,2]
                # cond = (s>204) * (((h > 0) * (h<76)) + ((h>200) * (h<255))) * (v > 80)
                # remaining_pixel = np.sum(cond)
                # tot_pixel = np.sum(maschera, axis=None)# len(np.where(maschera==True))
                # fraction_jumper = remaining_pixel/tot_pixel
                # if fraction_jumper > 0.08:
                #     technician_id = p_elem
            if technician_id == -1:
                output['location'] = 'no_technician'
            else:
                output['location'] = self.wheresWaldo(result[1], self.class_list, technician_id)
            # print("EDOARDO'S CODE:", time.monotonic()-start)
    
        return output, result
    