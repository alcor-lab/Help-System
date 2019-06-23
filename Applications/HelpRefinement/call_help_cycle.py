# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:45:52 2019

@author: fiora
"""

import numpy as np 
import numpy.matlib
import pickle
import itertools

import json
import re
import sys

with open ('output_collection2.pkl', 'rb') as fp:
    guarda_out = pickle.load(fp)  

with open ('ordered_collection.pkl', 'rb') as fp:
    ordered_collection = pickle.load(fp)  
    
with open ('hh_from_aat_mat.pkl', 'rb') as fp:
    hh_from_aat_mat = pickle.load(fp)    
############## Just for calling from the collection    
keys_videos_net = []
for kk,vv in guarda_out.items():
    keys_videos_net.append(kk)
    
    
keys_videos_objs = []

for kk,vv in ordered_collection.items():
        keys_videos_objs.append(kk)
        
        
## sampling videos
"verifichiamo le corrispondenze fra ordered_collection (oggetti)"
"e i video con i softmax delle attivita'"
common_videos =  [i for i in keys_videos_net if i in keys_videos_objs]

tested =[]
num_videos = 0
"Let us choose a random video in the common set "    

while num_videos < len(common_videos):
    
    rand_var = np.random.randint(1,len(keys_videos_net))
    curr_video = keys_videos_net[rand_var]
    if not(curr_video in tested):
        num_videos =num_videos+1
        tested.append(curr_video)        
                
        
        "Extract the data of both the video and the object"
        gg_i = guarda_out[curr_video]    
        objs_i = ordered_collection[curr_video] 
        
        "recall that the last element of objs_i is not considered"
        "the key numbers corresponding  to the softmax of the chosen video"    
               
        keys_vv =[]
        for kk,vv in gg_i.items(): 
            keys_vv.append(kk)
            
        past_in_calls = 5   
        
        help_answer_stack = []
        help_pred_stack =[]
    #    fout = open("outputx1.txt", "w+")
        for jj in range(len(keys_vv)-past_in_calls):
           print(' time =', jj) 
           now_softmax=[]
           help_softmax=[]
           obj_state =[]
           where_is = []
           if jj ==0:
             help_pred_stack = []
             help_answer_stack =[]
           for q in range(jj,jj+past_in_calls):
               
               kks = keys_vv[q]
               all_vals = gg_i[kks]
               all_vals_objs = objs_i[kks]
               obs_now = all_vals["now_softmax"]
               obs_help = all_vals["help_softmax"]
               obs_obj = all_vals_objs["obj_label"]
               obs_locations = all_vals_objs["location_label"]
               now_softmax.append(obs_now)
               help_softmax.append(obs_help)
               obj_state.append(obs_obj)
               where_is.append(obs_locations)
           help_pred_stack = compute_help(help_answer_stack,now_softmax,help_softmax,obj_state,where_is,jj)
           print('help pred = ', help_pred_stack)
           if not(is_empty(help_pred_stack)):
              pred_c_help = help_pred_stack.get(1)
              if pred_c_help != 'nah':
                 help_answer_stack.append(pred_c_help)
    
     
       
#    fout.close()