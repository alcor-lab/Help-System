# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 14:34:30 2019

@author: fiora
"""
import pickle
import os

"    Files to be loaded"
"    Transition matrices "
PKL_PATH = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(PKL_PATH, 'help_trans.pkl'), 'rb') as fp:
    help_trans = pickle.load(fp) 
    
    
with open(os.path.join(PKL_PATH, 'tech_trans_p.pkl'), 'rb') as fp:
    tech_trans = pickle.load(fp) 
    
with open(os.path.join(PKL_PATH,'aat_to_h_mat.pkl'), 'rb') as fp:
    aat_to_h_mat = pickle.load(fp)

with open(os.path.join(PKL_PATH,'hh_from_aat_mat.pkl'), 'rb') as fp:
    hh_from_aat_mat = pickle.load(fp)   

with open(os.path.join(PKL_PATH,'a2h.pkl'), 'rb') as fp:
    a2h =  pickle.load(fp)
    
with open(os.path.join(PKL_PATH, 'forward_actions.pkl'), 'rb') as fp:
    forward_actions = pickle.load(fp)
    
with open(os.path.join(PKL_PATH, 'forward_help.pkl'), 'rb') as fp:
    forward_help = pickle.load(fp)
    
with open(os.path.join(PKL_PATH, 'forward_help_comp.pkl'), 'rb') as fp:
    forward_help_comp = pickle.load(fp)
    
with open(os.path.join(PKL_PATH, 'consist_loc.pkl'), 'rb') as fp:
    consist_loc = pickle.load(fp)

with open(os.path.join(PKL_PATH, 'consist_obj.pkl'), 'rb') as fp:
    consist_obj = pickle.load(fp)    

"    Distributions   "
with open(os.path.join(PKL_PATH, 'pi_distrib_techA.pkl'), 'rb') as fp:
    pi_distrib_techA = pickle.load(fp)

with open(os.path.join(PKL_PATH, 'pi_distrib_help.pkl'), 'rb') as fp:
    pi_distrib_help = pickle.load(fp)   

"   Vocabularies "
with open(os.path.join(PKL_PATH, 'vocab_help.pkl'), 'rb') as fp:
    vocab_help = pickle.load(fp)  
    
with open(os.path.join(PKL_PATH, 'voc_actions.pkl'), 'rb') as fp:
    voc_actions = pickle.load(fp)  
    
"   Dictionaries  "
with open(os.path.join(PKL_PATH, 'id_to_word.pkl'), 'rb') as fp:
    id_to_word = pickle.load(fp)  
    
with open(os.path.join(PKL_PATH, 'word_to_id.pkl'), 'rb') as fp:
    word_to_id = pickle.load(fp)  
    
with open(os.path.join(PKL_PATH, 'voc_act_to_num.pkl'), 'rb') as fp:
    voc_act_to_num = pickle.load(fp)  

with open(os.path.join(PKL_PATH, 'num_to_voc_act.pkl'), 'rb') as fp:
    num_to_voc_act = pickle.load(fp)  
    
with open(os.path.join(PKL_PATH, 'num_to_voc_help.pkl'), 'rb') as fp:
    num_to_voc_help = pickle.load(fp)

with open(os.path.join(PKL_PATH, 'voc_help_to_num.pkl'), 'rb') as fp:
    voc_help_to_num= pickle.load(fp)
    
