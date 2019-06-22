# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 13:40:29 2019

@author: fiora
"""

import numpy as np 
import numpy.matlib
import pickle
import itertools
from itertools import groupby
import pandas as pd

import json
import re
import sys

import operator

from help_refinement.loadPars import *

"The query algorithm:"
" h0. get 5 past second of softmax: now next help: take argmax and compute the gradient"
"    if for now the mean is high and cov is low, and stab is high it is a pick choice: " 
"       if the choice == sil for the last 3 sec. -->                                "
"                                            don't know help (wait for next input)  "
"       if the choice =/= sil then                                                  "
"       hx.   choose, among the 5 sec, the help with max mean, min cov, max stability"
"             compute the transition from the previous help                         " 
"             compute transition from chosen actions                                "                                                             
"             if transition OK then choose this help and pass it                    " 
"             if transition not OK then                                             "
"               if the gradient of chosen help  was high wait for next input        "
"               if the gradientis of chosen help was not high then go to hx.        "
"  if the mean for now is not high or covariance is high or stability is low:       "
"  look for next most probable combination of help coherent with stable prediction  "      
      
"** stability is 1-number of unique items/ total items"

# =============================================================================
#  load parameters -->see  loadPars.py
# =============================================================================
# 
# =============================================================================
# =============================================================================
#                             Utilities
# =============================================================================
def get_names(data,vocab):   
#           ll = counts_A[:,0]
        vv = [np.int(x) for x in data]
        names_for_data = [vocab[xx]  for xx in vv] 
        return names_for_data
    

def find_in_L(item, listN):
    indexes = [i for i,x in enumerate(listN) if x == item] 
    return indexes

def join_vals(item):
       return (' '.join(item))
       

def is_empty(xx):
    return len(xx)==0

" Just computes a reasonable argmax --> next to be added after first exp. "


def repmatf(arrayX,row,col):
    repp=np.tile(arrayX, (row,col))
    return repp

def get_max(obs_now,n):
    hhx = np.zeros((n,1))
    for ii in range(n): 
        hhx[ii,:] = np.max(obs_now[ii,:])
    return hhx

def normalise_v(val):
    nn = val.shape[1]
    hhv = repmatf(get_max(val,len(val)),1,nn)
    val1 = np.zeros((val.shape))
    for jj in range(len(val)):
        ww = np.divide(val[jj,:],hhv[jj,:])**3
        qq = np.sum(ww)
        val1[jj,:] = np.divide(ww,qq)
    return val1

"search for list of actions in transition schema actions --> help forward_actions"
"return the min and if there are two equal returns the first"
def look_forward(queries,forward_actions):
    yy = []
    idx = []
    b = np.array(['sil'])    
    queries1 = np.setdiff1d(queries,b)
    for jj in range(len(forward_actions)):
        xx = forward_actions[jj] 
        found_i = set.issubset(getset(queries1),getset(xx))
        if found_i:
            uu = [q for q in xx if q!='sil']
            yy.append(uu)
            idx.append(jj)

    if not(is_empty(yy)):
       best_list = min(yy, key =len)
       l_best = yy.index(best_list)
       return best_list, idx[l_best]
    else:
       return [],[] 
   
#
def show_net_argmax(data_act,data_help):
    counts1,pp = compute_argmax_max(data_act,0)
    actions = [id_to_word[x] for x in counts1[:,0]]
    print('recognized actions: ', actions, counts1)
    counts2,pp = compute_argmax_max(data_help,0)
    uu =(-counts2[:,1]).argsort()[:3] 
    helpX = [id_to_word[x] for x in counts2[uu,0]]
    print('predicted help: ', helpX,uu)
    
    
def compute_argmax_max(data,choice):
    qxdata_flat =  list(itertools.chain(*data))
    qydata_flat  = np.array([np.array(xi) for xi in qxdata_flat])
    idx = np.zeros(qydata_flat.shape[0], dtype =int)
    if choice == 0:
        for jj in range(len(idx)):
            idx[jj] = np.argmax(qydata_flat[jj,:])
    if choice ==1:
        mu = np.mean(qydata_flat)
        idx = np.where(qydata_flat>2*mu)[1]
    freq0 = np.bincount(idx)
    ii = np.nonzero(freq0)[0]   ## positions where counts are non-zero
    counts = np.vstack((ii,freq0[ii])).T
    pp = np.divide(counts[:,1],idx.shape[0])
    return counts, pp

def recompose_list_into_help(net_predicted_help_L):
    found_corr = []
    for jj in range(len(vocab_help)):
        x = vocab_help[jj].split()
        val = set.intersection(getset(net_predicted_help_L),getset(x))
        if getset(val) == getset(net_predicted_help_L):
            print(vocab_help[jj])
            found_corr =  vocab_help[jj]
    return found_corr

def getset(l):
    if  type(l) == list:
        return set(l)
    elif type(l) ==set:
        return l
    elif (type(l) == numpy.ndarray) or (type(l) == numpy.array):
        return set(l.flatten())
    else:
        return {l}


         
def check_basic_consistency(net_predicted_help, preds_h_n,state_now,k_location):
    pred_h_n_new = []
    poss_dec =[]
    consistent = []
    inc = 0
    check = 0
    
    " If here net_predicted_help =/=[] "
    "we simply check if:"
    "1. is in the prior"
    "2. if there are objects in the state this are coherent"
    "3. If there is a known location is consistent"
    "WE finally decided that if net choice is clearly inconsistent with prior"
    if not(is_empty(state_now)):
        print('inside1')
        check = 1
        list_not_act1 = consist_obj[state_now]
        consistent = list(set.difference(getset(np.arange(0,len(vocab_help))),list_not_act1))
        preds_h_n_new = list(set.intersection(getset(preds_h_n),getset(consistent)))
        print('preds..',preds_h_n_new)
        if voc_help_to_num[net_predicted_help] in list_not_act1:
            net_consist = list(set.intersection(getset(net_predicted_help_num),getset(consistent)))
            print('inside 1.1')
            inc =-1
    elif not(is_empty(k_location)):
        check =1
        list_not_act2 = consist_loc[k_location]
        consistent = list(set.difference(getset(np.arange(0,len(vocab_help))),list_not_act2))
        preds_h_n_new = list(set.intersection(getset(preds_h_n),getset(consistent)))
        if voc_help_to_num[net_predicted_help] in list_not_act2:
            net_consist = list(set.intersection(getset(net_predicted_help_num),getset(consistent)))
            inc =-1
    if inc == 0 and check == 0:
         "no information from objects and locations"
         preds_h_n_new = preds_h_n

    return inc, preds_h_n_new, consistent
  
##
            

# =============================================================================
# end utilities
# =============================================================================


# =============================================================================
# #               OUTPUT
# =============================================================================

def build_vals(poss_dec,preds_h_n,prior_help_w,num_h):
    vals = []
    vals.append(poss_dec)
    if poss_dec == 'nah':
       vals.append('nah')
    else:
        if poss_dec in prior_help_w and len(prior_help_w)>1:
            vals2 = list(set.difference(getset(prior_help_w),getset(poss_dec)))
            if type(vals2)==list:
               vals2 = vals2[0]
            vals.append(vals2)
        else:
           vv = np.argmax(help_trans[num_h])
           vals2 = vocab_help[int(vv)]
           vals.append(vals2)
    
    return vals
# =============================================================================
#  initial state
# =============================================================================
" Initial step predict between two possibilities "

#
"to consider help_pred"
def get_commons(hh, data_help,first_h,sec_h):
    hypoth_help = hh.split(' ')
    counts_H, pp_H = compute_argmax_max(data_help,1)
    tt = counts_H[:,0]
    pred_help = [id_to_word[x] for x in tt if x!=0]       
    common = list(getset(hypoth_help).intersection(getset(pred_help)))
    if first_h == 1:
        hypoth_help2 = vocab_help[2]
    elif sec_h==1:
        hypoth_help2 = vocab_help[7]
    if common !=0:
       "1 if you are unsure and want to trust prediction"
       return  hh, hypoth_help2, common, 1     
    else:
       "0 if you do not care and trust prior"
       return  hh, hypoth_help2,  0   

def init_new(location):
    ## Modified ##
    tt0 = location=='on_ladder'
    first_h = 0
    sec_h = 0
    hh = []
    if not tt0:
           hh = vocab_help[7]   
           first_h = 1
           aa = id_to_word[9]
    else:
           hh = vocab_help[2]
           sec_h =1
           aa = id_to_word[15]
    if not is_empty(hh):
           hypoth_help = hh
           if first_h == 1:
              hypoth_help2 = vocab_help[2]
           elif sec_h==1:
              hypoth_help2 = vocab_help[7]
    else:
        hypoth_help = vocab_help[7]
        hypoth_help2 = vocab_help[2]
    return hypoth_help, hypoth_help2

" This work perfectly on video within 2 seconds, gives the correct result on all videos"    
def init(data_act,data_help,consider_help_pred):
    counts_A, pp_A = compute_argmax_max(data_act,1)
    tt10 = np.where(counts_A[:,0]==9)[0]
    tt11 = np.where(counts_A[:,0]==5)[0]
    tt20 = np.where(counts_A[:,0]==11)[0]
    tt21 = np.where(counts_A[:,0]==3)[0]
    first_h = 0
    sec_h = 0
    hh = []
    if not is_empty(tt10):
        q = tt10.shape[0]
        if pp_A[q]>0.01:
           hh = vocab_help[7]   
           first_h = 1
           aa = id_to_word[9]
    elif not is_empty(tt11):
        q = tt11.shape[0]
        if pp_A[q]>0.01:
           hh = vocab_help[7] 
           first_h =1
           aa = id_to_word[5]
    elif not is_empty(tt20):
        q = tt20.shape[0]
        if pp_A[q]>0.01:
           hh = vocab_help[2]  
           sec_h = 1
           aa = id_to_word[11]
    elif not is_empty(tt21):
        q = tt21.shape[0]
        if pp_A[q]>0.01:
           hh = vocab_help[2]
           sec_h = 1
           aa = id_to_word[3]
    if not is_empty(hh):
        if consider_help_pred:
           hypoth_help, hypoth_help2, common, cx = get_commons(hh, data_help,first_h,second_h)
        else: 
           hypoth_help = hh
           if first_h == 1:
              hypoth_help2 = vocab_help[2]
           elif sec_h==1:
              hypoth_help2 = vocab_help[7]
    else:
        hypoth_help = vocab_help[7]
        hypoth_help2 = vocab_help[2]
        cx = 1
    return hypoth_help, hypoth_help2
# =============================================================================
# ### verifying networ prediction of help
# =============================================================================
def verify_truth_of(net_predicted_help,preds_help_w,state_now,last_help):
      "hyp 1: net_prediction is in one of the transition prediction"
      unpacked0 = last_help[-1].split()
      if len(last_help)>1:
        unpacked1 = last_help[-2].split()
      if net_predicted_help in preds_help_w:
         "just check the tool"
         unpacked = net_predicted_help.split()
         common_tool = list(set.intersection(getset(state_now),getset(unpacked)))
         if 'get_from_technician' in unpacked and not(is_empty(common_tool)):  
             return True
         elif 'give_to_technician' in unpacked and common_tool==[]:  
             return True
         elif ('give_to_technician' not in unpacked) and ('get_from_technician' not in unpacked):
             return True
         elif (('give_to_technician' in unpacked1) or ('give_to_technician' in unpacked0)) and \
               ('get_from_technician'  in unpacked) and not(is_empty(list(set.intersection(getset(unpacked1),getset(unpacked)))))\
               or  not(is_empty(list(set.intersection(getset(unpacked0),getset(unpacked))))):
             return True
      else:
         return False 
    

# =============================================================================
# Managing difficult cases
#  Successive steps   
# =============================================================================
" successive steps look also on what it sees on table and look history"          
#get_objs_not_on_table(obj_state,past) 
### Modified
def get_objs_not_on_table(obj_state,past, table):
    nn =len(obj_state)
    table_flat =  list(itertools.chain(*obj_state))
    table_occ = {value: len(list(freq)) for value, freq in groupby(sorted(table_flat))}
    not_on_table = []
    freq = np.zeros(len(table))
    for ii in range(len(table)):
        xx = table[ii]
        value = table_occ.get(xx, "empty")
        if value == 'empty':
           not_on_table.append(xx)
        else:
           freq[ii] = table_occ.get(xx)
    if is_empty(not_on_table):   
        prob = freq/sum(freq)
        dic_table =dict(zip(table,prob))
        holded_tool ={k:v for k, v in dic_table.items() if v < 0.13}
        not_on_table  = list(holded_tool.keys())
        if not(is_empty(not_on_table)):
            not_on_table = not_on_table[0]
    return not_on_table

### modified  /changed
def get_where_is(where_is, where, past):
    where_appo = where.copy()
    where_appo.append('is_empty')
    where_is_bis =where_is.copy()
    for kk in range(past):
        if is_empty(where_is_bis[kk]):
            where_is_bis[kk] = {'is_empty':1.0}
    where_is_flat =  list(itertools.chain(*where_is_bis))
    where_occ = {value: len(list(freq)) for value, freq in groupby(sorted(where_is_flat))}
    
    for ii in range(len(where_appo)):
        xx = where_appo[ii]
        value = where_occ.get(xx, "empty")
        if value == 'empty':
            where_occ[xx]=0
            
    stat = max(where_occ.items(), key=operator.itemgetter(1))[0]
    if stat == 'is_empty':
        true_location = 'unknown'
    else: true_location = stat
    return true_location

"actions transtions "
def look_trans_act_help(data_act,last_help):
    " look for observed actions and their transitions to help" 
    next_help =[] 
    counts_A, pp_A = compute_argmax_max(data_act,0)
    curr_activ_w = get_names(counts_A[:,0],id_to_word)
    f_look_a, idx = look_forward(curr_activ_w,forward_actions)
      
#if
    if not(is_empty(f_look_a)):     #observed activities have been found in trans_activ-help
        next_help = forward_help_comp[idx]
    return next_help

#def check_tool()

def poss_decision_hold_tool(preds_h_n, data_act,last_help):
        poss_decision = []
        uu = [1,3,5]
        item_s = list(set.intersection(getset(preds_h_n),getset(uu)))
        if not(is_empty(item_s)):   
            "evidence is ok"
            item_s = item_s[0]
            if state_now in vocab_help[item_s]:
                state_now_val = 1   ## the table detection must be correct
                poss_decision = vocab_help[item_s]
                print('next_help prior = ', vocab_help[item_s])
##else                    
        else:  ## Error on table detection
             "evidence about table is not correct: look for the actions to check for transitions"
             "Look forward from current actions"
             "should do this cumulating with past actions and taking just the last"
             poss_decision = look_trans_act_help(data_act,last_help) 
        return poss_decision
    
    
def check_alternatives1(net_predicted_help,last_help,prior_help_w):
        if set.issubset(getset(net_predicted_help),getset(last_help)):
            poss_dec = 'nah'
            print('if 1')
        else:
            "check conditions on give and get"
            print('else of if 1')
            bool_val = verify_truth_of(net_predicted_help,prior_help_w,state_now,last_help)
            if bool_val == True:
                if not(set.issubset(getset(net_predicted_help),getset(last_help))):
                   poss_dec = net_predicted_help
                   print('if bool val')
            else:
                "here we are in truble and we  need to check a good amount of things because"
                "the net predicted help contrast with"
                "the prior transitions of the help"
                poss_dec =[]
        return poss_dec
#
def check_alternatives2(state_now,pred_h_n,data_act,last_help):
        "if we are here pss_dec =[]"
        
        if not(is_empty(state_now)):
                    print('state now !=[] ')
                    "According to evidence technician is holding a tool, check if it is correct"
                    "SI SCOMPONE!!!"
                    next_help_e = poss_decision_hold_tool(preds_h_n, data_act, last_help)
                    if set.issubset(getset(next_help_e),getset(last_help)):
                        poss_dec = 'nah'
                        print('dec = curr')
                    else:
                        poss_dec = next_help_e
                        print('next_help_e', next_help_e)
                    " look into transitions actions -help"       
        elif is_empty(state_now):
                     print('state now =[]')
                     next_help_e = look_trans_act_help(data_act,last_help)
                     if not(set.issubset(getset(next_help_e),getset(last_help))):
                         print('next =curr')
                         poss_dec = 'nah'
                     else:
                        poss_dec = next_help_e
                        print(poss_dec, 'last else next_help_e', next_help_e)
        return poss_dec

           
def check_alternatives3(state_now,k_location,last_help,prior_help_w, current_help):
    print('alternative 3')
    poss_help = []
    if k_location != 'unknown':############# changed
           print("k_location know", k_location)
           loc = consist_loc[k_location]
           vals_loc = list(set.difference(getset(np.arange(0,8)),getset(loc)))
           if len(vals_loc) > 1:
               "only one of them"
               if is_empty(state_now):
                       poss_help = vocab_help[vals_loc[0]]
                       poss_help2 = vocab_help[vals_loc[1]]
               else:
                       poss_help = vocab_help[vals_loc[1]]
                       poss_help2 = vocab_help[vals_loc[0]]
           if len(vals_loc) == 1:   
              if vocab_help[7] == current_help and k_location == 'at_guard_support':
                poss_help =  'nah'
                poss_help2 = 'nah'
              elif vocab_help[7] != current_help and k_location == 'at_guard_support':
                poss_help = vocab_help[vals_loc[0]]
                poss_help2 = vocab_help[0]
    if not(is_empty(poss_help)):
        if poss_help == 'nah' and poss_help2 == 'nah':
            poss_dec = poss_help
            poss_dec2 = poss_help
        if set.issubset(getset(poss_help),getset(last_help)):
           poss_dec ='nah'
           poss_dec2 = 'nah'
        else: 
            poss_dec = poss_help
            poss_dec2 = poss_help2
    if k_location == 'unknown' and is_empty(poss_help):
        if state_now == 'torch':
           if vocab_help[0] in last_help:
               poss_dec = vocab_help[5]
               poss_dec2 = vocab_help[4]
           else: poss_help =[]
        if state_now == 'cloth':
           if vocab_help[6] in last_help:
               poss_dec = vocab_help[1]
               poss_dec2 = vocab_help[5]
           else: poss_help =[]
    if poss_help == []:
        poss_dec = vocab_help[0]
        poss_dec2 = vocab_help[4]
    return poss_dec, poss_dec2

def choose_poss_help(cc,last_help,state_now):
    poss_help = []
    poss_dec =[]
    print(cc)
    for ww in range(len(cc)):
           poss_help.append(vocab_help[cc[ww]])
    if len(poss_help)==1:
        if set.issubset(getset(poss_help),getset(last_help)):
           poss_dec ='nah'
        else:
            if type(poss_help) == list:
               poss_dec = poss_help[0]
            else: poss_dec=poss_help
    else:
        "then on ladder and give-get"
        if not(is_empty(state_now)):
            if type(state_now) == list:
                state_now = state_now[0]
            if word_to_id[state_now] == 15:
                last_help_flat =  list(itertools.chain(*last_help_splitted))
                if state_now in last_help_flat and give_t in last_help_flat and get_t in last_help_flat:
                    poss_dec = 'nah'
                elif state_now in last_help_flat and give_t in last_help_flat and not(get_t in last_help_flat):
                    poss_dec = vocab_help[3]
        else: 
            poss_h = vocab_help[cc[0]]
            done = set.issubset(getset(pos_h), getset(last_help))
            if done:
                poss_dec = 'nah'
            else:
                poss_dec = poss_h

    print('POSS_DEC', poss_dec)
    return poss_dec

##############################################################################
# =============================================================================
#                     MAIN: compute_help
# =============================================================================
                    

def compute_help(help_answer_stack,now_softmax,help_softmax,obj_state,where_is, timeT=0):
    "initial language"
    table = ['cutter', 'spray_bottle', 'cloth', 'torch']
    where = ['on_ladder', 'at_guard_support']
    tools = ['cloth','cutter','spray_bottle','torch']
    
    "data"
    keys_pred = [1,2]
    past = 5
    data_act = []
    data_help = [] 
    help_pred_stack =[]
    
    for kk in range(past):
        
       obs_help = normalise_v(help_softmax[kk])
       obs_now = normalise_v(now_softmax[kk])
       data_act.append(obs_now)
       data_help.append(obs_help)
       
    # show_net_argmax(data_act,data_help)
    at_location = get_where_is(where_is,where, past)
    "at the beginning the help answer stack is empty"
    if is_empty(help_answer_stack):
        "if the recognition is = the hypothesis"
        hyp_help1, hyp_help2  = init_new(at_location)
        print('hyp_help1,hyp_help2', hyp_help1,hyp_help2)
        if not(is_empty(hyp_help1)):
            values =[hyp_help1,hyp_help2]
            help_pred_stack= dict(zip(keys_pred, values)) 
        elif is_empty(hyp_help1):
            print('waiting for more evidence')
            values =['nah','nah']
            help_pred_stack= dict(zip(keys_pred, values)) 
            
        return(help_pred_stack)
    else:  #first step predicted go on
 
# =============================================================================
#             1 Get next help from transitions: these may be more than one
#              2. look forward from current actions looking into a2h
#              1.1 if predicted help in help_pred_stack[-1]: do nothing"
#              1.2 if predicted help != help_pred_stack[-1]: compute next help
#             2. Compute next help from help_pred_stack[-1]
#             2.1 Compute actions expected from dict_help
# =============================================================================
        "three cases for help (poss) decision: "
        "i. the good help"
        "ii. nah (not an help), which amounts to waiting for more information,"
        "iii. search tree_poss, which amounts to check for some more evidence from the data"
##      
        "set some parameters"
        poss_dec =[]
        poss_dec1 = []
        poss_dec2 = []
        net_predicted_help =[]
        help_pred_stack = []
        
        last_help = help_answer_stack.copy()
        last_help_splitted = [last_help[i].split() for i in range(0,len(last_help))]
        give_t = 'give_to_technician'
        get_t = 'get_from_technician_and_put_on_the_table'
        "step 1. Try next help from net predicted help,  table state, previous help and transition"
        curr_help = help_answer_stack[-1]
        num_h = voc_help_to_num[curr_help]
        
        state_now = get_objs_not_on_table(obj_state, past, table)
        if 'cutter' in state_now:
            state_now = []
        at_location = get_where_is(where_is, where, past)
        print(at_location)
        counts2,pp = compute_argmax_max(data_help,0)
        uu =(-counts2[:,1]).argsort()[:3] 
        net_predicted_help_L = [id_to_word[x] for x in counts2[uu,0]]
        net_predicted_help = recompose_list_into_help(net_predicted_help_L)
        
                
##
            
        "1.1 Compute transition from current verified (from dialog) help"
        "if predicted help is one of the predicted from transition"
        "and if get is mentioned and the object is not-in state"
        "then exit with help_net_predicted"
        
        preds_h_n = np.where(help_trans[num_h,:]>0)[0]
        best_vals =(-help_trans[num_h,:]).argsort()[:2] 
        preds_vals = [help_trans[num_h,int(x)] for x in preds_h_n]
        prior_help_w = get_names(preds_h_n,vocab_help)
        
        
        done_trans = list(set.difference(getset(prior_help_w),getset(last_help)))
        if len(done_trans) == 1 and type(done_trans)==list:
            poss_dec1 = done_trans[0]
            n_b = voc_help_to_num[done_trans[0]]
            sec = list(set.difference(getset(best_vals),getset(n_b)))
            poss_dec2 = vocab_help[sec[0]]
        elif len(done_trans) == 1 and type(done_trans)!=list:
            poss_dec1 = done_trans
            n_b = voc_help_to_num[done_trans[0]]
            sec = list(set.difference(getset(best_vals),getset(n_b)))
            poss_dec2 = vocab_help[sec[0]]
#        elif set.issubset(getset(net_predicted_help),getset(last_help)):
#           done = True
#           poss_dec = 'nah'
        ### Modified
        
            
        if is_empty(poss_dec1):    
           poss_dec1,poss_dec2 = check_alternatives3(state_now,at_location,last_help,prior_help_w,curr_help)
           if is_empty(poss_dec1):
               poss_dec1 = 'nah'
               poss_dec2 ='nah'
               print('empty poss dec')
             
##  changed
    print('In the end: first choice = ', poss_dec1, 'second_choice =', poss_dec2) 
#    values = build_vals(poss_dec,preds_h_n,prior_help_w,num_h)  
#    idx =[i for i,x in enumerate(values0) if type(x) == list] 
#    values[idx[0]] = values[idx[0]][0]
    values = list([poss_dec1,poss_dec2])
    
    help_pred_stack= dict(zip(keys_pred, list(values)))   
    return help_pred_stack