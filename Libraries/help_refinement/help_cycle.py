# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 13:40:29 2019

@author: fiora
"""

import numpy as np 
import numpy.matlib
import pickle

import json
import re

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


# =============================================================================
#  load parameters -->see
# =============================================================================
def find_in_L(item, listN):
    indexes = [i for i,x in enumerate(listN) if x == item] 
    return indexes

def join_vals(item):
       return (' '.join(item))
       

def is_empty(xx):
    return len(xx)==0

" Just computes a reasonable argmax --> next to be added after first exp. "
def compute_best(data_act,data_help,true_last_help):
    aa =[]
    aa_name =[]
    ppa =[]
    mat_act_obs = np.array([np.array(xi) for xi in data_act]) 
    w1,w2,w3 =mat_act_obs.shape
    act_ob_np = np.reshape(mat_act_obs,(w1*w2,w3)) 
    for z in range(len(act_ob_np)):
                  ww =(-act_ob_np[z,:]).argsort()[:2] 
                  probs = act_ob_np[z,ww]
                  if ((np.abs(np.diff(probs)))>0.6)[0]:
                     tta = np.argmax(act_ob_np[z,:])
                     if tta>0:
                         tt_name = id_to_word[tta]
#                         print("Action: ",tt_name,"z= ",z)
                         pp = np.max(act_ob_np[z,:])
                         aa.append(tta)
                         aa_name.append(tt_name)  
                         ppa.append(pp) 
                          
    unique_a, counts_a = np.unique(aa, return_counts=True)   
    "build the hypothesis"          
    
        
        
    "help action"    
    hh_a_name =[]
    hh_a =[]
    pph_a =[] 
               
    mat_help_obs = np.array([np.array(xi) for xi in data_help]) 
    u1,u2,u3 = mat_help_obs.shape
    help_ob_np = np.reshape(mat_help_obs,(u1*u2,u3)) 
    for zz in range(0,len(help_ob_np),3):
                  uu =(-help_ob_np[zz,:]).argsort()[:2] 
                  probs_h = help_ob_np[zz,uu]
                  if ((np.abs(np.diff(probs_h)))>0.4)[0]:
                     helpX =[]
                     pX =[]
                     helpX_name = []
                     for z in range(3):
                        tth = np.argmax(help_ob_np[z,:])
                        tth_name = id_to_word[tth]
#                        print("Help: ",tth_name)
                        pp = np.max(help_ob_np[z,:])
                        helpX.append(tth)
                        helpX_name.append(tth_name)  
                        pX.append(pp) 
                     hh_a_name.append(helpX_name)
                     hh_a.append(helpX)
                     pph_a.append(pX)
    "eliminate repetitions"
    result = []
    if not(is_empty(hh_a_name)):
       ll_h = [list(helps) for helps in set(tuple(data) for data in hh_a_name)]
       result_appo = [(' '.join(h)) for h in ll_h]
       
       if len(result_appo)>1:
           probsH = np.array([np.array(xi) for xi in pph_a]) 
           mu = (1/3)*np.sum(probsH,axis=1)
           "first choose the one new, not in true_last_help"
           "if true_last_help is empty choose the one with highest prob"
           if not(is_empty(true_last_help)):
               last_help = true_last_help[-1]
               for jj in range(len(result_appo)):
                   tt = find_in_L(result_appo[jj],vocab_help)
                   val_trans = help_trans(last_help,tt[0])
                   if val_trans>0:
                      result.append(result_appo[jj])
           if is_empty(result):
               "no transition from previous, or at the beginning"
               "outpt the two with max prob"
               result_appo_2 = [(' '.join(h)) for h in hh_a_name]
               dict_choice = sorted(dict(zip(mu, result_appo_2)) )
               chosen_mus =(-mu).argsort()[:len(dict_choice)]
##
               for jj in range(len(chosen_mus)):
                   xx = dict_choice[chosen_mus[jj]]
                   if not(xx in result) and len(result)<2:
                       result.append(xx)
               
       else:  
          "case is unique"
          result = result_appo[0]

#    print('Result: ', result)
    return result

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


def check_help(help_pred_stack,help_answer_stack,now_softmax,help_softmax, horizon = 5):
    
    past = horizon
    data_act = []
    data_help = []

    for kk in range(past):
        obs_help = normalise_v(help_softmax[kk])
        obs_now = normalise_v(now_softmax[kk])
        data_act.append(obs_now)
        data_help.append(obs_help)
   
    result = compute_best(data_act,data_help,help_answer_stack)
#    print(result)
    if is_empty(result):
        result.append('nah')
    else:
        help_pred_stack.append(result)
    return help_pred_stack
       
   
    
    
    