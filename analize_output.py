import activity_network
import pprint
import pickle
import config
import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')
import itertools
import matplotlib


pp = pprint.PrettyPrinter(indent=4)

def save(obj, name):
    with open('dataset/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load(name):
    with open('dataset/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def create_cm_pic(cm_in, number_of_classes, word_list, tensor_name):
        cm = cm_in / cm_in.astype(np.float).sum(axis=1)
        fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=300, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(cm, cmap='Oranges')
        
        tick_marks = np.arange(number_of_classes)

        ax.set_xlabel('Predicted', fontsize=7)
        ax.set_xticks(tick_marks)
        c = ax.set_xticklabels(word_list, fontsize=4, rotation=-90,  ha='center')
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()

        ax.set_ylabel('True Label', fontsize=7)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(word_list, fontsize=4, va ='center')
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], '.2f') if cm[i,j]>0 else '.', horizontalalignment="center", fontsize=4, verticalalignment='center', color= "black")
            fig.set_tight_layout(True)

        if fig.canvas is None:
            matplotlib.backends.backend_agg.FigureCanvasAgg(fig)

        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        cv2.imwrite('dataset/results/confusion/' + tensor_name + '.jpg', data)

def create_confusion():
        word_to_id = load('word_to_id')
        id_to_label = load('id_to_label')
        ordered_collection = load('ordered_collection')
        output_collection = load('output_collection')
        
        word_to_acc = {}
        for pred_type in ['now', 'next', 'c3d', 'help']:
                word_to_acc[pred_type] = {}
                for word in word_to_id.keys():
                        word_to_acc[pred_type][word] = 0 

        now_cm = np.zeros((len(word_to_id), len(word_to_id)), dtype=np.int32)
        c3d_cm = np.zeros((len(word_to_id), len(word_to_id)), dtype=np.int32)
        next_cm = np.zeros((len(word_to_id), len(word_to_id)), dtype=np.int32)
        help_cm = np.zeros((len(word_to_id), len(word_to_id)), dtype=np.int32)
        obj_cm = np.zeros((len(word_to_id), len(word_to_id)), dtype=np.int32)
        place_cm = np.zeros((len(word_to_id), len(word_to_id)), dtype=np.int32)

        for path in output_collection:
                for sec in output_collection[path]:
                        now_softmax = output_collection[path][sec]['now_softmax']
                        next_softmax = output_collection[path][sec]['next_softmax']
                        help_softmax = output_collection[path][sec]['help_softmax']
                        c3d_softmax = output_collection[path][sec]['c3d_softmax']
                        now_word = now_softmax[-1,:]
                        c3d_word = c3d_softmax[-1,:]
                        next_word =next_softmax[-1,:]
                        action = help_softmax[0,:]
                        obj = help_softmax[1,:]
                        place = help_softmax[2,:]
                        now_id = np.argmax(now_word, axis=0)
                        c3d_id = np.argmax(c3d_word, axis=0)
                        next_id = np.argmax(next_word, axis=0)
                        action_id = np.argmax(action, axis=0)
                        obj_id = np.argmax(obj, axis=0)
                        place_id = np.argmax(place, axis=0)

                        now_label = id_to_label[ordered_collection[path][sec]['now_label']]
                        next_label = id_to_label[ordered_collection[path][sec]['next_label']]
                        help_label = id_to_label[ordered_collection[path][sec]['help']].split(' ')

                        if help_label == 'sil':
                                help_label = 'sil sil sil'

                        action_label = id_to_label[help_label[0]]
                        obj_label = id_to_label[help_label[1]]
                        place_label = id_to_label[help_label[2]]

                        now_cm[now_id, now_label] += 1
                        c3d_cm[c3d_id, now_label] += 1
                        next_cm[next_id, next_label] += 1
                        help_cm[action_id, action_label] += 1
                        obj_cm[obj_id, obj_label] += 1
                        place_cm[place_id, place_label] += 1

        word_len = len(word_to_id)
        word_list = list(word_to_id.keys())
        create_cm_pic(now_cm, word_len, word_list, 'now_cm')
        create_cm_pic(c3d_cm, word_len, word_list, 'c3d_cm')
        create_cm_pic(next_cm, word_len, word_list, 'next_cm')
        create_cm_pic(help_cm, word_len, word_list, 'help_cm')
        create_cm_pic(obj_cm, word_len, word_list, 'obj_cm')
        create_cm_pic(place_cm, word_len, word_list, 'place_cm')

create_confusion()             