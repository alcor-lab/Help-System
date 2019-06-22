import activity_network
import pprint
import pickle
import config
import cv2
import numpy as np
import os
from tqdm import tqdm

pp = pprint.PrettyPrinter(indent=4)

def save(obj, name):
    with open('dataset/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load(name):
    with open('dataset/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def test():
        net = activity_network.activity_network()
        id_to_word = load('id_to_word')
        word_to_id = load('word_to_id')
        id_to_label = load('id_to_label')
        makis_collection = load('demo_output_data')
        makis_start_sec = min(makis_collection['now_sm'].keys())
        path_collection = []
        output_collection = {}
        for root, dirs, files in os.walk(config.demo_path_video):
                for fl in files:
                        path = root + fl
                        path_collection.append(path)

        pbar_video = tqdm(total=len(path_collection), leave=False, desc='Videos')

        RED_COLOR = (0,0,255)
        GREEN_COLOR = (0,255,0)
        BLUE_COLOR = (255,0,0)
        YELLOW_COLOR = (255,255,0)
        print(path_collection)
        for path in path_collection:
                net.hidden_states_collection = {}
                output_collection[path] = {}
                video = cv2.VideoCapture(path)
                video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
                framecount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(video.get(cv2.CAP_PROP_FPS))
                seconds = int(framecount/fps)
                second_collection = []
                obj_list= []
                vers2_collection = []
                pbar_second = tqdm(total=seconds, leave=False, desc='seconds')
                width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out_width = width*2
                out_height = height*2
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                fourcc = cv2.VideoWriter_fourcc(*'MPEG')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('dataset/results/'+ path.split('/')[-1],fourcc, fps, (out_width,out_height))
        
                now_word = ''
                c3d_word = ''
                next_word = ''
                action = ''
                help_word = ''
                obj = ''
                place = ''
                action_prob = 0
                obj_prob    = 0     
                place_prob  = 0     
                now_prob    = 0     
                next_prob   = 0     
                c3d_prob   = 0      
                correct_now = 0
                correct_c3d = 0
                correct_next = 0
                correct_help = 0
                for s in range(seconds-1):
                        
                        linspace_frame = np.linspace(s*fps+1, (s+1)*fps+1, num=config.frames_per_step)
                        linspace_frame = [int(x) for x in linspace_frame]
                        if (linspace_frame[-1] == (framecount-1)):
                                linspace_frame[-1] -= 1
                        if (linspace_frame[-1] == (framecount-2)):
                                linspace_frame[-1] -= 2

                        extracted_frames = {}
                        # z = 0
                        frames_collection = []
                        # segment = [int(linspace_frame[0]), int(linspace_frame[-1])+1]
                        for frame in range(int(linspace_frame[0]), int(linspace_frame[-1])+1):
                                video.set(1, frame)
                                ret, im = video.read()
                                extracted_frames[frame] = im
                                frame = int(frame)
                                frame_prev = frame - 1
                                if frame in linspace_frame:
                                        if frame_prev in extracted_frames:
                                                im_prev = extracted_frames[frame_prev]
                                        else:
                                                video.set(1, frame_prev)
                                                ret, im_prev = video.read()
                                                extracted_frames[frame_prev] = im_prev

                                        flow = net.compute_optical_flow(im, im_prev)
                                        pafMat, heatMat = net.compute_pose(im)
                                        frame_processed = net.compound_channel(im, flow, heatMat, pafMat)
                                        frames_collection.append(frame_processed)
                        
                        second_matrix = net.compound_second_frames(frames_collection)
                        second_collection.append(second_matrix)
                        now_live_softmax = makis_collection['now_sm'][makis_start_sec + s][-1,:]
                        now_live_max = np.argmax(now_live_softmax, axis=0)
                        now_target_prob = now_live_softmax[now_live_max]
                        now_target = id_to_word[now_live_max]
                        # next_label = id_to_label[ordered_collection[path][s]['next_label']]
                        # help_label = id_to_label[ordered_collection[path][s]['help']]
                        sec_id_obj = makis_collection['obj_label'][makis_start_sec + s][0]
                        for obj_id in sec_id_obj:
                                print(sec_id_obj)
                                print(id_to_word[obj_id])
                        # sec_id_obj = {}
                        # for obj in sec_obj.keys():
                        #         position = word_to_id[obj]
                        #         value = sec_obj[obj]
                        #         sec_id_obj[position] = value

                        obj_list.append(sec_id_obj)
                        if s >= 3:
                                input_sec = second_collection[-4:]
                                input_obj = obj_list[-4:]
                                now_softmax, next_softmax, help_softmax, c3d_softmax = net.compute_activity_given_seconds_matrix(input_sec, s, input_obj)

                                output_collection[path][s] = {}
                                output_collection[path][s]['now_softmax'] = now_softmax
                                output_collection[path][s]['next_softmax'] = next_softmax
                                output_collection[path][s]['help_softmax'] = help_softmax
                                output_collection[path][s]['c3d_softmax'] = c3d_softmax

                                now_word = now_softmax[-1,:]
                                c3d_word = c3d_softmax[-1,:]
                                next_word =next_softmax[-1,:]
                                action = help_softmax[0,:]
                                obj = help_softmax[1,:]
                                place = help_softmax[2,:]

                                now_word_max = np.argmax(now_word, axis=0)
                                c3d_word_max = np.argmax(c3d_word, axis=0)
                                next_word_max = np.argmax(next_word, axis=0)
                                action_max = np.argmax(action, axis=0)
                                obj_max = np.argmax(obj, axis=0)
                                place_max = np.argmax(place, axis=0)

                                now_prob = now_word[now_word_max]
                                c3d_prob = c3d_word[c3d_word_max]
                                next_prob = next_word[next_word_max]
                                action_prob = action[action_max]
                                obj_prob = obj[obj_max]
                                place_prob = place[place_max]

                                now_word = id_to_word[now_word_max]
                                c3d_word = id_to_word[c3d_word_max]
                                next_word = id_to_word[next_word_max]
                                action = id_to_word[action_max]
                                obj = id_to_word[obj_max]
                                place = id_to_word[place_max]

                                help_word = action + ' ' + obj + ' ' + place

                                
                                if now_word == now_target:
                                        correct_now += 1
                                # if c3d_word == now_target:
                                #         correct_c3d += 1
                                # if next_label == next_word:
                                #         correct_next += 1
                                # if help_label == help_word:
                                #         correct_help += 1

                                # print('\n')
                                # print(' ', now_target, now_target, next_label, help_label)
                                # print(' ', now_word, c3d_word, next_word, action, obj, place)
                                # print('prep ', float(correct_now)/(s+1), float(correct_c3d)/(s+1), float(correct_next)/(s+1), float(correct_help)/(s+1))
                        
                        for frame in range(s*fps+1, (s+1)*fps+1):
                                video.set(1, frame)
                                ret, im = video.read()
                                im = cv2.resize(im, dsize=(out_width, out_height), interpolation=cv2.INTER_CUBIC)

                                thickness = 3
                                font_scale = 3

                                # color = BLUE_COLOR
                                # text = 'Help: ' + action + ' ' + str(action_prob) 
                                # cv2.putText(im, text ,(10,20),1,font_scale,color,thickness, bottomLeftOrigin=False)
                                # text = 'Help: '+ obj + ' ' + str(obj_prob)
                                # cv2.putText(im, text ,(10,50),1,font_scale,color,thickness, bottomLeftOrigin=False)
                                # text = 'Help: ' + place + ' ' + str(place_prob) 
                                # cv2.putText(im, text ,(10,80),1,font_scale,color,thickness, bottomLeftOrigin=False)
                                
                                if now_word == now_target:
                                        color = GREEN_COLOR
                                else:
                                        color = RED_COLOR
                                text = 'Now: ' + now_word + ' ' + str(now_prob) 
                                cv2.putText(im, text ,(10,110),1,font_scale,color,thickness, bottomLeftOrigin=False)

                                # color = BLUE_COLOR
                                # text = 'Next: ' + next_word + ' ' + str(next_prob)
                                # cv2.putText(im, text ,(10,140),1,font_scale,color,thickness, bottomLeftOrigin=False)

                                color = BLUE_COLOR
                                text = 'Live Now: ' + now_target + ' ' + str(now_target_prob) 
                                cv2.putText(im, text ,(10,170),1,font_scale,color,thickness, bottomLeftOrigin=False)
                                color = BLUE_COLOR
                                text = 'c3d_word Now: ' + c3d_word + ' ' + str(c3d_prob) 
                                cv2.putText(im, text ,(10,200),1,font_scale,color,thickness, bottomLeftOrigin=False)
                                
                                out.write(im)
                        
                        pbar_second.update(1)
                pbar_second.refresh()
                pbar_second.clear()
                pbar_second.close()
                # save(output_collection, 'output_collection')
                pbar_video.update(1)
                out.release()

        pbar_video.refresh()
        pbar_video.clear()
        pbar_video.close()
test()