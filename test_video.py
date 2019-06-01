import activity_network
import pprint
import pickle
import config
import cv2
import numpy as np
import os
from tqdm import tqdm
import prep_dataset_manager
import prep_dataset_manager as prep_dataset_man

pp = pprint.PrettyPrinter(indent=4)

def save(obj, name):
    with open('dataset/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load(name):
    with open('dataset/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def extract_preprocessed_one_input(video_path, segment, prep_dataset):
        one_input = np.zeros(shape=(config.frames_per_step, config.out_H, config.out_W, 7), dtype=float)
        extracted_frames = {}
        frame_list = []
        try:
            linspace_frame = np.linspace(segment[0], segment[1], num=config.frames_per_step)
            z = 0
            for frame in linspace_frame:
                try:
                    one_input[z, :, :, :] = prep_dataset.get_matrix(video_path, frame)
                    z += 1
                except Exception as e:
                    print(e)
                    pass
        except Exception as e:
            print(e)
            pass
        frame_list = extracted_frames.keys()
        return one_input, frame_list

def test():
        prep_dataset = prep_dataset_man.prep_dataset()
        net = activity_network.activity_network()
        test_collection = load('train_collection')
        id_to_word = load('id_to_word')
        id_to_label = load('id_to_label')
        ordered_collection = load('ordered_collection')
        path_collection = []
        output_collection = {}
        for key1 in test_collection.keys():
                for key2 in test_collection[key1].keys():
                        for key3 in test_collection[key1][key2].keys():
                                for entry in test_collection[key1][key2][key3]:
                                        video_name = entry['path']
                                        if video_name not in path_collection:
                                               path_collection.append(video_name)

        pbar_video = tqdm(total=len(path_collection), leave=False, desc='Videos')

        for path in path_collection:
                net.hidden_states_collection = {}
                output_collection[path] = {}
                video = cv2.VideoCapture(path)
                video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
                framecount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(video.get(cv2.CAP_PROP_FPS))
                seconds = int(framecount/fps)
                second_collection = []
                pbar_second = tqdm(total=seconds, leave=False, desc='seconds')
                width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                fourcc = cv2.VideoWriter_fourcc(*'MPEG')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('dataset/results/'+ path.split('/')[-1],fourcc, fps, (width,height))
        
                now_word = ''
                next_word = ''
                action = ''
                obj = ''
                place = ''
                correct_now = 0
                correct_c3d = 0
                correct_next = 0
                correct_help = 0
                for s in range(seconds):
                        
                        linspace_frame = np.linspace(s*fps+1, (s+1)*fps+1, num=config.frames_per_step)
                        linspace_frame = [int(x) for x in linspace_frame]
                        if (linspace_frame[-1] == (framecount-1)):
                                linspace_frame[-1] -= 1
                        if (linspace_frame[-1] == (framecount-2)):
                                linspace_frame[-1] -= 2

                        extracted_frames = {}
                        z = 0
                        frames_collection = []
                        segment = [int(linspace_frame[0]), int(linspace_frame[-1])+1]
                        one_input, frame_list = extract_preprocessed_one_input(path, segment, prep_dataset)
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
                        
                        vers2_matrix = net.compound_second_frames(frames_collection)
                        second_matrix = net.compound_second_frames(one_input)
                        second_collection.append(second_matrix)
                        print(one_input.shape)
                        print(second_matrix.shape)
                        print(vers2_matrix.shape)
                        print(numpy.array_equal(one_input, second_matrix))
                        print(numpy.array_equal(one_input, vers2_matrix))
                        print(numpy.array_equal(vers2_matrix, second_matrix))
                        if s >= 3:
                                input_sec = second_collection[-4:]
                                now_softmax, next_softmax, help_softmax, c3d_softmax = net.compute_activity_given_seconds_matrix(input_sec, s)
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
                                now_word = id_to_word[np.argmax(now_word, axis=0)]
                                c3d_word = id_to_word[np.argmax(c3d_word, axis=0)]
                                next_word = id_to_word[np.argmax(next_word, axis=0)]
                                action = id_to_word[np.argmax(action, axis=0)]
                                obj = id_to_word[np.argmax(obj, axis=0)]
                                place = id_to_word[np.argmax(place, axis=0)]
                                help_word = action + ' ' + obj + ' ' + place
                                now_target = id_to_label[ordered_collection[path][s-config.seq_len+1]['now_label']]
                                next_label = id_to_label[ordered_collection[path][s]['next_label']]
                                help_label = id_to_label[ordered_collection[path][s]['help']]
                                if now_word == now_target:
                                        correct_now += 1
                                if c3d_word == now_target:
                                        correct_c3d += 1
                                if next_label == next_word:
                                        correct_next += 1
                                if help_label == help_word:
                                        correct_help += 1

                                print('\n')
                                print(' ', now_word, c3d_word, next_word, action, obj, place)
                                print(' ', now_target, now_target, next_label, help_label)
                                print(' ', float(correct_now)/(s+1), float(correct_c3d)/(s+1), float(correct_next)/(s+1), float(correct_help)/(s+1))
                        
                        for frame in range(s*fps+1, (s+1)*fps+1):
                                video.set(1, frame)
                                ret, im = video.read()
                                text = 'Now: ' + now_word + '. Next: ' + next_word 
                                cv2.putText(im, text ,(10,10),1,1,(255,255,255))
                                text = 'Help: ' + action + ' ' + obj + ' ' + place 
                                cv2.putText(im, text ,(20,20),1,1,(255,255,255))
                                out.write(im)
                        
                        # pbar_second.update(1)
                pbar_second.refresh()
                pbar_second.clear()
                pbar_second.close()
                save(output_collection, 'output_collection')
                pbar_video.update(1)
                out.release()

        pbar_video.refresh()
        pbar_video.clear()
        pbar_video.close()
test()