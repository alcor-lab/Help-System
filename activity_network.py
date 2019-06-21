
import logging
import logging.config
from logging.config import dictConfig
import os
logging_config = dict(
    version = 1,
    formatters = {
        'f': {'format':
              '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'}
        },
    handlers = {
        'h': {'class': 'logging.StreamHandler',
              'formatter': 'f',
              'level': logging.WARNING}
        },
    root = {
        'handlers': ['h'],
        'level': logging.WARNING,
        },
)

dictConfig(logging_config)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.WARNING)


import tensorflow as tf
from tqdm import tqdm
import cv2
import numpy as np
import multiprocessing.dummy as mt
import config
import pprint
pp = pprint.PrettyPrinter(indent=4)

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x for x in local_device_protos if x.device_type == 'GPU']
    # return local_device_protos

class activity_network:
    def __init__(self, sess=None):
        # creating a Session
        if sess is None:
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        else:
            self.sess = sess

        # load architecture in graph and weights in session and initialize

        self.graph = tf.get_default_graph()
        self.architecture = tf.train.import_meta_graph('model/activity_network_model.ckpt.meta')
        self.latest_ckp = tf.train.latest_checkpoint('model')

        self.architecture.restore(self.sess, self.latest_ckp)
        # Show progress bar to visualize datasets creation
        self.use_pbar = True
        self.hidden_states_collection = {}

        # Retrieving Pose input and outputs
        self.pose_input = self.graph.get_tensor_by_name('image:0')
        pose_out_name_1 = [n.name for n in tf.get_default_graph().as_graph_def().node if 'Stage6_L1_5_pointwise/BatchNorm/FusedBatchNorm' in n.name][0]
        self.pose_out_1 = self.graph.get_tensor_by_name(pose_out_name_1 + ":0")
        pose_out_name_2 = [n.name for n in tf.get_default_graph().as_graph_def().node if 'Stage6_L2_5_pointwise/BatchNorm/FusedBatchNorm' in n.name][0]
        self.pose_out_2 = self.graph.get_tensor_by_name(pose_out_name_2 + ":0")

        # Retrieving activity recognition network inputs and outputs
        self.input = self.graph.get_tensor_by_name("Input/Input:0")
        self.h_input = self.graph.get_tensor_by_name("Input/h_input:0")
        self.c_input = self.graph.get_tensor_by_name("Input/c_input:0")
        self.obj_input = self.graph.get_tensor_by_name("Object_Input/obj_input:0")
        self.c3d_softmax = self.graph.get_tensor_by_name("Network/Activity_Recognition_Network/c3d_classifier/Softmax:0")
        self.now_softmax = self.graph.get_tensor_by_name("Network/Activity_Recognition_Network/Now_Decoder_inference/softmax_out:0")
        self.help_softmax = self.graph.get_tensor_by_name("Network/Activity_Recognition_Network/Help_Decoder_inference/softmax_out:0")
        self.next_softmax = self.graph.get_tensor_by_name("Network/Activity_Recognition_Network/Next_classifier/softmax_out:0")
        self.c_out = self.graph.get_tensor_by_name("Network/Activity_Recognition_Network/Lstm_encoder/c_out:0")
        self.h_out = self.graph.get_tensor_by_name("Network/Activity_Recognition_Network/Lstm_encoder/h_out:0")
        self.empyt_labels = np.zeros(shape=(4, 1,config.seq_len + 1), dtype=int)

    def create_graph_log(self):
        # This function create a tensorboard log which shows the network as_graph_def
        file_writer = tf.summary.FileWriter("tensorboardLogs", tf.get_default_graph())
        file_writer.close()

    def compute_optical_flow(self, frame, frame_prev):
        #Takes in input current frame and previous frame and return optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray_prev, gray, flow=None,
                                                        pyr_scale=0.5, levels=1,
                                                        winsize=15, iterations=3,
                                                        poly_n=5, poly_sigma=1.1, flags=0)
        norm_flow = flow
        norm_flow = cv2.normalize(flow, norm_flow, 0, 255, cv2.NORM_MINMAX)
        return norm_flow

    def compute_pose(self, img):
        #Takes in input current frame and return pafMat, heatMat
        shape_img = img.shape
        if shape_img[0] != config.op_input_height:
            img = cv2.resize(img, dsize=(config.op_input_height, config.op_input_width), interpolation=cv2.INTER_CUBIC)
        pafMat, heatMat = self.sess.run([self.pose_out_1, self.pose_out_2], feed_dict={'image:0': [img]})
        heatMat, pafMat = heatMat[0], pafMat[0]
        
        heatMat = np.amax(heatMat, axis=2)
        heatMat = cv2.resize(heatMat, dsize=(config.out_H, config.out_W), interpolation=cv2.INTER_CUBIC)
        norm_heatMat = cv2.normalize(heatMat, None, 0, 255, cv2.NORM_MINMAX)

        pafMat = np.amax(pafMat, axis=2)
        pafMat = cv2.resize(pafMat, dsize=(config.out_H, config.out_W), interpolation=cv2.INTER_CUBIC)
        norm_pafMat = cv2.normalize(pafMat, None, 0, 255, cv2.NORM_MINMAX)

        return norm_pafMat, norm_heatMat

    def compound_channel(self, img, flow, heatMat, pafMat):
        # stack rgb, flow, heatMat and pafMat
        frame = np.zeros(shape=(config.out_H, config.out_W, 7), dtype=np.uint8)
        shape_img = img.shape
        shape_flow = flow.shape
        shape_heatMat = heatMat.shape
        shape_pafMat = pafMat.shape
        if shape_img[0] != config.op_input_height:
            img = cv2.resize(img, dsize=(config.out_H, config.out_W), interpolation=cv2.INTER_CUBIC)
        if shape_flow[0] != config.op_input_height:
            flow = cv2.resize(flow, dsize=(config.out_H, config.out_W), interpolation=cv2.INTER_CUBIC)
        if shape_heatMat[0] != config.op_input_height:
            heatMat = cv2.resize(heatMat, dsize=(config.out_H, config.out_W), interpolation=cv2.INTER_CUBIC)
        if shape_pafMat[0] != config.op_input_height:
            pafMat = cv2.resize(pafMat, dsize=(config.out_H, config.out_W), interpolation=cv2.INTER_CUBIC)
        frame[..., :3] = img
        frame[..., 3] = heatMat
        frame[..., 4] = pafMat
        frame[..., 5:7] = flow
        frame = frame.astype(np.uint8)
        return frame

    def generate_obj_tensor(self, dict_obj):
        obj_input = np.zeros(shape=(1, 1, config.seq_len, config.vocab_len), dtype=float)
        for indx in range(len(dict_obj)):
            obj = dict_obj[indx]
            for obj_id in obj:
                obj_input[0,0,indx,obj_id] = obj[obj_id]
        
        return obj_input

    def compound_second_frames(self, frames):
        # group frames from one second
        number_of_frames = len(frames)
        if number_of_frames != config.frames_per_step:
            raise ValueError('too many frames per second')
        second_tensor = np.zeros(shape=(config.frames_per_step, config.out_H, config.out_W, 7), dtype=np.uint8)

        i=0
        for frame in frames:
            shape_frame = frame.shape
            if shape_frame[-1] != config.input_channels:
                raise ValueError('frame not yet preprocessed')
            second_tensor[i, ...] = frame
            i += 1
        return second_tensor

    def create_input_tensor_given_seconds(self, seconds):
        # create input tensor given list of second matrix
        number_of_second = len(seconds)
        
        if number_of_second != config.seq_len:
            raise ValueError('not correct number of seconds')
        
        tensor = np.zeros(shape=(1, 1, config.seq_len, config.frames_per_step, config.out_H, config.out_W, 7), dtype=np.uint8)

        i=0
        for second in seconds:
            shape_second = second.shape
            if shape_second[0] != config.frames_per_step:
                raise ValueError('not correct number of frame in second matrix')
            tensor[0, 0, i, :, :, :, :] = second
            i += 1

        return tensor

    def create_input_tensor_given_preprocessed_frame(self, frames_collection):
        # create input tensor given list of second composed of list of preprocessed frames
        number_of_second = len(frames_collection)
        
        if number_of_second != config.seq_len:
            raise ValueError('not correct number of seconds')

        compoundend_seconds = []
        for frames in frames_collection:
            compoundend_seconds.append(self.compound_second_frames(frames))

        tensor = self.create_input_tensor_given_seconds(compoundend_seconds)

        return tensor

    def retrieve_hidden_state(self, second_id):
        c = np.zeros(shape=(1, len(config.encoder_lstm_layers), 1, config.hidden_states_dim), dtype=float)
        h = np.zeros(shape=(1, len(config.encoder_lstm_layers), 1, config.hidden_states_dim), dtype=float)
        retrieve_id = second_id - config.seq_len 
        if retrieve_id in self.hidden_states_collection.keys():
            c[0, :, :, :] = self.hidden_states_collection[retrieve_id]['c']
            h[0, :, :, :] = self.hidden_states_collection[retrieve_id]['h']
        else:
            print('\nATTENTION: hiddend state not found')
        return c, h

    def save_hidden_state(self, second_id, c, h):
        if second_id not in self.hidden_states_collection:
            self.hidden_states_collection[second_id] = {}
        self.hidden_states_collection[second_id]['h'] = h
        self.hidden_states_collection[second_id]['c'] = c

    def compute_activity_given_tensor(self, tensor, second_count, dict_obj):     
        # compute results from network given tensor and last second time count 
        c, h =self.retrieve_hidden_state(second_count)
        obj_tensor = self.generate_obj_tensor(dict_obj)
        now_softmax, help_softmax, next_softmax, c3d_softmax, c_out, h_out = self.sess.run([self.now_softmax, self.help_softmax, self.next_softmax, self.c3d_softmax,
                                                                                self.c_out, self.h_out],
                                                                                feed_dict={self.input: tensor,
                                                                                            self.h_input: h,
                                                                                            self.c_input: c,
                                                                                            self.obj_input: obj_tensor})
        if config.debug_frames:
            self.img_save_input(tensor, second_count)
        self.save_hidden_state(second_count, c_out, h_out)

        return now_softmax[0,:4,:], next_softmax, help_softmax[0,:3,:], c3d_softmax
    
    def compute_activity_given_seconds_matrix(self, seconds, second_count, dict_obj):     
        # compute results from network given list of second tensor and last second time count
        tensor = self.create_input_tensor_given_seconds(seconds)

        now_softmax, next_softmax, help_softmax, c3d_softmax = self.compute_activity_given_tensor(tensor, second_count, dict_obj)

        return now_softmax, next_softmax, help_softmax, c3d_softmax

    def compute_activity_given_frame_list(self, frames_collection, second_count, dict_obj):
        # compute results from network given list of list of frames and last second time count
        tensor = self.create_input_tensor_given_preprocessed_frame(frames_collection)

        now_softmax, next_softmax, help_softmax, c3d_softmax = self.compute_activity_given_tensor(tensor, second_count, dict_obj)

        return now_softmax, next_softmax, help_softmax, c3d_softmax

    def save_frame(self, frame_matrix, sec_id, frame_per_step):
        frame_path = 'debug_frames/' + str(sec_id) + '_' + str(frame_per_step)
        cv2.imwrite(frame_path + '_rgb.jpg',frame_matrix[:, :, :3])
        cv2.imwrite(frame_path + '_flow_1.jpg',frame_matrix[:, :, 5])
        cv2.imwrite(frame_path + '_flow_2.jpg',frame_matrix[:, :, 6])
        cv2.imwrite(frame_path + '_heatMat_CH3.jpg',frame_matrix[:, :, 3])
        cv2.imwrite(frame_path + '_pafMat_CH4.jpg',frame_matrix[:, :, 4])

    def img_save_input(self, tensor, sec_id):
        if not os.path.exists('debug_frames/'):
            os.makedirs('debug_frames/')
        shape_tensor = tensor.shape
        print(shape_tensor)
        for frame_per_step in range(shape_tensor[3]):
            img_tensor = tensor[0, 0, -1, frame_per_step, ...]
            self.save_frame(img_tensor, sec_id, frame_per_step)