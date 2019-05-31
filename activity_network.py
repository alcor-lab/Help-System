
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

# from network_seq import activity_network
# from network_seq import Training
# from network_seq import Input_manager

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x for x in local_device_protos if x.device_type == 'GPU']
    # return local_device_protos

class activity_network:
    def __init__(self, sess=None):
        # creating a Session
        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess

        # load architecture in graph and weights in session and initialize

        self.graph = tf.get_default_graph()
        # self.architecture = tf.train.import_meta_graph('model/activity_network_model.ckpt.meta')
        # self.latest_ckp = tf.train.latest_checkpoint('model')

        # self.architecture.restore(self.sess, self.latest_ckp)

        # number_of_classes = IO_tool.num_classes
        # available_gpus = get_available_gpus()
        # j=0
        # Net_collection = {}
        # Input_net = Input_manager(len(available_gpus), IO_tool)
        # for device in available_gpus:
        #     with tf.device(device.name):
        #         print(device.name)
        #         with tf.variable_scope('Network') as scope:
        #             if j>0:
        #                 scope.reuse_variables()
        #             Net_collection['Network_' + str(j)] = activity_network(number_of_classes, Input_net, j, IO_tool)
        #             j = j+1
        # with tf.device(available_gpus[-1].name):
        # Train_Net = Training(Net_collection, IO_tool)
        # IO_tool.start_openPose()
        # train_writer = tf.summary.FileWriter("logdir/train", sess.graph)
        # val_writer = tf.summary.FileWriter("logdir/val", sess.graph)
        
        # IO_tool.openpose.load_openpose_weights()
        # sess.run(Train_Net.init)
        # Train_Net.model_saver.restore(sess, tf.train.latest_checkpoint('./checkpoint'))

        # self.architecture = tf.train.import_meta_graph('./checkpoint')
        self.architecture = tf.train.import_meta_graph('model/activity_network_model.ckpt.meta')
        self.create_graph_log()
        ckpts = tf.train.latest_checkpoint('./checkpoint')
        vars_in_checkpoint = tf.train.list_variables(ckpts)
        var_rest = []
        for el in vars_in_checkpoint:
            var_rest.append(el[0])
        variables = tf.contrib.slim.get_variables_to_restore()
        var_list = [v for v in variables if v.name.split(':')[0] in var_rest]
        loader = tf.train.Saver(var_list=var_list)
        loader.restore(self.sess, ckpts)

        # self.saver = self.graph.get_tensor_by_name("Saver_and_Loader/whole_saver/saver:0")
        # self.saver.restore(self.sess, self.latest_ckp )

        # tf.saved_model.loader.load(self.sess, export_dir = 'model/activity_network_model.ckpt.meta')

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
        self.input = self.graph.get_tensor_by_name("Inputs/Input/Input:0")
        self.h_input = self.graph.get_tensor_by_name("Inputs/Input/h_input:0")
        self.c_input = self.graph.get_tensor_by_name("Inputs/Input/c_input:0")
        self.c3d_softmax = self.graph.get_tensor_by_name("Network/Activity_Recognition_Network/c3d_classifier/Softmax:0")
        self.now_softmax = self.graph.get_tensor_by_name("Network/Activity_Recognition_Network/Now_Decoder_inference/softmax_out:0")
        self.help_softmax = self.graph.get_tensor_by_name("Network/Activity_Recognition_Network/Help_Decoder_inference/softmax_out:0")
        self.next_softmax = self.graph.get_tensor_by_name("Network/Activity_Recognition_Network/Next_classifier/softmax_out:0")
        self.c_out = self.graph.get_tensor_by_name("Network/Activity_Recognition_Network/Lstm_encoder/c_out:0")
        self.h_out = self.graph.get_tensor_by_name("Network/Activity_Recognition_Network/Lstm_encoder/h_out:0")

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
        pafMat = np.amax(pafMat, axis=2)
        heatMat = cv2.resize(heatMat, dsize=(config.out_H, config.out_W), interpolation=cv2.INTER_CUBIC)
        pafMat = cv2.resize(pafMat, dsize=(config.out_H, config.out_W), interpolation=cv2.INTER_CUBIC)
        norm_pafMat = cv2.normalize(pafMat, None, 0, 255, cv2.NORM_MINMAX)
        norm_heatMat = cv2.normalize(heatMat, None, 0, 255, cv2.NORM_MINMAX)

        # pafMat, heatMat = self.IO_tool.openpose.compute_pose_frame(img)
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
        frame[..., 3] = pafMat
        frame[..., 4] = heatMat
        frame[..., 5:7] = flow
        return frame

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
        
        tensor = np.zeros(shape=(4, 1, config.seq_len, config.frames_per_step, config.out_H, config.out_W, 7), dtype=np.uint8)

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

    def compute_activity_given_tensor(self, tensor, second_count):     
        # compute results from network given tensor and last second time count 
        c, h =self.retrieve_hidden_state(second_count)

        for seq in range(tensor.shape[2]):
            for frame in range(tensor.shape[3]):
                cv2.imwrite('test_pic/'+str(second_count) +str(seq) +str(frame)+ '_rgb.jpg',tensor[0,0,seq,frame, :, :, :3])
                cv2.imwrite('test_pic/'+str(second_count) +str(seq) +str(frame) + '_flow_1.jpg',tensor[0,0,seq,frame, :, :, 5])
                cv2.imwrite('test_pic/'+str(second_count) +str(seq) +str(frame) + '_flow_2.jpg',tensor[0,0,seq,frame, :, :, 6])
                cv2.imwrite('test_pic/'+str(second_count) +str(seq) +str(frame) + '_pafMat.jpg',tensor[0,0,seq,frame, :, :, 3])
                cv2.imwrite('test_pic/'+str(second_count) +str(seq) +str(frame) + '_heatMat.jpg',tensor[0,0,seq,frame, :, :, 4])


        now_softmax, help_softmax, next_softmax, c3d_softmax, c_out, h_out = self.sess.run([self.now_softmax, self.help_softmax, self.next_softmax, self.c3d_softmax,
                                                                                self.c_out, self.h_out],
                                                                                feed_dict={self.input: tensor,
                                                                                            self.h_input: h,
                                                                                            self.c_input: c})

        self.save_hidden_state(second_count, c_out, h_out)

        return now_softmax[0,:4,:], next_softmax, help_softmax[0,:3,:], c3d_softmax
    
    def compute_activity_given_seconds_matrix(self, seconds, second_count):     
        # compute results from network given list of second tensor and last second time count
        tensor = self.create_input_tensor_given_seconds(seconds)

        now_softmax, next_softmax, help_softmax, c3d_softmax = self.compute_activity_given_tensor(tensor, second_count)

        return now_softmax, next_softmax, help_softmax, c3d_softmax

    def compute_activity_given_frame_list(self, frames_collection, second_count):
        # compute results from network given list of list of frames and last second time count
        tensor = self.create_input_tensor_given_preprocessed_frame(frames_collection)

        now_softmax, next_softmax, help_softmax, c3d_softmax = self.compute_activity_given_tensor(tensor, second_count)

        return now_softmax, next_softmax, help_softmax, c3d_softmax

    
# activity_network = activity_network()