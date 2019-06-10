from help_basic.network_wrapper import NetworkWrapper
from help_recognition.activity_network import ActivityNetwork

import threading
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2


def displayResult(images_pack, label, delay):

    # TODO the below fors can be vectorised for improved performance
    # TODO remove magic numbers
    for s in range(images_pack.shape[1]-1, images_pack.shape[1]): #range(images_pack.shape[1]):
        for fr in range(0,images_pack.shape[2],2):
                frame = images_pack[0, s, fr, :, :, :]
                # print(s, fr, images_pack.shape[2])
                
                # write white label, 1/2 of original font size, top left corner
                # # of image, standard thickness (1)
                cv2.putText(frame, label, (round(frame.shape[0] * 0.05),
                                           round(frame.shape[1] * 0.05)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0xFF, 0xFF, 0xFF), 1)
                cv2.imshow('Recognition results', frame)
                key = cv2.waitKey(delay) & 0xFF
    return key 



class HelpWrapper(NetworkWrapper):
    def __init__(self, *args, **kwargs):
        super(HelpWrapper, self).__init__(*args, **kwargs)

        self.help_pred = None
        self.help_pred_labels = None
        self.y_pred = None
        self.y_pred_labels = None

        with self.sess.as_default(), self.sess.graph.as_default():
            self.nn = ActivityNetwork(self.meta_path, self.checkpoint_path, sess=self.sess, device=self.device)

    def set_images_cache(self, images_cache):
        self.images_cache = images_cache

    def set_data(self, images_pack, images_list, second, objects):
        self.images_pack = images_pack
        self.images_list = images_list
        self.second = second
        self.objects = objects

    def get_data(self):
        return self.help_softmax, self.now_softmax, self.next_softmax

    def get_predictions(self):
        return self.help_pred_labels, self.y_pred_labels, self.help_pred, self.y_pred

    def spin(self):
        t = threading.Thread(name='help_network', target=self._execute)
        t.start()
        return t
    
    def _execute(self):
        start = time.time()

        # processing images
        with self.sess.as_default(), self.sess.graph.as_default():
            x, x_mask = self.images_cache.process_images(self.images_pack, self.images_list)
            processing = time.time()
            self.now_softmax, self.next_softmax, self.help_softmax, _ = \
                self.nn.compute_activity_given_tensor(x, self.second, self.objects)
            end = time.time()
            print('processing time:{} secs'.format(processing - start))
            print('network time:{} secs'.format(end - processing))
            print('total time:{} secs'.format(end - start))
            
            self.y_pred = np.expand_dims(np.argmax(self.now_softmax, axis=1), 0)
            self.y_pred_labels = [self.labels[i] for i in self.y_pred.tolist()[0]]

            self.help_pred = np.expand_dims(np.argmax(self.help_softmax, axis=1), 0)
            self.help_pred_labels = [self.labels[i] for i in self.help_pred.tolist()[0]]            

    def visualize(self, delay):
        if self.images_pack is not None:
            # printing outputs
            # print(self.help_pred_labels)
            
            # # estimated activity publishing
            # if self.output_proxy:
            #     publish_string(self.output_proxy, self.y_pred_labels[0])
            #     publish_string(self.output_proxy, self.help_pred_labels[0])
            
            # frames/activity display
            if self.display:
                key = displayResult(self.images_pack, self.y_pred_labels[-1], delay)
                if key == ord('r'):
                    # Press key `r` to reset network
                    print("Resetting hidden state")
                    self.nn.hidden_states_collection = {}
