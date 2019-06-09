from help_basic.network_wrapper import NetworkWrapper
from help_recognition.activity_network import ActivityNetwork

import threading
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2


def displayResult(images_pack, label):

    # TODO the below fors can be vectorised for improved performance
    # TODO remove magic numbers
    for id in range(images_pack.shape[1]-1,images_pack.shape[1]): #range(images_pack.shape[1]):
        for fr in range(images_pack.shape[2]):
                frame = images_pack[0, id, fr, :, :, :]
                # write white label, 1/2 of original font size, top left corner
                # # of image, standard thickness (1)
                cv2.putText(frame, label, (round(frame.shape[0] * 0.05),
                                           round(frame.shape[1] * 0.05)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0xFF, 0xFF, 0xFF), 1)
                cv2.imshow('Recognition results', frame)
                cv2.waitKey(1)


class HelpWrapper(NetworkWrapper):
    def __init__(self, *args, **kwargs):
        super(HelpWrapper, self).__init__(*args, **kwargs)

        with self.sess.as_default(), self.sess.graph.as_default():
            self.nn = ActivityNetwork(self.meta_path, self.checkpoint_path, sess=self.sess, device=self.device)

    def set_images_cache(self, images_cache):
        self.images_cache = images_cache

    def set_data(self, images_pack, images_list, second):
        self.images_pack = images_pack
        self.images_list = images_list
        self.second = second

    def get_data(self):
        return self.help_softmax

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
                self.nn.compute_activity_given_tensor(x, self.second)
            end = time.time()
            print('processing time:{} secs'.format(processing - start))
            print('network time:{} secs'.format(end - processing))
            print('total time:{} secs'.format(end - start))
            self.y_pred = np.expand_dims(np.argmax(self.now_softmax, axis=1), 0)
            self.help_pred = np.expand_dims(np.argmax(self.help_softmax, axis=1), 0)
            # return images_pack, y_pred, help_pred

    def visualize(self):
        if self.images_pack is not None:
            # printing outputs
            y_pred_labels = [self.labels[i] for i in self.y_pred.tolist()[0]]
            help_pred_labels = [self.labels[i] for i in self.help_pred.tolist()[0]]
            print(help_pred_labels)
            
            # estimated activity publishing
            if self.output_proxy:
                publish_string(output_proxy, y_pred_labels[0])
                publish_string(output_proxy, help_pred_labels[0])
            
            # frames/activity display
            if self.display:
                displayResult(self.images_pack, y_pred_labels[-1])
