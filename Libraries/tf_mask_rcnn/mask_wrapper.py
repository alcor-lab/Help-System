from help_basic.network_wrapper import NetworkWrapper
from tf_mask_rcnn.MaskSH import MaskSH
from tf_mask_rcnn import visualize

import threading
import time
import matplotlib.pyplot as plt
import numpy as np

class MaskWrapper(NetworkWrapper):
    def __init__(self, *args, **kwargs):
        super(MaskWrapper, self).__init__(*args, **kwargs)

        with self.sess.as_default(), self.sess.graph.as_default():
            self.nn = MaskSH(self.checkpoint_path, sess=self.sess, device=self.device)
    
        self.class_list = self.nn.class_list

        if self.display:
            self.colors = visualize.random_colors(len(self.class_list))
            self.ax = plt.axes()


    def visualize(self):
        for elem in self.output.keys():
            print(elem, '--->', self.output[elem])
        
        if self.output_proxy:
            pass

        if self.display:
            r = self.results[0]
            visualize.display_instances(self.images[0], r['rois'], r['masks'], r['class_ids'],
                                            self.class_list, r['scores'], colors=self.colors, pause=0.5, ax=self.ax)

    def set_data(self, images):
        self.images = images

    def get_data(self):
        return self.output, self.results

    def spin(self):
        t = threading.Thread(name='mask_network', target=self._execute)
        t.start()
        return t

    def _execute(self):
        start = time.time()
            
        preparation = time.time()
        with self.sess.as_default(), self.sess.graph.as_default(): 
            self.output, self.results = self.nn.detect(self.images)
        end = time.time()
        
        print('data retrieval time:{} secs'.format(preparation - start))
        print('network time:{} secs'.format(end - preparation))
        print('total time:{} secs'.format(end - start))
            



