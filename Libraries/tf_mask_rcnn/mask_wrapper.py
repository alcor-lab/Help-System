from help_basic.network_wrapper import NetworkWrapper
from tf_mask_rcnn.MaskSH import MaskSH
from tf_mask_rcnn import visualize

import threading
import time
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image


class MaskWrapper(NetworkWrapper):
    def __init__(self, *args, **kwargs):
        super(MaskWrapper, self).__init__(*args, **kwargs)

        with self.sess.as_default(), self.sess.graph.as_default():
            self.nn = MaskSH(self.checkpoint_path, sess=self.sess, device=self.device)
    
        self.class_list = self.nn.class_list

        self.output = None
        self.results = None
        
        self.data_ready = False
        self._data_thread = None

        if self.display:
            self.ax = plt.axes()


    def visualize(self):
        for elem in self.output.keys():
            print(elem, '--->', self.output[elem])
        print([self.class_list[x] for x in self.results[1]['class_ids']])
        print([x for x in self.results[1]['scores']])
        # if self.output_proxy:
        #     pass

        if self.display:
            r = self.results[0]
            visualize.display_instances(self.images[0], r['rois'], r['masks'], r['class_ids'],
                                            self.class_list, r['scores'], pause=0.5, ax=self.ax)

    def set_data(self, images):
        self._preprocess_data(images)
    
    def _preprocess_data(self, images):
        self.data_ready = False
        start = time.monotonic()
        images = [im[:,:,::-1] for im in images]
        images = [np.array(Image.fromarray(im).resize((int(1080*4/3),1080), resample=Image.BICUBIC)) for im in images]
        # print("Mask Preprocessing time:",time.monotonic()-start)
        self.images = images
        self.data_ready = True

    def prepare_data(self, images):
        self._data_thread = threading.Thread(target=self._preprocess_data, args=(images,))
        self._data_thread.start()

    def get_data(self):
        return self.output, self.results

    def spin(self):
        t = threading.Thread(name='mask_network', target=self._execute)
        t.start()
        return t

    def _execute(self):
        start = time.time()
            
        if self._data_thread:
            self._data_thread.join()
            self._data_thread = None

        if not self.data_ready: 
            self.output = None
            self.results = None
            return

        preparation = time.time()
        with self.sess.as_default(), self.sess.graph.as_default(): 
            self.output, self.results = self.nn.detect(self.images.copy())
        end = time.time()
        
        # print('Mask data retrieval time:{} secs'.format(preparation - start))
        # print('Mask network time:{} secs'.format(end - preparation))
        # print('Mask total time:{} secs'.format(end - start))
            



