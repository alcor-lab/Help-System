from abc import ABC, abstractmethod

import threading

class NetworkWrapper(ABC):

    def __init__(self, meta_path='', checkpoint_path='', sess=None, graph=None, device=None, display=True, labels=[], output_proxy=None):
        self.meta_path = meta_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.sess = sess
        self.graph = graph
        self.display = display
        self.labels = labels
        self.output_proxy = output_proxy
    
    @abstractmethod
    def set_data():
        pass

    def spin_once(self):
        t = self.spin()
        t.join()
    
    @abstractmethod
    def get_data():
        pass

    @abstractmethod
    def spin(self):
        pass

