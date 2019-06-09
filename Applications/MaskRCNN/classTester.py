import __init__
from __init__ import PROJECT_DIR

from tf_mask_rcnn.MaskSH import *
import cv2
import matplotlib.pyplot as plt

DATA_PATH = os.path.join(PROJECT_DIR, 'Data', 'dataset')
MODEL_PATH = os.path.join(PROJECT_DIR, 'Data','model_data','tf_mask_rcnn')

DEVICE_LIST = ("/cpu:0","/gpu:0","/gpu:1","/gpu:2")

THREADS = 7

def network_config():
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=THREADS)
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement=True
    tf_config.log_device_placement=False

    return tf_config
    
if __name__ == "__main__":
        tf_config = network_config()
        with tf.Session(config=tf_config) as sess:       
                m = MaskSH(MODEL_PATH, sess=sess, device=DEVICE_LIST[1])

                image = cv2.imread(os.path.join(DATA_PATH, "image0.png"))
                ax = plt.axes()
                for i in range(2):
                        a, result = m.detect([image])
                        r = result[0]
                        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                        m.class_list, r['scores'], pause=1, ax = ax)

                        for elem in a.keys():
                                print(elem, '--->', a[elem])