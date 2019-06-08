import __init__
from __init__ import PROJECT_DIR

from tf_mask_rcnn.MaskSH import *
import cv2

DATA_PATH = os.path.join(PROJECT_DIR, 'Data', 'dataset')
MODEL_PATH = os.path.join(PROJECT_DIR, 'Data','model_data','tf_mask_rcnn')

if __name__ == "__main__":
                
        m = MaskSH(MODEL_PATH)

        image = cv2.imread(os.path.join(DATA_PATH,"image0.png"))
        
        for i in range(2):
                a, result = m.detect([image])
                r = result[0]
                visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                m.class_list, r['scores'])

                for elem in a.keys():
                        print(elem, '--->', a[elem])
