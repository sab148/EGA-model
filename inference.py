
"""
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import cv2
import numpy as np
import torch.nn.functional as F
from evaluation import BoxEvaluator
from evaluation import MaskEvaluator
from evaluation import configure_metadata
from util import t2n
import pickle
_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = 224


def normalize_scoremap(cam):
    """
    Args:
        cam: numpy.ndarray(size=(H, W), dtype=np.float)
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(cam).any():
        return np.zeros_like(cam)
    if cam.min() == cam.max():
        return np.zeros_like(cam)
    cam -= cam.min()
    cam /= cam.max()
    return cam


class CAMComputer(object):
    def __init__(self, model, loader, metadata_root, mask_root,
                 iou_threshold_list, dataset_name, split, 
                 multi_contour_eval, tencrop=False, cam_curve_interval=.001):
        self.model = model
        self.model.eval()
        self.loader = loader
        self.tencrop=tencrop
        metadata = configure_metadata(metadata_root)
        cam_threshold_list = list(np.arange(0, 1, cam_curve_interval))

        self.evaluator = {"OpenImages": MaskEvaluator,
                          "CUB": BoxEvaluator,
                          "ILSVRC": BoxEvaluator
                          }[dataset_name](metadata=metadata,
                                          dataset_name=dataset_name,
                                          split=split,
                                          cam_threshold_list=cam_threshold_list,
                                          iou_threshold_list=iou_threshold_list,
                                          mask_root=mask_root,
                                          multi_contour_eval=multi_contour_eval)

    def compute_and_evaluate_cams(self):
        print("Computing and evaluating cams.")
        cnt = 0
        pred_prob = []
        with open('pred_prob.pkl', 'rb')  as f :
            pred_prob = pickle.load(f)
        for images, targets, image_ids in self.loader:
            image_size = images.shape[2:]

            images = images.cuda()
            targets = targets.cuda()
            cams, pred = self.model(images, targets, return_cam=True)

            pred = pred.argmax(dim=1)
            cams = t2n(cams)
 #           print('cams', cams.shape)
            for i , (cam, image_id) in enumerate(zip(cams, image_ids)):
                cnt += 1
#                print('pred', pred[i])
 ##               print('tar', targets[i])
                if pred_prob[cnt-1] != targets[i] :
                  continue
#                print('cam', cam.shape)
                cam_resized = cv2.resize(cam, image_size, interpolation=cv2.INTER_CUBIC)
                cam_normalized = normalize_scoremap(cam_resized)                
                if i % 70 == 0:
                  dessin = True
                else :
                  dessin = False
                self.evaluator.accumulate(cam_normalized, image_id, dessin)
        return self.evaluator.compute(cnt)
