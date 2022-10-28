# Weather4cast 2022 Starter Kit
#
# Copyright (C) 2022
# Institute of Advanced Research in Artificial Intelligence (IARAI)

# This file is part of the Weather4cast 2022 Starter Kit.
# 
# The Weather4cast 2022 Starter Kit is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
# 
# The Weather4cast 2022 Starter Kit is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Contributors: Aleksandra Gruca, Pedro Herruzo, David Kreil, Stephen Moran


from sklearn.metrics import confusion_matrix
import numpy as np
import torch as t
import sys

def get_confusion_matrix(y_true, y_pred): 
    """get confusion matrix from y_true and y_pred

    Args:
        y_true (numpy array): ground truth 
        y_pred (numpy array): prediction 

    Returns:
        confusion matrix
    """    
    
    labels = [0,1]
    return confusion_matrix(y_true, y_pred, labels=labels).ravel()

def recall_precision_f1_acc(y, y_hat):
    """ returns metrics for recall, precision, f1, accuracy

    Args:
        y (numpy array): ground truth 
        y_hat (numpy array): prediction 

    Returns:
        recall(float): recall/TPR 
        precision(float): precision/PPV
        F1(float): f1-score
        acc(float): accuracy
        csi(float): critical success index
    """  
      
    # pytorch to numpy
    y, y_hat = [o.cpu() for o in [y, y_hat]]
    y, y_hat = [np.asarray(o) for o in [y, y_hat]]

    cm = get_confusion_matrix(y.ravel(), y_hat.ravel())
    if len(cm)==4:
        tn, fp, fn, tp = cm
        recall, precision, F1, acc, csi = 0, 0, 0, 0, 0

        if (tp + fn) > 0:
            recall = tp / (tp + fn)
        
        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        
        if (precision + recall) > 0:
            F1 = 2 * (precision * recall) / (precision + recall)
        
        if (tp + fn + fp) > 0: 
            csi = tp / (tp + fn + fp)

        if (tn+fp+fn+tp) > 0:
            acc = (tn + tp) / (tn+fp+fn+tp)
    else:
        print("FATAL ERROR: cannot create confusion matrix")
        print("EXITING....")
        sys.exit()

    return recall, precision, F1, acc, csi


def iou_class(y_pred: t.Tensor, y_true: t.Tensor):
    #y_true, y_pred = [o.cpu() for o in [y_true, y_pred]]
    #y_true, y_pred = [np.asarray(o) for o in [y_true, y_pred]]
    y_pred = y_pred.int()
    y_true = y_true.int()
    # Outputs: BATCH X H X W
    
    intersection = (y_pred & y_true).float().sum()  # Will be zero if Truth=0 or Prediction=0
    union = (y_pred | y_true).float().sum()  # Will be zero if both are 0
    
    if union>0:
        iou = intersection / union
    else:
        iou = 0

    iou = iou.cpu()
    return iou
