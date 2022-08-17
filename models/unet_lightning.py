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


import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.evaluate import *        

#models
from models.baseline_UNET3D import UNet as Base_UNET3D # 3_3_2 model selection

VERBOSE = False
#VERBOSE = True

class UNet_Lightning(pl.LightningModule):
    def __init__(self, UNet_params: dict, params: dict,
                 **kwargs):
        super(UNet_Lightning, self).__init__()

        self.in_channels = params['in_channels']
        self.start_filts = params['init_filter_size']
        self.model = Base_UNET3D(in_channels=self.in_channels,start_filts =  self.start_filts)

        self.save_hyperparameters()
        self.params = params
        #self.example_input_array = np.zeros((44,252,252))
   
        self.main_metric = 'BCE with logits' #mse [log(y+1)-yhay]'

        self.val_batch = 0
        
        self.prec = 7

        pos_weight = torch.tensor(params['pos_weight']);
        if VERBOSE: print("Positive weight:",pos_weight);

        self.loss = params['loss']
        self.bs = params['batch_size']
        self.loss_fn = {
            'smoothL1': nn.SmoothL1Loss(), 'L1': nn.L1Loss(), 'mse': F.mse_loss,
            'BCELoss': nn.BCELoss(), 
            'BCEWithLogitsLoss': nn.BCEWithLogitsLoss(pos_weight=pos_weight), 'CrossEntropy': nn.CrossEntropyLoss(), 'DiceBCE': DiceBCELoss(), 'DiceLoss': DiceLoss()
            }[self.loss]

        self.relu = nn.ReLU() # None
        t = f"============== n_workers: {params['n_workers']} | batch_size: {params['batch_size']} \n"+\
            f"============== loss: {self.loss} | weight: {pos_weight} (if using BCEwLL)"
        print(t)
    
    def on_fit_start(self):
        """ create a placeholder to save the results of the metric per variable """
        metric_placeholder = {self.main_metric: -1}
        self.logger.log_hyperparams(self.hparams, metric_placeholder)
        
    def forward(self, x):
        x = self.model(x)
        #if self.loss =='BCELoss':
        #x = self.relu(x)
        return x

    def retrieve_only_valid_pixels(self, x, m):
        """ we asume 1s in mask are invalid pixels """
        ##print(f"x: {x.shape} | mask: {m.shape}")
        return x[~m]

    def get_target_mask(self, metadata):
        mask = metadata['target']['mask']
        #print("mask---->", mask.shape)
        return mask
    
    def _compute_loss(self, y_hat, y, agg=True, mask=None):
        if mask is not None:
            y_hat = self.retrieve_only_valid_pixels(y_hat, mask)
            y = self.retrieve_only_valid_pixels(y, mask)
        # print("================================================================================")
        # print(y_hat.shape, y_hat.min(), y_hat.max())
        # print(y.shape, y.min(), y.max())
        if agg:
            loss = self.loss_fn(y_hat, y)
        else:
            loss = self.loss_fn(y_hat, y, reduction='none')
        return loss
    
    def training_step(self, batch, batch_idx, phase='train'):
        x, y, metadata  = batch
        if VERBOSE:
            print('x', x.shape, 'y', y.shape, '----------------- batch')
        y_hat = self.forward(x)
        if VERBOSE:
            print('y_hat', y_hat.shape, 'y', y.shape, '----------------- model')
        mask = self.get_target_mask(metadata)
        loss = self._compute_loss(y_hat, y, mask=mask)
        #recall, precision, F1, acc = recall_precision_f1_acc(y, y_hat)
        #LOGGING
        #values = {'{phase}_loss': loss, '{phase}_acc': acc, '{phase}_recall': recall, '{phase}_precision': precision, 'F1': F1}
        self.log(f'{phase}_loss', loss,batch_size=self.bs, sync_dist=True)
        return loss
                
    def validation_step(self, batch, batch_idx, phase='val'):
        #data_start = timer()
        x, y, metadata  = batch
        #data_end = timer()
        if VERBOSE:
            print('x', x.shape, 'y', y.shape, '----------------- batch')
        y_hat = self.forward(x)
        mask = self.get_target_mask(metadata)
        if VERBOSE:
            print('y_hat', y_hat.shape, 'y', y.shape, '----------------- model')

        loss = self._compute_loss(y_hat, y, mask=mask)

        # todo: add the same plot as in `test_step`

        if self.loss=="BCEWithLogitsLoss":
            print("applying thresholds to y_hat logits")
            # set the logits threshold equivalent to sigmoid(x)>=0.5
            idx_gt0 = y_hat>=0
            y_hat[idx_gt0] = 1
            y_hat[~idx_gt0] = 0
        
        recall, precision, F1, acc, csi = recall_precision_f1_acc(y, y_hat)
        iou = iou_class(y_hat, y)

        #LOGGING
        self.log(f'{phase}_loss', loss, batch_size=self.bs, sync_dist=True)
        values = {'val_acc': acc, 'val_recall': recall, 'val_precision': precision, 'val_F1': F1, 'val_iou': iou, 'val_CSI': csi}
        self.log_dict(values, batch_size=self.bs, sync_dist=True)
    
        return loss

    def validation_epoch_end(self, outputs, phase='val'):
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log(f'{phase}_loss_epoch', avg_loss, prog_bar=True,
                 batch_size=self.bs, sync_dist=True)
        self.log(self.main_metric, avg_loss, batch_size=self.bs, sync_dist=True)


    def test_step(self, batch, batch_idx, phase='test'):
        x, y, metadata = batch
        if VERBOSE:
            print('x', x.shape, 'y', y.shape, '----------------- batch')
        y_hat = self.forward(x)
        mask = self.get_target_mask(metadata)
        if VERBOSE:
            print('y_hat', y_hat.shape, 'y', y.shape, '----------------- model')
        loss = self._compute_loss(y_hat, y, mask=mask)
        ## todo: add the same plot as in `test_step`
        if self.loss=="BCEWithLogitsLoss":
            print("applying thresholds to y_hat logits")
            # set the logits threshold equivalent to sigmoid(x)>=0.5
            idx_gt0 = y_hat>=0
            y_hat[idx_gt0] = 1
            y_hat[~idx_gt0] = 0
        
        recall, precision, F1, acc, csi = recall_precision_f1_acc(y, y_hat)
        iou = iou_class(y_hat, y)

        #LOGGING
        self.log(f'{phase}_loss', loss, batch_size=self.bs, sync_dist=True)
        values = {'test_acc': acc, 'test_recall': recall, 'test_precision': precision, 'test_F1': F1, 'test_iou': iou, 'test_CSI': csi}
        self.log_dict(values, batch_size=self.bs, sync_dist=True)
        
        return 0, y_hat

    def predict_step(self, batch, batch_idx, phase='predict'):
        x, y, metadata = batch
        y_hat = self.model(x)
        mask = self.get_target_mask(metadata)
        if VERBOSE:
            print('y_hat', y_hat.shape, 'y', y.shape, '----------------- model')
        if self.loss=="BCEWithLogitsLoss":
            print("applying thresholds to y_hat logits")
            # set the logits threshold equivalent to sigmoid(x)>=0.5
            idx_gt0 = y_hat>=0
            y_hat[idx_gt0] = 1
            y_hat[~idx_gt0] = 0
        return y_hat

    def configure_optimizers(self):
        if VERBOSE: print("Learning rate:",self.params["lr"], "| Weight decay:",self.params["weight_decay"])
        optimizer = torch.optim.AdamW(self.parameters(),
                                     lr=float(self.params["lr"]),weight_decay=float(self.params["weight_decay"])) 
        return optimizer

    def seq_metrics(self, y_true, y_pred):
        text = ''
        cm = confusion_matrix(y_true, y_pred).ravel()
        if len(cm)==4:
            tn, fp, fn, tp = cm
            recall, precision, F1 = 0, 0, 0

            if (tp + fn) > 0:
                recall = tp / (tp + fn)
            r = f'r: {recall:.2f}'
            
            if (tp + fp) > 0:
                precision = tp / (tp + fp)
            p = f'p: {precision:.2f}'

            if (precision + recall) > 0:
                F1 = 2 * (precision * recall) / (precision + recall)
            f = f'F1: {F1:.2f}'

            acc = (tn + tp) / (tn+fp+fn+tp)
            text = f"{r} | {p} | {f} | acc: {acc} "

        return text

def main():
    print("running")
if __name__ == 'main':
    main()

#PyTorch
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, targets, inputs, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()      
        print('intersection', intersection)
        print('inputs', inputs.sum())
        print('targets', targets.sum())                     
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        print(1-dice)
        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
