# Cross-Entropy Loss composed by two terms
# 1. Cross-Entropy Loss for first augmented image
# 2. Cross-Entropy Loss for second augmented image

import torch as th
import torch.nn as nn
import torch.nn.functional as F
#import utils  # missing file


class Loss(nn.Module):
    def __init__(self, row_tau=0.1, col_tau=0.1, eps=1e-8):
        super(Loss, self).__init__()
        self.row_tau = row_tau
        self.col_tau = col_tau
        self.eps = eps

    def forward(self, cls_out):
        total_loss = 0.0
        num_loss_terms = 0

        
        # We have only one classifier - need to be adjusted


        #for cls_idx, cls_out in enumerate(out):  # classifiers 
            # gather samples from all workers
       # cls_out = [utils.AllGather.apply(x).float() for x in cls_out]

        #const = cls_out[0].shape[0] / cls_out[0].shape[1]
        
        #batch_size = len(cls_out[0])
        C = len(cls_out[0][0])
        N = len(cls_out[0])
        const = N/C
        
        #print("num classes: ", C)
        #print("batch size: ", N)
        #print("const: ", const)
        target = []

        for view_i_idx, view_i in enumerate(cls_out):
            #print("view_i_idx: ", view_i_idx)
            #print("view_i: ", (view_i))
            #print("view_i: ", th.stack(view_i))
            #print("###1:  ",view_i[0].requires_grad)
            view_i_target = F.softmax(view_i/ self.col_tau, dim=0)
            #view_i_target = F.softmax(th.stack(view_i)/ self.col_tau, dim=0)
    
            #print("###2:  ",view_i_target.requires_grad)
            #print("view_i_target: ", view_i_target)
            #view_i_target = utils.keep_current(view_i_target)
            view_i_target = F.normalize(view_i_target, p=1, dim=1, eps=self.eps)
            #print("###1:  ",view_i_target.requires_grad)
            target.append(view_i_target)
        

        for view_j_idx, view_j in enumerate(cls_out):  # view j
            view_j_pred = F.softmax(view_j / self.row_tau, dim=1)
            #view_j_pred = F.softmax(th.stack(view_j) / self.row_tau, dim=1)
            
            view_j_pred = F.normalize(view_j_pred, p=1, dim=0, eps=self.eps)
            #view_j_pred = utils.keep_current(view_j_pred)
            view_j_log_pred = th.log(const * view_j_pred + self.eps)

            for view_i_idx, view_i_target in enumerate(target):

                if view_i_idx == view_j_idx or (view_i_idx >= 2 and view_j_idx >= 2):
                    # skip cases when it's the same view, or when both views are 'local' (small)
                    continue

                # cross entropy
                loss_i_j = - th.mean(th.sum(view_i_target * view_j_log_pred, dim=1))
                total_loss += loss_i_j
                num_loss_terms += 1

        total_loss /= num_loss_terms
        
        return total_loss

