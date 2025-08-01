from deepali.losses.functional import mse_loss, dice_loss, lcc_loss, grad_loss
import torch

class VoxelMorphDataLoss(torch.nn.Module):
    '''
    Voxelmorph loss functions
    '''
    def __init__(self, use_mse=True, smoothness_weight=0.005):
        '''
        Constructor for the VoxelmorphLoss class

        :param use_mse: Use Mean Squared Error loss if True, else use Local Correlation Coefficient loss
        :param smoothness_weight: Weight for the smoothness loss
        '''
        super(VoxelMorphDataLoss, self).__init__()
        self.use_mse = use_mse
        self.smoothness_weight = smoothness_weight

    def lcc_loss(self, y_true, y_pred):
        '''
        Compute the Local Correlation Coefficient loss
        '''
        return lcc_loss(y_pred, y_true)

    def mse(self, y_true, y_pred):
        '''
        Compute the Mean Squared Error
        '''
        return mse_loss(y_true, y_pred)
    
    def smoothness_loss(self, flow):
        '''
        Compute the smoothness loss
        '''
        return grad_loss(flow)

    def forward(self, y_true, y_pred, flow):
        '''
        Forward pass for the loss function
        '''
        if self.use_mse:
            loss = self.mse(y_true, y_pred) + self.smoothness_weight * self.smoothness_loss(flow)
        else:
            loss = self.lcc_loss(y_true, y_pred) + self.smoothness_weight * self.smoothness_loss(flow)
        return loss
    
class VoxelMorphSegLoss(torch.nn.Module):
    def __init__(self):
        super(VoxelMorphSegLoss, self).__init__()
    
    def dice_loss(self, y_true, y_pred):
        return dice_loss(y_true, y_pred)
    
    def forward(self, y_true, y_pred):
        return self.dice_loss(y_true, y_pred)
    
class VoxelMorphLoss(torch.nn.Module):
    def __init__(self, data_loss, seg_loss=None, seg_weight=0.01):
        super(VoxelMorphLoss, self).__init__()
        self.data_loss = data_loss
        self.seg_loss = seg_loss
        self.seg_weight = seg_weight
    
    def _forward_seg(self, y_true, y_pred, flow, y_seg_true, y_seg_pred):
        data_loss = self.data_loss(y_true, y_pred, flow)
        seg_loss = self.seg_loss(y_seg_true, y_seg_pred)
        return data_loss + self.seg_weight * seg_loss
    
    def _forward_data(self, y_true, y_pred, flow):
        return self.data_loss(y_true, y_pred, flow)
    
    def forward(self, y_true, y_pred, flow, y_seg_true=None, y_seg_pred=None):
        if self.seg_loss is not None:
            return self._forward_seg(y_true, y_pred, flow, y_seg_true, y_seg_pred)
        return self._forward_data(y_true, y_pred, flow)

    