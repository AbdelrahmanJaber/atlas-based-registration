from deepali.losses.functional import dice_loss, mi_loss

class ImageMetrics:
    '''
    Class to compute image metrics
    '''
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon

    def dice_loss(self, y_pred, y_true, threshold=0.5):
        '''
        Compute the Dice loss
        '''
        # Binarize predictions and ground truth based on the threshold
        y_pred_bin = (y_pred >= threshold).float()
        y_true_bin = (y_true >= threshold).float()
        return dice_loss(y_pred_bin, y_true_bin, reduction='mean')
    
    def mi_loss(self, y_pred, y_true):
        '''
        Compute the Mutual Information loss
        '''
        return mi_loss(y_pred, y_true, normalized=True)