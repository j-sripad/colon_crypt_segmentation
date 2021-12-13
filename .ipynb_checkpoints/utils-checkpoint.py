from torch.nn.modules.loss import _Loss
import segmentation_models_pytorch as sm
import torch
import torch.nn.functional as F


class DiceLoss(_Loss):
    def __init__(self,eps=1e-7):
        super().__init__()
        self.eps = eps
    
    def forward(self,input:torch.Tensor,target:torch.Tensor)->torch.Tensor:
        #cite https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
        """Computes the Sørensen–Dice loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the dice loss so we
        return the negated dice loss.
        Args:
            true: a tensor of shape [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
        Returns:
            dice_loss: the Sørensen–Dice loss.
        """
        num_classes = input.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[target.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(input)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[target.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(input, dim=1)
        true_1_hot = true_1_hot.type(input.type())
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
        return (1 - dice_loss)
        
        

    

        