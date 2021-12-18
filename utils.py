from torch.nn.modules.loss import _Loss
import segmentation_models_pytorch as sm
import torch
import torch.nn.functional as F
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
import cv2



class DiceScore(nn.Module):
    
    #from segmentation_models torch belt source code
    
    def __init__(self,eps = 1e-6,loss=False) -> None:
        super(DiceScore, self).__init__()
        self.eps =  1e-6
        self.loss = loss
        

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        target_one_hot = torch.eye(2)[target.squeeze(1)]
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(1. - dice_score) if self.loss else torch.mean(dice_score)
   
    
    
    
def get_train_transforms()->transforms:
    train_transform =  A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9,
                             border_mode=cv2.BORDER_REFLECT),
            A.OneOf([
                A.ElasticTransform(p=.3),
                A.GaussianBlur(p=.3),
                A.GaussNoise(p=.3),
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.1),
                A.PiecewiseAffine(p=0.3),
            ], p=0.3),
            A.OneOf([
                A.HueSaturationValue(15,25,0),
                A.CLAHE(clip_limit=2),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            ], p=0.3),


        ])
    return train_transform

def get_val_transforms()->transforms:
    validation_transform = A.Compose([ToTensorV2()])
    
    return validation_transform

def preprocessing_fucntion(preprocesing_function=None):
    return A.Compose([A.Lambda(image=preprocesing_function),ToTensorV2()])
    