# Deeplab_network package
from .model import Model
from .network import Deeplab_v2, ResNet_segmentation, Deeplab_v2_TF2, ResNet_segmentation_TF2

__all__ = ['Model', 'Deeplab_v2', 'ResNet_segmentation', 'Deeplab_v2_TF2', 'ResNet_segmentation_TF2']
