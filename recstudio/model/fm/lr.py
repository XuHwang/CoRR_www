import torch
from collections import OrderedDict
from ..basemodel import BaseRanker
from ..module import ctr, LambdaLayer, MLPModule, HStackLayer
from ..loss_func import BCEWithLogitLoss
from recstudio.data.dataset import MFDataset


class LR(BaseRanker):

    @staticmethod
    def _get_dataset_class():
        return MFDataset

    def _get_scorer(self, train_data):
        return ctr.LinearLayer(self.fields, train_data)

    def _get_loss_func(self):
        return BCEWithLogitLoss(threshold=self.rating_threshold)
