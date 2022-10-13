import imp
from multiprocessing.spawn import import_main_path
import torch
from recstudio.data.dataset import SeqDataset
from recstudio.ann import sampler

from ..basemodel import BaseRetriever
from .. import loss_func, scorer


class CORE(BaseRetriever):

    def _get_dataset_class():
        return SeqDataset

    def _get_query_encoder(self, train_data):
        return super()._get_query_encoder(train_data)

    def _get_item_encoder(self, train_data):
        return super()._get_item_encoder(train_data)

    def _get_score_func(self):
        return scorer.InnerProductScorer()

    def _get_sampler(self, train_data):
        return sampler.UniformSampler(train_data.num_items)