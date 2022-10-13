import torch
from collections import OrderedDict
from recstudio.data.dataset import MFDataset

from ..basemodel import BaseRanker
from ..loss_func import BCEWithLogitLoss
from ..module import ctr, LambdaLayer, MLPModule, HStackLayer


class DCN(BaseRanker):

    def _set_data_field(self, data):
        data.use_field = set([data.fiid, data.fuid, data.frating])

    def _get_dataset_class():
        return MFDataset

    def _get_scorer(self, train_data):
        embedding = ctr.Embeddings(
            self.fields,
            self.embed_dim,
            train_data)
        return torch.nn.Sequential(OrderedDict({
            'embedding': embedding,
            'flatten': LambdaLayer(lambda x: x.view(*x.shape[:-2], -1)),
            'cross_net': HStackLayer(
                ctr.CrossNetwork(embedding.num_features * self.embed_dim, self.config['num_layers']),
                MLPModule(
                    [embedding.num_features * self.embed_dim] + self.config['mlp_layer'],
                    dropout=self.config['dropout'],
                    batch_norm=self.config['batch_norm'])),
            'cat': LambdaLayer(lambda x: torch.cat(x, dim=-1)),
            'fc': torch.nn.Linear(embedding.num_features*self.embed_dim + self.config['mlp_layer'][-1], 1),
            'squeeze': LambdaLayer(lambda x: x.squeeze(-1))
        }))

    def _get_loss_func(self):
        # return BCEWithLogitLoss(self.rating_threshold)
        from ..loss_func import BinaryCrossEntropyLoss
        return BinaryCrossEntropyLoss()

    def _get_retriever(self, train_data):
        from recstudio.ann.sampler import UniformSampler
        return UniformSampler(train_data.num_items)
