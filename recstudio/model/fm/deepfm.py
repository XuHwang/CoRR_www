import torch
from collections import OrderedDict
from recstudio.data.dataset import MFDataset

from ..basemodel import BaseRanker
from ..loss_func import BCEWithLogitLoss
from ..module import ctr, LambdaLayer, MLPModule, HStackLayer


class DeepFM(BaseRanker):

    def _set_data_field(self, data):
        data.use_field = set([data.fiid, data.fuid, data.frating])

    def _get_dataset_class():
        return MFDataset

    def _get_scorer(self, train_data):
        embedding = ctr.Embeddings(
            self.fields,
            self.embed_dim,
            train_data,)

        linear = ctr.LinearLayer(self.fields, train_data)

        return torch.nn.Sequential(
            HStackLayer(OrderedDict({
                'linear': linear,
                'fm_mlp': torch.nn.Sequential(
                    embedding,
                    HStackLayer(
                        ctr.FMLayer(reduction='sum'),
                        torch.nn.Sequential(
                            LambdaLayer(lambda x: x.view(x.size(0), -1)),
                            MLPModule([embedding.num_features*self.embed_dim]+self.config['mlp_layer'],
                                      self.config['activation'], self.config['dropout']),
                            torch.nn.Linear(self.config['mlp_layer'][-1], 1),
                            LambdaLayer(lambda x: x.squeeze(-1))
                        )
                    ),
                    LambdaLayer(lambda x: x[0]+x[1])
                )
            })),
            LambdaLayer(lambda x: x[0]+x[1])
        )

    def _get_loss_func(self):
        # return BCEWithLogitLoss(self.rating_threshold)
        from ..loss_func import BinaryCrossEntropyLoss
        return BinaryCrossEntropyLoss()

    def _get_retriever(self, train_data):
        from recstudio.ann.sampler import UniformSampler
        return UniformSampler(train_data.num_items)
