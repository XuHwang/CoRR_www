import torch
from collections import OrderedDict
from recstudio.data.dataset import MFDataset

from ..basemodel import BaseRanker
from ..loss_func import BCEWithLogitLoss
from ..module import ctr, LambdaLayer, MLPModule, HStackLayer


class WideDeep(BaseRanker):

    def _get_dataset_class():
        return MFDataset

    def _get_scorer(self, train_data):
        embedding = ctr.Embeddings(
            self.fields,
            self.embed_dim,
            train_data)
        mlp = MLPModule([embedding.num_features * self.embed_dim] + self.config['mlp_layer'],
                        activation_func = self.config['activation'],
                        dropout = self.config['dropout'],
                        batch_norm = self.config['batch_norm'])
        linear = ctr.LinearLayer(self.fields, train_data)
        scorer = torch.nn.Sequential(
            HStackLayer(
                OrderedDict({
                    'wide': linear,
                    'deep': torch.nn.Sequential(
                        embedding,
                        LambdaLayer(lambda x: x.view(*x.shape[:-2], -1)),
                        mlp,
                        torch.nn.Linear(self.config['mlp_layer'][-1], 1),
                        LambdaLayer(lambda x: x.squeeze(-1))
                    )
                })
            ),
            LambdaLayer(lambda x: x[0]+x[1])
        )
        return scorer

    def _get_loss_func(self):
        return BCEWithLogitLoss(threshold=self.rating_threshold)