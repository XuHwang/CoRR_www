import torch
from recstudio.data.dataset import AEDataset
from recstudio.model.basemodel import BaseRetriever, Recommender
from recstudio.model.loss_func import SoftmaxLoss
from recstudio.model.module import MLPModule
from recstudio.model.scorer import InnerProductScorer


class RecVAE(BaseRetriever):

    def add_model_specific_args(parent_parser):
        parent_parser = Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('MultiVAE')
        parent_parser.add_argument("--dropout", type=int, default=0.5, help='dropout rate for MLP layers')
        parent_parser.add_argument("--encoder_dims", type=int, nargs='+', default=64, help='MLP layer size for encoder')
        parent_parser.add_argument("--decoder_dims", type=int, nargs='+', default=64, help='MLP layer size for decocer')
        parent_parser.add_argument("--activation", type=str, default='relu', help='activation function for MLP layers')
        parent_parser.add_argument("--anneal_max", type=float, default=1.0, help="max anneal coef for KL loss")
        parent_parser.add_argument("--anneal_total_step", type=int, default=2000, help="total anneal steps")
        return parent_parser

    def _get_dataset_class():
        return AEDataset

    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items, self.embed_dim, 0)

    def _get_query_encoder(self, train_data):
        return super()._get_query_encoder(train_data)
