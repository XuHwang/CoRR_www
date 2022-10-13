import torch.optim as optim

from .rankflowmf import RankFlowMF
from ..basemodel import Recommender
from recstudio.ann import sampler
from recstudio.data import dataset

class IndependentMF(RankFlowMF):
    
    def add_model_specific_args(parent_parser):
        parent_parser = Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('IndependentMF')
        parent_parser.add_argument("--dropout", type=float, default=0.3, help='dropout rate')
        parent_parser.add_argument("--activation", type=str, default='relu', help='activation for MLP')
        parent_parser.add_argument("--retriever", type=str, default='mf')
        parent_parser.add_argument("--ranker", type=str, default='deepfm')
        return parent_parser


    def _get_dataset_class():
        return dataset.MFDataset


    def _get_optimizers(self):
        opt_re = optim.Adam(
            self.retriever.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'])
        opt_ra = optim.Adam(
            self.scorer.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'])
        return [{'optimizer': opt_re}, {'optimizer': opt_ra}]


    def current_epoch_optimizers(self, nepoch):
        return self.optimizers


    def forward(self, batch):
        bs = batch[self.fiid].size(0)
        loss_1 = self.retriever.training_step(batch)
        pos_score = self.scorer(batch)
        _, neg_id, __ = self.retriever.sampler(bs, self.config['ranker_negative_count'], batch[self.fiid])
        neg_batch = self._generate_neg_batch(batch, neg_id.detach())
        neg_score = self.scorer(neg_batch).view(bs, -1)
        loss_2 = self.loss_fn(None, pos_score, None, neg_score, None)
        loss = loss_1 + loss_2
        return loss


    def training_step(self, batch):
        loss = self.forward(batch)
        return {'loss': loss, 'independent_loss': loss.detach()}
