import torch
import torch.optim as optim
import torch.nn.functional as F

from .corrmf import CoRRMF
from ..basemodel import Recommender
from recstudio.data import dataset
from recstudio.model import loss_func

class RankFlowMF(CoRRMF):

    def add_model_specific_args(parent_parser):
        parent_parser = Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('RankFlowMF')
        parent_parser.add_argument("--dropout", type=float, default=0.3, help='dropout rate')
        parent_parser.add_argument("--activation", type=str, default='relu', help='activation for MLP')
        parent_parser.add_argument("--alpha", type=float, default=0.5, help='alpha for tutor learning loss')
        parent_parser.add_argument("--retriever", type=str, default='mf')
        parent_parser.add_argument("--ranker", type=str, default='deepfm')
        return parent_parser


    def _get_dataset_class():
        return dataset.MFDataset


    def _get_retriever(self, train_data):
        retriever = super()._get_retriever(train_data)
        retriever.loss_fn = loss_func.BinaryCrossEntropyLoss()
        return retriever


    def _get_loss_func(self):
        return torch.nn.BCEWithLogitsLoss()


    def _get_optimizers(self):
        opt_ind_re = optim.Adam(
            self.retriever.parameters(),
            lr=0.001,
            weight_decay=self.config['weight_decay'])
        opt_ind_ra = optim.Adam(
            self.scorer.parameters(),
            lr=0.001,
            weight_decay=self.config['weight_decay'])
        opt_re = optim.SGD(
            self.retriever.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'])
        opt_ra = optim.SGD(
            self.scorer.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'])
        return [{'optimizer': opt_ind_re}, {'optimizer': opt_ind_ra}, {'optimizer': opt_re}, {'optimizer': opt_ra}]


    def current_epoch_optimizers(self, nepoch):
        epochs_per_cycle = self.config['every_n_epoch_self'] + self.config['every_n_epoch_tutor']
        if nepoch < self.config['independent_epochs']:
            return self.optimizers[:2]
        else:
            nepoch = nepoch - self.config['independent_epochs']
            if (nepoch % epochs_per_cycle) < self.config['every_n_epoch_self']:
                return self.optimizers[:2]
            else:
                return self.optimizers[2:]


    def _generate_data(self, batch):
        user_hist = batch['user_hist']
        topk_scores, topk_items = self.retriever.topk(batch, self.config['K'][0], user_h=user_hist)
        target, _ = user_hist.sort()
        idx_ = torch.searchsorted(target, topk_items)
        idx_[idx_ == target.size(1)] = target.size(1) - 1
        label = (torch.gather(target, 1, idx_) == topk_items).float()
        return topk_scores, topk_items, label


    def forward(self, batch, self_learning_epoch=True, independent_epoch=False):
        bs = batch[self.fiid].size(0)
        if independent_epoch:
            loss_1 = self.retriever.training_step(batch)
            pos_score = self.scorer(batch)
            _, neg_id, __ = self.retriever.sampler(bs, self.neg_count, batch[self.fiid])
            neg_batch = self._generate_neg_batch(batch, neg_id.detach())
            neg_score = self.scorer(neg_batch).view(bs, -1)
            loss_2 = self.loss_fn(None, pos_score, None, neg_score, None)
            loss = loss_1 + loss_2
        else:
            _, neg_id, __ = self.retriever.sampler(bs, self.neg_count, batch[self.fiid])
            neg_vec = self.retriever.item_encoder(self.retriever._get_item_feat(neg_id))
            query = self.retriever.query_encoder(self.retriever._get_query_feat(batch))
            neg_score_re = self.retriever.score_func(query, neg_vec)


            neg_score_re, topk_items, label = self._generate_data(batch)
            neg_batch = self._generate_neg_batch(batch, topk_items.detach())
            score_ra = self.scorer(neg_batch).view(-1, self.config['K'][0])
            pos_vec = self.retriever.item_encoder(self.retriever._get_item_feat(batch))
            pos_score_re = self.retriever.score_func(query, pos_vec)
            # scores_1, _idx = torch.topk(neg_score_re, k=self.config['K'][0], dim=-1)
            # top_neg_id = torch.gather(neg_id, -1, _idx)
            # scores_1, topk_items_1 = self.retriever.topk(batch, self.config['K'][0], user_h=batch[self.fiid].view(-1,1))
            # neg_batch = self._generate_neg_batch(batch, top_neg_id.detach())
            # neg_score = self.scorer(neg_batch).view(-1, self.config['K'][0])
            if self_learning_epoch:
                # pos_vec = self.retriever.item_encoder(self.retriever._get_item_feat(batch))
                # pos_score_re = self.retriever.score_func(query, pos_vec)
                loss_1 = self.retriever.loss_fn(None, pos_score_re, None, neg_score_re, None)
                # loss_1 = self.retriever.training_step(batch)
                pos_score = self.scorer(batch)
                loss_2 = self.loss_fn(score_ra, label)
                loss = loss_1 + loss_2
            else:
                scores_1 = torch.sigmoid(scores_1)
                neg_score = torch.sigmoid(score_ra)
                l_mse = torch.mean((scores_1 - neg_score.detach()) ** 2)
                _, _idx = torch.topk(neg_score, self.config['K'][1], -1)
                score_1_pos = torch.gather(scores_1, -1, _idx)
                score_1_pos_sum = score_1_pos.sum(dim=-1)
                score_1_neg_sum = scores_1.sum(dim=-1) - score_1_pos_sum
                l_ranking = - F.logsigmoid(score_1_pos_sum / self.config['K'][1] - \
                    score_1_neg_sum / (self.config['K'][0]-self.config['K'][1])).mean()
                # loss = self.config['alpha'] * l_mse + (1-self.config['alpha']) * l_ranking
                loss = l_ranking
        return loss


    def training_step(self, batch, nepoch):
        epochs_per_cycle = self.config['every_n_epoch_self'] + self.config['every_n_epoch_tutor']
        if nepoch < self.config['independent_epochs']:
            loss = self.forward(batch, independent_epoch=True)
            return {'loss': loss, 'independent_loss': loss.detach()}
        else:
            nepoch = nepoch - self.config['independent_epochs']
            if (nepoch % epochs_per_cycle) < self.config['every_n_epoch_self']:
                loss = self.forward(batch, True)
                return {'loss': loss, 'self_learning_loss': loss.detach()}
            else:
                loss = self.forward(batch, False)
                return {'loss': loss, 'tutor_learning_loss': loss.detach()}
