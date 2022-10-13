import torch
from torch import optim
import torch.nn.functional as F
from .corr import CoRR
from recstudio.ann import sampler
from recstudio.data import dataset
from recstudio.model import loss_func

class RankFlow(CoRR):

    def _set_data_field(self, data):
        data.use_field = data.use_field.union(set(['user_hist']))

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
            lr=0.001,
            weight_decay=self.config['weight_decay'])
        opt_ra = optim.SGD(
            self.scorer.parameters(),
            lr=0.001,
            weight_decay=self.config['weight_decay'])
        opt_re_t = optim.SGD(
            self.retriever.parameters(),
            lr=0.0001,
            weight_decay=self.config['weight_decay'])
        opt_ra_t = optim.SGD(
            self.scorer.parameters(),
            lr=0.0001,
            weight_decay=self.config['weight_decay'])
        return [{'optimizer': opt_ind_re}, {'optimizer': opt_ind_ra},
                {'optimizer': opt_re}, {'optimizer': opt_ra},
                {'optimizer': opt_re_t}, {'optimizer': opt_ra_t}]


    def current_epoch_optimizers(self, nepoch):
        epochs_per_cycle = self.config['every_n_epoch_self'] + self.config['every_n_epoch_tutor']
        if nepoch < self.config['independent_epochs']:
            return self.optimizers[:2]
        else:
            nepoch = nepoch - self.config['independent_epochs']
            if (nepoch % epochs_per_cycle) < self.config['every_n_epoch_self']:
                return self.optimizers[2:4]
            else:
                return self.optimizers[4:]


    @staticmethod
    def _get_dataset_class():
        return dataset.SeqDataset


    def _get_retriever(self, train_data):
        retriever = super()._get_retriever(train_data)
        retriever.sampler = sampler.UniformSampler(train_data.num_items, retriever.score_func)
        retriever.loss_fn = loss_func.BinaryCrossEntropyLoss()
        return retriever


    def _get_loss_func(self):
        # return loss_func.BinaryCrossEntropyLoss()
        return torch.nn.BCEWithLogitsLoss()


    def _generate_data(self, batch):
        user_hist = batch['user_hist']
        topk_scores, topk_items = self.retriever.topk(batch, self.config['K'][0], user_h=user_hist)
        label = (topk_items == batch[self.fiid].view(-1,1)).float()
        # target, _ = user_hist.sort()
        # idx_ = torch.searchsorted(target, topk_items)
        # idx_[idx_ == target.size(1)] = target.size(1) - 1
        # label = (torch.gather(target, 1, idx_) == topk_items).float()
        return topk_scores, topk_items, label


    def forward(self, batch, self_learning_epoch=True, independent_epoch=False):
        bs = batch[self.fiid].size(0)
        if independent_epoch:
            loss_1 = self.retriever.training_step(batch)
            pos_score = self.scorer(batch)
            _, neg_id, __ = self.retriever.sampler(bs, self.neg_count, batch[self.fiid])
            neg_batch = self._generate_neg_batch(batch, neg_id)
            neg_score = self.scorer(neg_batch).view(bs, -1)
            score_ra = torch.cat((pos_score.view(-1,1), neg_score), dim=-1)
            label_ra = torch.zeros_like(score_ra)
            label_ra[:, 0] = 1.0
            loss_2 = self.loss_fn(score_ra, label_ra)
            loss = loss_1 + loss_2
        else:
            K = self.config['K']
            _, neg_id, __ = self.retriever.sampler(bs, self.neg_count, batch[self.fiid])
            neg_vec = self.retriever.item_encoder(self.retriever._get_item_feat(neg_id))
            query = self.retriever.query_encoder(self.retriever._get_query_feat(batch))
            neg_scores = self.retriever.score_func(query, neg_vec)

            # neg_score_re, _idx = torch.topk(neg_scores, k=K[0], dim=-1)
            # top_neg_id = torch.gather(neg_id, -1, _idx)

            neg_score_re, topk_items, label = self._generate_data(batch)
            neg_batch = self._generate_neg_batch(batch, topk_items.detach())
            score_ra = self.scorer(neg_batch).view(-1, self.config['K'][0])
            pos_vec = self.retriever.item_encoder(self.retriever._get_item_feat(batch))
            pos_score_re = self.retriever.score_func(query, pos_vec)
            # pos_score_ra = self.scorer(batch)
            if self_learning_epoch:
                loss_1 = self.retriever.loss_fn(None, pos_score_re, None, neg_scores, None)
                # loss_2 = self.loss_fn(None, pos_score_ra, None, neg_score_ra[:, :K[0]], None)
                loss_2 = self.loss_fn(score_ra, label)
                loss = loss_1 + loss_2
            else:
                scores_1 = torch.sigmoid(neg_score_re)
                scores_2 = torch.sigmoid(score_ra)
                l_mse = torch.mean((scores_1 - scores_2.detach()) ** 2)
                _, _idx = torch.topk(scores_2, self.config['K'][1], -1)
                score_1_pos = torch.gather(scores_1, -1, _idx)
                score_1_pos_sum = score_1_pos.sum(dim=-1)
                score_1_neg_sum = scores_1.sum(dim=-1) - score_1_pos_sum
                l_ranking = - F.logsigmoid(score_1_pos_sum/K[1] - score_1_neg_sum/(K[0]-K[1]+1)).mean()
                loss = self.config['alpha'] * l_mse + (1-self.config['alpha']) * l_ranking
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
