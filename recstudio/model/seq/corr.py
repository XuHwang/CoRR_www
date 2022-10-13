from typing import Dict

import torch
import recstudio.eval as eval
import recstudio.model as recmodel
from recstudio.ann import sampler
from recstudio.data import dataset
from recstudio.model import basemodel, loss_func, module, init

# TODO: fix bugs
class CoRR(basemodel.BaseRanker):
    def __init__(self, config: Dict = None, **kwargs):
        super().__init__(config, **kwargs)
        self.retriever = kwargs.get('retriever', None)
        self.scorer = kwargs.get('scorer', None)
        self.alternating = self.config.get('alternating', False)
        self.alternating = kwargs.get('alternating', self.alternating)

    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.tst_sampler = sampler.MaskedUniformSampler(train_data.num_items)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('CoRR')
        parent_parser.add_argument('--retriever', type=str, choices=['SASRec', 'Caser'], default='SASRec')
        parent_parser.add_argument('--ranker', type=str, choices=['DIN', 'BST'], default='BST')
        parent_parser.add_argument('--sampler', type=str, choices=['midx', 'rand'], default='midx')
        parent_parser.add_argument('--retrieve_method', type=str, choices=['none', 'toprand', 'top&rand'], default='none')
        parent_parser.add_argument('--num_neg', type=int, nargs='+', default=[100,20], help='number of negative samples')
        parent_parser.add_argument('--without_kl', action='store_true', help="whether to use kl-loss")
        return parent_parser

    def _init_parameter(self):
        for name, module in self.named_children():
            # if name not in ['sampler', 'retriever']:
            if isinstance(module, basemodel.Recommender):
                module._init_parameter()
            else:
                module.apply(init.xavier_normal_initialization)

    @staticmethod
    def _get_dataset_class():
        return dataset.SeqDataset


    def _set_data_field(self, data):
        data.use_field = set([data.fiid, data.fuid, data.frating])


    def _get_retriever(self, train_data):
        if self.retriever is None:
            if self.config['retriever'].lower() == 'sasrec':
                retriever = recmodel.seq.SASRec(self.config)
            else:
                retriever = recmodel.seq.Caser(self.config)
            # retriever = recmodel.seq.GRU4Rec(self.config)
            # retriever = recmodel.seq.FPMC(self.config)
            retriever._init_model(train_data)
            if 'sampler' not in self.config:
                retriever.sampler = sampler.UniformSampler(train_data.num_items, retriever.score_func)
            else:
                if self.config['sampler'] == 'midx':
                    retriever.sampler = sampler.MIDXSamplerUniform(train_data.num_items-1, 8, retriever.score_func)
                elif self.config['sampler'] == 'rand':
                    retriever.sampler = sampler.UniformSampler(train_data.num_items, retriever.score_func)
            retriever.loss_fn = loss_func.SampledSoftmaxLoss()
            return retriever


    def _get_loss_func(self):
        return loss_func.SampledSoftmaxLoss()


    def _get_scorer(self, train_data):
        if self.config['ranker'].lower() == 'din':
            return module.ctr.DINScorer(
                    train_data.fuid, train_data.fiid, train_data.num_users, train_data.num_items,
                    self.config['din_embed_dim'], self.config['attention_mlp'], self.config['fc_mlp'],
                    activation=self.config['din_activation'], batch_norm=self.config['din_bn'],
                    dropout=self.config['din_dropout_rate'])
        else:
            return module.ctr.BehaviorSequenceTransformer(
                        train_data.fuid, train_data.fiid, train_data.num_users,
                        train_data.num_items, train_data.config['max_seq_len'],
                        self.embed_dim, self.config['hidden_size'], self.config['layer_num'],
                        self.config['head_num'], self.config['dropout_rate'])


    def _cal_distill_loss(self, score_re, score_ra, log_prob=None):
        if log_prob is not None:
            score_re = score_re - log_prob
            score_ra = score_ra - log_prob
        dist_re = torch.log_softmax(score_re, -1)
        dist_ra = torch.log_softmax(score_ra, -1)
        kl_loss = torch.nn.functional.kl_div(dist_re, dist_ra.detach(), \
            log_target=True, reduction='batchmean')
        return kl_loss


    def forward(self, batch, retrieve_epoch=False):
        pos_id = batch[self.fiid]
        if 'top' in self.config['retrieve_method']:
            (log_pos_prob, neg_id, log_neg_prob), query = self.retriever.sampling(\
                batch, self.config['num_neg'], method='none', \
                return_query=True)
        else:
            (log_pos_prob, neg_id, log_neg_prob), query = self.retriever.sampling(\
                batch, self.config['num_neg'], method=self.config['retrieve_method'], \
                return_query=True)

        loss = 0
        if (not self.alternating) or (self.alternating and not retrieve_epoch):
            pos_score_ra = self.scorer(batch) # B
            neg_batch = self._generate_neg_batch(batch, neg_id)
            neg_score_ra = self.scorer(neg_batch)    # N*B
            neg_score_ra = neg_score_ra.view(pos_id.size(0), -1) # B x N
            loss_ra = self.loss_fn(None, pos_score_ra, log_pos_prob.detach(), neg_score_ra, log_neg_prob.detach())
            loss = loss + loss_ra

        if (not self.alternating) or (self.alternating and retrieve_epoch):
            neg_score_re = self.retriever.score_func(query, self.retriever.item_encoder(self.retriever._get_item_feat(neg_id)))
            pos_score_re = self.retriever.score_func(query, self.retriever.item_encoder(self.retriever._get_item_feat(batch)))

            # loss_re = self.loss_fn(None, pos_score_re, torch.zeros_like(pos_score_re), neg_score_re, torch.zeros_like(neg_score_re))
            loss_re = self.retriever.loss_fn(None, pos_score_re, log_pos_prob.detach(), neg_score_re, log_neg_prob.detach())

            # For top-based sampling method, sampling for ssl and kl loss should be different.
            # sampling for ssl is set as uniform sampling
            if 'top' in self.config['retrieve_method']:
                (log_pos_prob, neg_id, log_neg_prob), _ = self.retriever.sampling(\
                    batch, self.config['num_neg'], method=self.config['retrieve_method'], \
                    return_query=False)

                neg_score_re = self.retriever.score_func(query, self.retriever.item_encoder(self.retriever._get_item_feat(neg_id)))


            if self.alternating or ('top' in self.config['retrieve_method']):
                with torch.no_grad():
                    pos_score_ra = self.scorer(batch) # B
                    neg_batch = self._generate_neg_batch(batch, neg_id)
                    neg_score_ra = self.scorer(neg_batch)    # B*N
                    neg_score_ra = neg_score_ra.view(pos_id.size(0), -1) # B x N

            if not self.config['without_kl']:
                loss_distill = self._cal_distill_loss(neg_score_re, neg_score_ra.detach(), log_neg_prob.detach())
                loss = loss + loss_re + loss_distill
            else:
                loss = loss + loss_re
        return loss


    def training_step(self, batch, nepoch):
        if self.alternating:
            epochs_per_cycle = self.config['every_n_epoch_retriever'] + self.config['every_n_epoch_ranker']
            if (nepoch % epochs_per_cycle) < self.config['every_n_epoch_ranker']:
                loss = self.forward(batch, False)
                return {'loss': loss, 'ranker_loss': loss.detach()}
            else:
                loss = self.forward(batch, True)
                return {'loss': loss, 'retriever_loss': loss.detach()}
        else:
            loss = self.forward(batch)
            return {"loss": loss}


    def topk(self, batch, topk, user_hist=None, return_retriever_score=False, return_ranker_result=False):
        if self.retriever is None:
            raise NotImplementedError("`topk` function not supported for ranker without retriever.")
        else:
            score_re, topk_items_re = self.retriever.topk(batch, topk, user_hist)
            topk_batch = self._generate_neg_batch(batch, topk_items_re)
            score = self.scorer(topk_batch).view(-1, topk)
            _, sorted_idx = score.sort(dim=-1, descending=True)
            topk_items = torch.gather(topk_items_re, -1, sorted_idx)
            score = torch.gather(score_re, -1, sorted_idx)

            if return_retriever_score:
                # return score, topk_items, score_re, topk_items_re, score_ra, topk_items_ra
                return score, topk_items, score_re, topk_items_re
            else:
                return score, topk_items

    def _update_item_vector(self):
        self.retriever._update_item_vector()
        self.retriever.sampler.update(self.retriever.item_vector)


    @torch.no_grad()
    def _test_step(self, batch, metric, cutoffs=None):
        rank_m = eval.get_rank_metrics(metric)
        bs = batch[self.frating].size(0)
        if len(rank_m) > 0:
            assert cutoffs is not None, 'expected cutoffs for topk ranking metrics.'

        topk = self.config['topk']
        _, topk_items = self.topk(batch, topk, batch['user_hist'], False)
        # score, topk_items, score_re, topk_items_re, score_ra, topk_items_ra = self.topk(batch, topk, batch['user_hist'], True)
        if batch[self.fiid].dim() > 1:
            target, _ = batch[self.fiid].sort()
            idx_ = torch.searchsorted(target, topk_items)
            idx_[idx_ == target.size(1)] = target.size(1) - 1
            target = torch.gather(target, 1, idx_)
            label = target == topk_items
            pos_rating = batch[self.frating]
        else:
            target = batch[self.fiid].view(-1, 1)
            label = target == topk_items
            pos_rating = batch[self.frating].view(-1, 1)

        metric = {f"{name}@{cutoff}": func(label, pos_rating, cutoff) for cutoff in cutoffs for name, func in rank_m}


        return metric, bs
