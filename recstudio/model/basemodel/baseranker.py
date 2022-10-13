import torch
import inspect
import recstudio.eval as eval
from recstudio.ann import sampler
from .recommender import Recommender
from .baseretriever import BaseRetriever
from recstudio.ann.sampler import UniformSampler
from typing import Dict, List, Optional, Tuple, Union


class BaseRanker(Recommender):

    def _init_model(self, train_data, drop_unused_field=True):
        self.rating_threshold = train_data.config.get('ranker_rating_threshold', 0)
        super()._init_model(train_data, drop_unused_field)
        self.fiid = train_data.fiid
        self.fuid = train_data.fuid
        self.scorer = self._get_scorer(train_data)
        if self.retriever is None:
            self.retriever = self._get_retriever(train_data)
        if self.retriever is None:
            self.logger.warning('No retriever is used, topk metrics is not supported.')

    def _set_data_field(self, data):
        data.use_field = data.field - set([data.ftime])

    def _get_retriever(self, train_data):
        return None

    def _get_scorer(self, train_data):
        return None

    def _generate_neg_batch(self, batch, neg_id):
        num_neg = neg_id.size(-1)
        neg_id = neg_id.view(-1)
        neg_items = self._get_item_feat(neg_id)
        if isinstance(neg_items, torch.Tensor): # only id
            neg_items = {self.fiid: neg_items}  # convert to dict
        neg_batch = {}

        for k, v in batch.items():
            if (k not in neg_items):
                neg_batch[k] = v.unsqueeze(1).expand(-1, num_neg,
                                *tuple([-1 for i in range(len(v.shape)-1)]))
                neg_batch[k] = neg_batch[k].reshape(-1, *(v.shape[1:]))
            else:
                neg_batch[k] = neg_items[k]
        return neg_batch

    def forward(self, batch):
        # calculate scores
        if self.retriever is None:
            score = self.scorer(batch)
            if isinstance(score, Dict):
                score = score['score']['pos_score']
            return {'pos_score': score, 'label': batch[self.frating]}
        else:
            # only positive samples in batch
            assert self.neg_count is not None, 'expecting neg_count is not None.'
            if isinstance(self.retriever, UniformSampler):
                bs = batch[self.fiid].size(0)
                pos_prob, neg_item_idx, neg_prob = self.retriever(bs, self.neg_count, batch[self.fiid])
            else:
                pos_prob, neg_item_idx, neg_prob = self.retriever.sampling(
                    batch, self.neg_count, method=self.config['retrieve_method'])
            pos_score = self.scorer(batch).view(-1, 1)

            neg_batch = self._generate_neg_batch(batch, neg_item_idx)
            neg_score = self.scorer(neg_batch).view(-1, self.neg_count)
            return {'pos_score': pos_score, 'log_pos_prob': pos_prob, 'neg_score': neg_score,
                    'log_neg_prob': neg_prob, 'label': batch[self.frating]}

    def score(self, batch, neg_id=None):
        # designed for cascade algorithm like RankFlow and CoRR
        if neg_id is not None:
            neg_batch = self._generate_neg_batch(batch, neg_id)
            num_neg = neg_id.size(-1)
            return self.scorer(neg_batch).view(-1, num_neg)
        else:
            return self.scorer(batch)

    def build_index(self):
        raise NotImplementedError("build_index for ranker not implemented now.")

    def training_step(self, batch):
        y_h = self.forward(batch)
        loss = self.loss_fn(**y_h)
        return loss

    def validation_step(self, batch):
        eval_metric = self.config['val_metrics']
        if self.config['cutoff'] is not None:
            cutoffs = self.config['cutoff'] if isinstance(self.config['cutoff'], List) \
                else [self.config['cutoff']]
        else:
            cutoffs = None
        return self._test_step(batch, eval_metric, cutoffs)

    def test_step(self, batch):
        eval_metric = self.config['test_metrics']
        if self.config['cutoff'] is not None:
            cutoffs = self.config['cutoff'] if isinstance(self.config['cutoff'], List) \
                else [self.config['cutoff']]
        else:
            cutoffs = None
        return self._test_step(batch, eval_metric, cutoffs)

    def topk(self, batch, topk, user_hist=None, return_retriever_score=False):
        if (self.retriever is None) and (not isinstance(self.scorer, BaseRetriever)):
            raise NotImplementedError("`topk` function not supported for ranker without retriever.")

        elif (self.retriever is None) and (isinstance(self.scorer, BaseRetriever)):
            score, topk_items = self.scorer.topk(batch, topk, user_hist)
            return score, topk_items

        elif isinstance(self.retriever, sampler.Sampler):
            if not isinstance(self.scorer, BaseRetriever):
                bs = batch[self.fiid].size(0)
                step = self.config['topk_item_step']
                num_items = self.retriever.num_items
                # self.logger.warning("all the items will be scored with the scorer, which will be time-consuming.")
                scores = []
                for i in range(1, num_items+1, step):
                    item_ids = torch.arange(i, min(i+step, num_items+1), device=self._parameter_device, dtype=torch.long)
                    item_ids = item_ids.squeeze_(0).repeat(bs, 1)
                    new_batch = self._generate_neg_batch(batch, item_ids)
                    _score = self.scorer(new_batch).view(-1, item_ids.size(-1))
                    scores.append(_score)
                scores = torch.cat(scores, dim=-1)
                score, topk_items = torch.topk(scores, topk, -1)
                topk_items = topk_items + 1
            else:
                score, topk_items = self.topk(batch, topk, user_hist)
            return score, topk_items

        else:
            score_re, topk_items_re = self.retriever.topk(batch, topk, user_hist)
            topk_batch = self._generate_neg_batch(batch, topk_items_re)
            score = self.scorer(topk_batch).view(-1, topk)
            _, sorted_idx = score.sort(dim=-1, descending=True)
            topk_items = torch.gather(topk_items_re, -1, sorted_idx)
            score = torch.gather(score_re, -1, sorted_idx)
            if return_retriever_score:
                return score, topk_items, score_re, topk_items_re
            else:
                return score, topk_items

    def _test_step(self, batch, metric, cutoffs=None):
        rank_m = eval.get_rank_metrics(metric)
        pred_m = eval.get_pred_metrics(metric)
        bs = batch[self.frating].size(0)
        if len(rank_m) > 0:
            assert cutoffs is not None, 'expected cutoffs for topk ranking metrics.'
            assert (not self.config['fmeval']), 'expected the fmeval to be False when ' \
                'rank metrics are required.'

        result = self.forward(batch)
        pred_metric = {n: f(result['pos_score'], (result['label'] > self.rating_threshold).float())
                        for n, f in pred_m}
        if (self.retriever is None) and (not isinstance(self.scorer, BaseRetriever)):
            if len(rank_m) > 0:
                self.logger.warning("`retriever` is None, only precision metrics are calculated.")
            # result = self.forward(batch)
            # result = [f(result['pos_score'], (result['label'] > self.rating_threshold)) for n, f in pred_m], bs
            return pred_metric, bs
        else:
            # TODO(@AngusHuang17): both pred_m and rank_m are required.
            topk = self.config['topk']
            score, topk_items = self.topk(batch, topk, batch['user_hist'])
            if batch[self.fiid].dim() > 1:
                target, _ = batch[self.fiid].sort()
                idx_ = torch.searchsorted(target, topk_items)
                idx_[idx_ == target.size(1)] = target.size(1) - 1
                label = torch.gather(target, 1, idx_) == topk_items
                pos_rating = batch[self.frating]
            else:
                label = batch[self.fiid].view(-1, 1) == topk_items
                pos_rating = batch[self.frating].view(-1, 1)
            rank_metric = {f"{name}@{cutoff}": func(label, pos_rating, cutoff) for cutoff in cutoffs for name, func in rank_m}
            rank_metric.update(pred_metric)
            return rank_metric, bs

    def _update_item_vector(self):
        if self.retriever is not None:
            if hasattr(self.retriever, '_update_item_vector'):
                self.retriever._update_item_vector()

        if isinstance(self.scorer, BaseRetriever):
            if hasattr(self.scorer, '_update_item_vector'):
                self.scorer._update_item_vector()
