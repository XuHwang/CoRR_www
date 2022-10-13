from typing import Dict

import torch, math
import torch.nn.functional as F
import recstudio.eval as eval
import recstudio.model as recmodel
from recstudio.ann import sampler
from recstudio.data import dataset
from recstudio.model import basemodel, loss_func, module, init
from recstudio.utils import get_model



class CoRR(basemodel.BaseRanker):
    def __init__(self, config: Dict = None, logger = None, model_list=None, **kwargs):
        super().__init__(config, logger, **kwargs)

        if model_list is not None:  # define with model list
            self._len = len(model_list)
            for i in range(self._len):
                if isinstance(model_list[i], str):  # model name
                    model_class, model_conf = get_model(model_list[i])
                    model_list[i] = model_class(model_conf, logger)
                elif isinstance(model_list[i], basemodel.Recommender):
                    pass
                else:
                    raise TypeError(f"items in model_list are required to be str(name of models) or recstudio.Recommender, but get {type(model_list[i])}.")

            if len(model_list) == 2:
                self.retriever = model_list[0]
                self.scorer = model_list[1]
            elif len(model_list) > 2:
                self.retriever = CoRR(config, logger, model_list[:-1])
                self.scorer = model_list[-1]
            else:
                raise ValueError("CoRR requires at least 2 models as model_list.")

        # elif (retriever is not None) and (scorer is not None):  # define with retriever and scorer
        #     self.retriever = retriever
        #     self.scorer = scorer
        #     self._len = len(retriever) + 1

        else:
            self.retriever = None
            self.scorer = None
            self._len = 0


    def __len__(self):
        return self._len


    def _init_model(self, train_data):
        # super()._init_model(train_data)
        self.retriever._init_model(train_data)
        self._init_model(train_data)
        # TODO: maybe errors happen in members


    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
    #     parent_parser.add_argument_group('CoRR')
    #     parent_parser.add_argument('--num_neg', type=int, nargs='+', default=[100,20], help='number of negative samples')
    #     return parent_parser


    def _init_parameter(self):
        for name, module in self.named_children():
            module.apply(init.xavier_normal_initialization)

    @staticmethod
    def _get_dataset_class():
        return dataset.SeqDataset


    def _set_data_field(self, data):
        data.use_field = set([data.fiid, data.fuid, data.frating])
        #TODO: do not drop feat for first model, get feat union for all models


    def _get_loss_func(self):
        # TODO: loss func should be attached to each model
        return loss_func.SampledSoftmaxLoss()


    def _cal_distill_loss(self, score_re, score_ra, log_prob=None):
        if log_prob is not None:
            score_re = score_re - log_prob.detach()
            score_ra = score_ra - log_prob.detach()
        dist_re = torch.log_softmax(score_re, -1)
        dist_ra = torch.log_softmax(score_ra, -1)
        kl_loss = torch.nn.functional.kl_div(dist_re, dist_ra.detach(), log_target=True, reduction='mean')
        return kl_loss


    def score(self, batch, query=None, neg_id=None):
        if isinstance(self, basemodel.BaseRetriever):
            return self.score(batch, query=query, neg_id=neg_id)
        elif isinstance(self, basemodel.BaseRanker):
            return self.score(batch, neg_id=neg_id)
        else:   # nn.Module
            if neg_id is None:
                return self.scorer(batch)
            else:
                neg_batch = self._generate_neg_batch(batch, neg_id)
                num_neg = neg_id.size(-1)
                return self.scorer(neg_batch).view(-1, num_neg)


    def sampling(self, batch, num_neg, pos_items, method='is', excluding_hist=False, user_hist=None, t=1, eps=1e-9):

        if method in ("none", "is", "dns"):

            (log_pos_prob, neg_id, log_neg_prob), _ = self.retriever.sampling(batch, num_neg[:-1], pos_items, user_hist, method, excluding_hist, t)

            if method == 'none':
                return (log_pos_prob, neg_id, log_neg_prob), None

            elif method == 'is':
                pos_score = self.score(batch)
                neg_score = self.score(batch, neg_id)
                weight_n = neg_score / t - log_neg_prob
                weight_p = pos_score / t - log_pos_prob
                weight = torch.cat((weight_p, weight_n), -1)
                prob = torch.softmax(weight, -1)
                _idx = torch.multinomial(prob, num_neg[-1], replacement=True)
                neg_id = torch.gather(neg_id, -1, _idx)
                neg_prob = torch.gather(prob, -1, _idx) +eps
                pos_prob = prob[:, 0] + eps
                return (pos_prob.log(), neg_id, neg_prob.log()), None

            elif method == 'dns':
                neg_score = self.score(batch, neg_id)
                _, _idx = torch.topk(neg_score, -1)
                neg_id = torch.gather(neg_id, -1, _idx)
                return (None, neg_id, None), None

        elif method == 'toprand':
            _, top_id = self.retriever.topk(batch, 2*num_neg[-1])
            _idx = torch.randint(0, 2*num_neg[-1], size=(top_id.size(0), num_neg[-1]), device=top_id)
            neg_id = torch.gather(top_id, -1, _idx)
            return (None, neg_id, None), None

        elif method == "brute":
            raise NotImplementedError("Not supproted for brute sampling method.")

        else:
            raise ValueError(f"Not supported for sampling method {method}.")


    def forward(self, batch):
        if isinstance(self.retriever, CoRR):
            loss_cascade = self.retriever.forward(batch)
        else:

            # TODO
            # loss_re = xxx
            loss_cascade = None

        pos_id = batch[self.fiid]
        (log_pos_prob, neg_id, log_neg_prob), _ = self.retriever.sampling(batch, self.neg_count, pos_id, method=self.config['sampling_method'])

        if isinstance(self.retriever, basemodel.BaseRetriever):
            query = self.retriever.query_encoder(self.retriever._get_query_feat(batch))
        else:
            query = None

        pos_score_re = self.retriever.score(batch, query)
        neg_score_re = self.retriever.score(batch, query, neg_id)
        # loss_re = self.loss_fn(pos_score_re, log_pos_prob, neg_score_re, log_neg_prob)

        pos_score_ra = self.score(batch)
        neg_score_ra = self.score(batch, neg_id)
        loss_ra = self.loss_fn(pos_score_ra, log_pos_prob, neg_score_ra, log_neg_prob)

        loss_distill = self._cal_distill_loss(neg_score_re, neg_score_ra, log_neg_prob.detach())

        loss = loss_re + loss_distill
        if loss_cascade is not None:
            loss = loss + loss_cascade
        return loss
        # return {'loss': loss, 'loss_re': loss_re.detach(), 'loss_ra': loss_ra.detach(), 'loss_distill': loss_distill.detach()}


    def training_step(self, batch, nepoch):
        loss = self.forward(batch)
        return {'loss': loss}


    def topk(self, batch, topk, user_hist=None, return_retriever_score=False):
        if self.retriever is None:
            raise NotImplementedError("`topk` function not supported for ranker without retriever.")
        else:
            score_re, topk_items_re = self.retriever.topk(batch, topk, user_hist)
            score = self.score(batch, topk_items_re)
            _, sorted_idx = score.sort(dim=-1, descending=True)
            topk_items = torch.gather(topk_items_re, -1, sorted_idx)
            score = torch.gather(score, -1, sorted_idx)

            if return_retriever_score:
                return score, topk_items, score_re, topk_items_re
            else:
                return score, topk_items


    def _update_item_vector(self):
        self.retriever._update_item_vector()


    @torch.no_grad()
    def _test_step(self, batch, metric, cutoffs=None):
        rank_m = eval.get_rank_metrics(metric)
        pred_m = eval.get_pred_metrics(metric)
        bs = batch[self.frating].size(0)
        if len(rank_m) > 0:
            assert cutoffs is not None, 'expected cutoffs for topk ranking metrics.'

        topk = self.config['topk']
        score, topk_items, score_re, topk_items_re = self.topk(batch, topk, batch['user_hist'], True)
        # score, topk_items, score_re, topk_items_re, score_ra, topk_items_ra = self.topk(batch, topk, batch['user_hist'], True)
        if batch[self.fiid].dim() > 1:
            target, _ = batch[self.fiid].sort()
            idx_ = torch.searchsorted(target, topk_items)
            idx_[idx_ == target.size(1)] = target.size(1) - 1
            target = torch.gather(target, 1, idx_)
            label = target == topk_items
            label_re = target == topk_items_re
            pos_rating = batch[self.frating]
        else:
            target = batch[self.fiid].view(-1, 1)
            label = target == topk_items
            label_re = target == topk_items_re
            pos_rating = batch[self.frating].view(-1, 1)

        metric = {f"{name}@{cutoff}": func(label, pos_rating, cutoff) for cutoff in cutoffs for name, func in rank_m}
        metric_re = {f"{name}-re@{cutoff}": func(label_re, pos_rating, cutoff) for cutoff in cutoffs for name, func in rank_m}
        metric.update(metric_re)


        # calculate AUC and loglos
        test_neg_id, _ = self.sampler(score.new_zeros((score.size(0), 1)), 1, user_hist=batch['user_hist'])
        pos_score_ra = self.scorer(batch)
        test_neg_batch = self._generate_neg_batch(batch, test_neg_id)
        test_neg_score_ra = self.scorer(test_neg_batch)
        pred = torch.cat((pos_score_ra, test_neg_score_ra), dim=0)
        target = torch.cat((torch.ones_like(pos_score_ra), torch.zeros_like(test_neg_score_ra)), dim=0)
        auc = torch.mean((pos_score_ra > test_neg_score_ra).float())
        logloss = eval.logloss(pred, target)
        metric['auc-ra'] = auc
        metric['logloss-ra'] = logloss

        query = self.retriever.query_encoder(self.retriever._get_query_feat(batch))
        pos_vec = self.retriever.item_encoder(self.retriever._get_item_feat(batch))
        pos_score_re = self.retriever.score_func(query, pos_vec)
        neg_vec = self.retriever.item_encoder(self.retriever._get_item_feat(test_neg_id))
        test_neg_score_re = self.retriever.score_func(query, neg_vec).squeeze(1)
        pred = torch.cat((pos_score_re, test_neg_score_re), dim=0)
        auc = torch.mean((pos_score_re > test_neg_score_re).float())
        logloss = eval.logloss(pred, target)
        metric['auc-re'] = auc
        metric['logloss-re'] = logloss

        return metric, bs
