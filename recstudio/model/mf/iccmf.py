import torch
from collections import OrderedDict
from typing import Dict
from ..module import ctr
from ..module.layers import HStackLayer, LambdaLayer, MLPModule
from ..basemodel import BaseRanker, Recommender
from .bpr import BPR
from .. import loss_func, scorer, init, module
from recstudio.data import MFDataset
from recstudio.ann import sampler
import recstudio.eval as eval


class ICCMF(BaseRanker):


    def __init__(self, config: Dict = None, **kwargs):
        super().__init__(config, **kwargs)
        self.retriever = kwargs.get('retriever', None)
        self.scorer = kwargs.get('scorer', None)
        self.alternating = self.config.get('alternating', False)
        self.alternating = kwargs.get('alternating', self.alternating)


    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.sampler = sampler.MaskedUniformSampler(train_data.num_items)


    def add_model_specific_args(parent_parser):
        parent_parser = Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('ICCMF')
        parent_parser.add_argument("--dropout", type=float, default=0.3, help='dropout rate')
        parent_parser.add_argument("--activation", type=str, default='relu', help='activation for MLP')
        return parent_parser


    def _init_parameter(self):
        for name, module in self.named_children():
            if isinstance(module, Recommender):
                module._init_parameter()
            else:
                module.apply(init.xavier_normal_initialization)


    @staticmethod
    def _get_dataset_class():
        return MFDataset


    def _set_data_field(self, data):
        data.use_field = set([data.fiid, data.fuid, data.frating])


    def _get_retriever(self, train_data):
        if self.retriever is None:
            retriever = BPR(self.config)
            retriever._init_model(train_data)
            retriever.loss_fn = loss_func.BinaryCrossEntropyLoss()
            return retriever


    def _get_loss_func(self):
        class ICCLoss(torch.nn.Module):
            def __init__(self, sigma):
                super(ICCLoss, self).__init__()
                self.bce_loss_func = torch.nn.BCELoss()
                self.sigma = sigma

            def forward(self, model_preds, label):
                indicators = None
                ks = model_preds.min(dim=1)[0]
                for i in range(model_preds.size(1)):
                    k = torch.tanh(torch.tensor([i])).to(model_preds.device)
                    # k = torch.tensor([i]).to(model_preds.device)
                    indicator = torch.sigmoid((model_preds[:, i] - k)/self.sigma).unsqueeze(1)
                    indicators = indicator if indicators is None else torch.cat((indicators, indicator), dim=1)
                weights = torch.cumprod(torch.cat((torch.ones((indicators.size(0), 1, indicators.size(2))).to(indicators.device), indicators), dim=1), dim=1)[:, :-1]
                weights = (1 - indicators) * weights
                score = torch.sum(model_preds * weights, dim=1)
                loss = self.bce_loss_func(score, label)
                return loss
        return ICCLoss(sigma=self.config['sigma'])


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


    def forward(self, batch, self_learning_epoch=True, independent_epoch=False):
        bs = batch[self.fiid].size(0)
        _, neg_item_idx, __, query = self.retriever._sample(
                batch,
                neg = self.neg_count,
            )
        # retriever
        pos_vec = self.retriever.item_encoder(self.retriever._get_item_feat(batch))
        neg_vec = self.retriever.item_encoder(self.retriever._get_item_feat(neg_item_idx))
        pos_score_re = self.retriever.score_func(query, pos_vec)
        neg_score_re = self.retriever.score_func(query, neg_vec)

        # ranker
        pos_score_ra = self.scorer(batch)
        neg_batch = self._generate_neg_batch(batch, neg_item_idx)
        neg_score_ra = self.scorer(neg_batch).view(bs, -1)

        score_re = torch.cat((pos_score_re.view(-1,1), neg_score_re), dim=-1)
        score_ra = torch.cat((pos_score_ra.view(-1,1), neg_score_ra), dim=-1)
        scores = torch.sigmoid(torch.stack((score_re, score_ra), dim=1))
        labels = scores.new_zeros((scores.size(0), scores.size(2)))
        labels[:, 0] = 1
        return scores, labels


    def training_step(self, batch):
        scores, labels = self.forward(batch)
        loss = self.loss_fn(scores, labels)
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
                return score, topk_items, score_re, topk_items_re
            else:
                return score, topk_items


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
            # label_ra = target == topk_items_ra
            pos_rating = batch[self.frating]
        else:
            target = batch[self.fiid].view(-1, 1)
            label = target == topk_items
            label_re = target == topk_items_re
            # label_ra = target == topk_items_ra
            pos_rating = batch[self.frating].view(-1, 1)

        metric = {f"{name}@{cutoff}": func(label, pos_rating, cutoff) for cutoff in cutoffs for name, func in rank_m}
        metric_re = {f"{name}-re@{cutoff}": func(label_re, pos_rating, cutoff) for cutoff in cutoffs for name, func in rank_m}
        metric.update(metric_re)

        # calculate AUC and logloss
        pos_id = batch[self.fiid]
        if pos_id.dim() == 1:
            pos_id = pos_id.unsqueeze(1)
        test_neg_id, _ = self.sampler(score.new_zeros((score.size(0), 1)), pos_id.size(1), user_hist=batch['user_hist'])
        pos_batch = self._generate_neg_batch(batch, pos_id)
        pos_score_ra = self.scorer(pos_batch).view(bs, pos_id.size(1))
        test_neg_batch = self._generate_neg_batch(batch, test_neg_id)
        test_neg_score_ra = self.scorer(test_neg_batch).view(bs, test_neg_id.size(1))
        pred = torch.cat((pos_score_ra, test_neg_score_ra), dim=0)
        target = torch.cat((torch.ones_like(pos_score_ra), torch.zeros_like(test_neg_score_ra)), dim=0)
        auc = torch.mean((pos_score_ra > test_neg_score_ra).float())
        logloss = eval.logloss(pred, target)
        metric['auc-ra'] = auc
        metric['logloss-ra'] = logloss

        query = self.retriever.query_encoder(self.retriever._get_query_feat(batch))
        pos_vec = self.retriever.item_encoder(self.retriever._get_item_feat(pos_id))
        pos_score_re = self.retriever.score_func(query, pos_vec)
        neg_vec = self.retriever.item_encoder(self.retriever._get_item_feat(test_neg_id))
        test_neg_score_re = self.retriever.score_func(query, neg_vec)
        pred = torch.cat((pos_score_re, test_neg_score_re), dim=0)
        auc = torch.mean((pos_score_re > test_neg_score_re).float())
        target = torch.cat((torch.ones_like(pos_score_re), torch.zeros_like(test_neg_score_re)), dim=0)
        logloss = eval.logloss(pred, target)
        metric['auc-re'] = auc
        metric['logloss-re'] = logloss

        return metric, bs