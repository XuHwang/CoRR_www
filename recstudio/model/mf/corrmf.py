from collections import OrderedDict

import torch
from recstudio.model import seq
import recstudio.eval as eval
import recstudio.model as recmodel
from recstudio.ann import sampler
from recstudio.data import dataset
from recstudio.model import loss_func
from recstudio.model.module import ctr
from recstudio.model.module.layers import MLPModule, LambdaLayer, HStackLayer

# TODO: fix bugs


class CoRRMF(seq.CoRR):

    def add_model_specific_args(parent_parser):
        parent_parser = recmodel.basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('CoRRMF')
        parent_parser.add_argument("--mlp_layer", type=int, nargs='*', help='MLP layer size for DeepFM')
        parent_parser.add_argument("--activation", type=str, default='relu', help='activation function for MLP')
        parent_parser.add_argument("--dropout", type=float, default=0.3, help='dropout rate for MLP')
        parent_parser.add_argument(
            "--sampler", type=str, choices=['rand', 'midx'],
            default='midx', help='sampler for retriever')
        parent_parser.add_argument(
            "--retrieve_method", type=str, choices=['none', 'toprand', 'top&rand'],
            default='none', help='retrieve method after sampling')
        parent_parser.add_argument("--num_neg", type=int, nargs='+', default=20,
                                   help="number of negative samples to retrieve")
        parent_parser.add_argument("--retriever", type=str, default='mf')
        parent_parser.add_argument("--ranker", type=str, default='deepfm')
        return parent_parser

    def _get_dataset_class():
        return dataset.MFDataset

    def _get_retriever(self, train_data):
        if self.retriever is None:
            if self.config['retriever'].lower() == 'mf':
                retriever = recmodel.mf.BPR(self.config)
            elif self.config['retriever'].lower() == 'dssm':
                retriever = recmodel.mf.DSSM(self.config)
            else:
                raise ValueError(f"Retriever {self.config['retriever']} not supported.")
            retriever._init_model(train_data)
            if 'sampler' not in self.config:
                retriever.sampler = sampler.UniformSampler(train_data.num_items, retriever.score_func)
            else:
                if self.config['sampler'] == 'midx':
                    retriever.sampler = sampler.MIDXSamplerUniform(train_data.num_items, 8, retriever.score_func)
                elif self.config['sampler'] == 'rand':
                    retriever.sampler = sampler.UniformSampler(train_data.num_items, retriever.score_func)
            retriever.loss_fn = loss_func.SampledSoftmaxLoss()
            return retriever

    def _get_scorer(self, train_data):
        embedding = ctr.Embeddings(
            fields=self.fields,
            data=train_data,
            embed_dim=self.embed_dim)

        if self.config['ranker'].lower() == 'deepfm':
            scorer = torch.nn.Sequential(
                embedding,
                HStackLayer(
                    torch.nn.Sequential(ctr.FMLayer(), LambdaLayer(lambda x: x.sum(-1))),
                    torch.nn.Sequential(
                        LambdaLayer(lambda x: x.view(x.size(0), -1)),
                        MLPModule([embedding.num_features*self.embed_dim]+self.config['mlp_layer'], self.config['activation'], self.config['dropout']),
                        torch.nn.Linear(self.config['mlp_layer'][-1], 1),
                        LambdaLayer(lambda x: x.squeeze(-1))
                    )
                ),
                LambdaLayer(lambda x: x[0]+x[1])
            )
        elif self.config['ranker'].lower() == 'dcn':
            scorer = torch.nn.Sequential(OrderedDict({
                'embedding': embedding,
                'flatten': LambdaLayer(lambda x: x.view(*x.shape[:-2], -1)),
                'cross_net': HStackLayer(
                    ctr.CrossNetwork(embedding.num_features * self.embed_dim, self.config['num_layers']),
                    MLPModule(
                        [embedding.num_features * self.embed_dim] + self.config['mlp_layer'],
                        dropout=self.config['dropout'],
                        batch_norm=self.config['batch_norm'])),
                'cat': LambdaLayer(lambda x: torch.cat(x, dim=-1)),
                'fc': torch.nn.Linear(embedding.num_features*self.embed_dim + self.config['mlp_layer'][-1], 1),
                'squeeze': LambdaLayer(lambda x: x.squeeze(-1))
            }))
        return scorer

    @torch.no_grad()
    def _test_step(self, batch, metric, cutoffs=None):
        rank_m = eval.get_rank_metrics(metric)
        bs = batch[self.frating].size(0)
        if len(rank_m) > 0:
            assert cutoffs is not None, 'expected cutoffs for topk ranking metrics.'

        topk = self.config['topk']
        score, topk_items= self.topk(batch, topk, batch['user_hist'], False)
        if batch[self.fiid].dim() > 1:
            target, _ = batch[self.fiid].sort()
            idx_comb = torch.searchsorted(target, topk_items)
            idx_comb[idx_comb == target.size(1)] = target.size(1) - 1
            target_comb = torch.gather(target, 1, idx_comb)
            label = target_comb == topk_items
            pos_rating = batch[self.frating]
        else:
            target = batch[self.fiid].view(-1, 1)
            label = target == topk_items
            pos_rating = batch[self.frating].view(-1, 1)

        metric = {f"{name}@{cutoff}": func(label, pos_rating, cutoff) for cutoff in cutoffs for name, func in rank_m}
        return metric, bs


    def forward(self, batch, retrieve_epoch=False):
        pos_id = batch[self.fiid]
        if 'top' in self.config['retrieve_method']:
            (log_pos_prob, neg_id, log_neg_prob), query = self.retriever.sampling(
                batch, self.config['num_neg'], method='none',
                return_query=True)
        else:
            (log_pos_prob, neg_id, log_neg_prob), query = self.retriever.sampling(
                batch=batch, num_neg=self.config['num_neg'], method=self.config['retrieve_method'],
                return_query=True)

        loss = 0
        if (not self.alternating) or (self.alternating and not retrieve_epoch):
            pos_score_ra = self.scorer(batch)  # B
            neg_batch = self._generate_neg_batch(batch, neg_id)
            neg_score_ra = self.scorer(neg_batch)    # N*B
            neg_score_ra = neg_score_ra.view(pos_id.size(0), -1)  # B x N
            loss_ra = self.loss_fn(None, pos_score_ra, log_pos_prob.detach(), neg_score_ra, log_neg_prob.detach())
            loss = loss + loss_ra

        if (not self.alternating) or (self.alternating and retrieve_epoch):
            neg_score_re = self.retriever.score_func(query, self.retriever.item_encoder(
                self.retriever._get_item_feat(neg_id)))
            pos_score_re = self.retriever.score_func(query, self.retriever.item_encoder(
                self.retriever._get_item_feat(batch)))

            loss_re = self.retriever.loss_fn(
                None, pos_score_re, log_pos_prob.detach(),
                neg_score_re, log_neg_prob.detach())

            if 'top' in self.config['retrieve_method']:
                log_pos_prob, neg_id, log_neg_prob = self.retriever.sampling(
                    batch, self.config['num_neg'], pos_id, method=self.config['retrieve_method'],
                    return_query=False)

                neg_score_re = self.retriever.score_func(query, self.retriever.item_encoder(
                    self.retriever._get_item_feat(neg_id)))

            if self.alternating or ('top' in self.config['retrieve_method']):
                with torch.no_grad():
                    pos_score_ra = self.scorer(batch)  # B
                    neg_batch = self._generate_neg_batch(batch, neg_id)
                    neg_score_ra = self.scorer(neg_batch)    # B*N
                    neg_score_ra = neg_score_ra.view(pos_id.size(0), -1)  # B x N

            loss_distill = self._cal_distill_loss(neg_score_re, neg_score_ra.detach(), log_neg_prob.detach())

            loss = loss + loss_re + loss_distill
        return loss
